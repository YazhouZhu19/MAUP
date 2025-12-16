import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from transformers import Dinov2Model


class FewShotSeg(nn.Module):
    
    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()
        self.device = torch.device('cuda')
        self.scaler = 5.0
        self.fg_sampler = np.random.RandomState(1289)
        
        # MAUP parameters
        self.Nf = 30
        self.Nmin = 3
        self.Nmax = 10
        self.gamma = 0.5
        self.tau_percentile = 95
        self.n_uncertainty_points = 2
        self.dilation_radius = 8

        # Load SAM (SAM-ViT-H)
        self.SAM = sam_model_registry['vit_h'](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
        self.SAM = self.SAM.eval()
        self.SAM_Encoder = self.SAM.image_encoder.eval()
        self.SAM_Predictor = SamPredictor(self.SAM)
        
        # Load DINOv2 (DINOv2-ViT-L/14)
        self.dinov2 = Dinov2Model.from_pretrained('./dinov2_weights')
        self.dinov2.eval()

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.img_size = supp_imgs[0][0].shape[-2:]
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask], dim=0).view(
            supp_bs, self.n_ways, self.n_shots, *img_size)

        # Feature extraction with DINOv2
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + 
                                [torch.cat(qry_imgs, dim=0)], dim=0)
        
        with torch.no_grad():
            outputs_dinov2 = self.dinov2(pixel_values=imgs_concat)
        
        fts_dinov2 = outputs_dinov2.last_hidden_state[:, 1:]
        B, N, C = fts_dinov2.shape
        H = W = int(N ** 0.5)
        fts_dinov2 = fts_dinov2.transpose(1, 2).reshape(B, C, H, W)
        fts_dinov2_upsampled = F.interpolate(fts_dinov2, size=(64, 64), mode='bilinear', align_corners=True)
        
        supp_embedding_dinov2 = fts_dinov2_upsampled[:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *fts_dinov2_upsampled.shape[-2:])
        qry_embedding_dinov2 = fts_dinov2_upsampled[self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *fts_dinov2_upsampled.shape[-2:])

        # Compute periphery mask (Eq. 3)
        kernel = self._create_circular_kernel(self.dilation_radius)
        supp_mask_np = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_np.astype(np.uint8), kernel, iterations=1)
        supp_periphery_mask = supp_dilated_mask - supp_mask_np
        
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots, *img_size))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots, *img_size))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()

        outputs = []
        for epi in range(supp_bs):
            supp_embedding = supp_embedding_dinov2
            qry_embedding = qry_embedding_dinov2

            # Regional Prototype Generation
            regional_prototypes = self.regional_prototype_generation(
                supp_embedding[[epi], 0, 0],
                supp_mask[[epi], 0, 0]
            )

            # Similarity map computation (Eq. 2)
            similarity_maps = self.compute_similarity_maps(
                qry_embedding[epi, 0],
                regional_prototypes[0]
            )

            # Periphery similarity map
            periphery_prototype = self.compute_periphery_prototype(
                supp_embedding[[epi], 0, 0],
                supp_periphery_mask[[epi], 0, 0]
            )
            periphery_sim_map = self.compute_single_similarity_map(
                qry_embedding[epi, 0],
                periphery_prototype[0]
            )

            # MAUP prompting
            input_points, point_labels = self.MAUP(similarity_maps, periphery_sim_map)

            # SAM prediction
            qry_img = qry_imgs[0]
            qry_img_sam = self.preprocess_image_for_sam(qry_img)
            
            with torch.no_grad():
                self.SAM_Predictor.set_image(qry_img_sam)
                mask, score, logit = self.SAM_Predictor.predict(
                    point_coords=input_points,
                    point_labels=point_labels,
                    multimask_output=False
                )
            best_mask = mask

            best_mask_t = torch.from_numpy(best_mask.astype(bool)).float().unsqueeze(0).cuda()
            preds = torch.cat((1.0 - best_mask_t, best_mask_t), dim=1)
            outputs.append(preds)

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *outputs[0].shape[1:])
        return output

    def _create_circular_kernel(self, radius):
        """Create circular structuring element."""
        diameter = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        return kernel

    def regional_prototype_generation(self, supp_fts, supp_mask):
        """RPG module - Voronoi partition + MAP (Eq. 1)."""
        B, C, h, w = supp_fts.shape
        
        fg_mask = F.interpolate(
            supp_mask.unsqueeze(1).float(), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=True
        ).squeeze(1) > 0.5
        
        batch_prototypes = []
        
        for b in range(B):
            fg_mask_b = fg_mask[b]
            fts_b = supp_fts[b]
            fg_coords = torch.stack(torch.where(fg_mask_b), dim=1)
            
            if fg_coords.shape[0] < self.Nf:
                n_repeat = (self.Nf // max(1, fg_coords.shape[0])) + 1
                fg_coords = fg_coords.repeat(n_repeat, 1)[:self.Nf]
            
            centers = self._farthest_point_sampling(fg_coords, self.Nf)
            dist = fg_coords.float().reshape(-1, 1, 2) - centers.float().reshape(1, -1, 2)
            region_labels = torch.argmin((dist ** 2).sum(-1), dim=1)
            fg_features = fts_b.permute(1, 2, 0)[fg_mask_b]
            
            prototypes = []
            for i in range(self.Nf):
                region_mask = (region_labels == i)
                if region_mask.sum() > 0:
                    proto = fg_features[region_mask].mean(0)
                else:
                    proto = fg_features.mean(0)
                prototypes.append(proto)
            
            prototypes = torch.stack(prototypes, dim=0)
            batch_prototypes.append(prototypes)
        
        return torch.stack(batch_prototypes, dim=0)

    def _farthest_point_sampling(self, points, n_samples):
        """Farthest Point Sampling for Voronoi centers."""
        N = points.shape[0]
        centers = []
        
        idx = self.fg_sampler.choice(N)
        centers.append(points[idx])
        
        for _ in range(n_samples - 1):
            dist = points.float().reshape(-1, 1, 2) - torch.stack(centers, dim=0).float().reshape(1, -1, 2)
            min_dist = (dist ** 2).sum(-1).min(1)[0]
            idx = torch.argmax(min_dist)
            centers.append(points[idx])
        
        return torch.stack(centers, dim=0)

    def compute_similarity_maps(self, qry_fts, prototypes):
        """Compute similarity maps (Eq. 2)."""
        C, H, W = qry_fts.shape
        Nf = prototypes.shape[0]
        
        qry_fts_norm = F.normalize(qry_fts.reshape(C, -1), p=2, dim=0)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        
        sim_maps = torch.mm(proto_norm, qry_fts_norm)
        sim_maps = sim_maps.reshape(Nf, H, W)
        sim_maps = F.interpolate(
            sim_maps.unsqueeze(0), 
            size=self.img_size, 
            mode='bilinear', 
            align_corners=True
        ).squeeze(0)
        
        return sim_maps * self.scaler

    def compute_single_similarity_map(self, qry_fts, prototype):
        """Compute single similarity map."""
        sim = F.cosine_similarity(qry_fts, prototype.view(-1, 1, 1), dim=0) * self.scaler
        sim = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0),
            size=self.img_size,
            mode='bilinear',
            align_corners=True
        ).squeeze()
        return sim

    def compute_periphery_prototype(self, supp_fts, periphery_mask):
        """Compute periphery prototype for negative prompts."""
        B, C, h, w = supp_fts.shape
        
        mask_resized = F.interpolate(
            periphery_mask.unsqueeze(1).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)
        
        masked_fts = supp_fts * mask_resized.unsqueeze(1)
        proto = masked_fts.sum(dim=(-2, -1)) / (mask_resized.sum(dim=(-2, -1), keepdim=True) + 1e-5)
        return proto

    def MAUP(self, similarity_maps, periphery_sim_map):
        """Multi-center Adaptive Uncertainty-aware Prompting."""
        H, W = self.img_size
        Nf = similarity_maps.shape[0]
        
        # Mean similarity map (Eq. 4)
        mean_sim_map = similarity_maps.mean(dim=0)
        
        # Uncertainty map (Eq. 8) - population variance
        diff_squared = (similarity_maps - mean_sim_map.unsqueeze(0)) ** 2
        uncertainty_map = diff_squared.sum(dim=0) / Nf
        
        # Mean similarity path (Eq. 5-7)
        tau_mean = torch.quantile(mean_sim_map.view(-1), self.tau_percentile / 100.0)
        Q_mean_mask = mean_sim_map >= tau_mean
        Q_mean_coords = torch.stack(torch.where(Q_mean_mask), dim=1)
        
        if Q_mean_coords.shape[0] > 0:
            complexity = self._compute_complexity(mean_sim_map)
            k = max(self.Nmin, min(self.Nmax, int(self.gamma * complexity)))
            k = min(k, Q_mean_coords.shape[0])
            
            if k > 1 and Q_mean_coords.shape[0] > k:
                Q_mean_coords_np = Q_mean_coords.cpu().numpy()
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Q_mean_coords_np)
                mean_positive_points = kmeans.cluster_centers_
            else:
                mean_positive_points = Q_mean_coords[:k].cpu().numpy()
        else:
            mean_positive_points = np.array([[H // 2, W // 2]])
        
        # Uncertainty path (Eq. 9)
        tau_uncert = torch.quantile(uncertainty_map.view(-1), self.tau_percentile / 100.0)
        Q_uncert_mask = uncertainty_map >= tau_uncert
        Q_uncert_coords = torch.stack(torch.where(Q_uncert_mask), dim=1)
        
        if Q_uncert_coords.shape[0] >= self.n_uncertainty_points:
            indices = np.random.choice(Q_uncert_coords.shape[0], size=self.n_uncertainty_points, replace=False)
            uncert_positive_points = Q_uncert_coords[indices].cpu().numpy()
        elif Q_uncert_coords.shape[0] > 0:
            uncert_positive_points = Q_uncert_coords.cpu().numpy()
        else:
            uncert_positive_points = np.array([]).reshape(0, 2)
        
        # Merge positive prompts (Eq. 10)
        positive_points = np.concatenate([mean_positive_points, uncert_positive_points], axis=0)
        
        # Negative prompts (Eq. 11)
        tau_neg = torch.quantile(periphery_sim_map.view(-1), self.tau_percentile / 100.0)
        Q_neg_mask = periphery_sim_map >= tau_neg
        Q_neg_coords = torch.stack(torch.where(Q_neg_mask), dim=1)
        
        if Q_neg_coords.shape[0] > 0:
            n_neg = min(len(positive_points), Q_neg_coords.shape[0])
            if n_neg > 0:
                indices = np.random.choice(Q_neg_coords.shape[0], size=n_neg, replace=False)
                negative_points = Q_neg_coords[indices].cpu().numpy()
            else:
                negative_points = np.array([]).reshape(0, 2)
        else:
            negative_points = np.array([]).reshape(0, 2)
        
        # Convert (y,x) to (x,y) for SAM
        positive_points_xy = positive_points[:, ::-1].copy() if len(positive_points) > 0 else np.array([]).reshape(0, 2)
        negative_points_xy = negative_points[:, ::-1].copy() if len(negative_points) > 0 else np.array([]).reshape(0, 2)
        
        if len(positive_points_xy) == 0:
            positive_points_xy = np.array([[W // 2, H // 2]])
        
        input_points = np.concatenate([positive_points_xy, negative_points_xy], axis=0).astype(np.float32)
        point_labels = np.concatenate([
            np.ones(len(positive_points_xy)),
            np.zeros(len(negative_points_xy))
        ], axis=0).astype(np.int32)
        
        return input_points, point_labels

    def _compute_complexity(self, sim_map):
        """Compute target region complexity (Eq. 6)."""
        threshold = sim_map.median()
        binary_mask = (sim_map > threshold).float().cpu().numpy().astype(np.uint8)
        
        area = binary_mask.sum() / binary_mask.size
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        perimeter = perimeter / (2 * (binary_mask.shape[0] + binary_mask.shape[1]))
        
        return area + perimeter

    def preprocess_image_for_sam(self, img):
        """Preprocess image for SAM input."""
        img_max, img_min = img.max(), img.min()
        img_sam = (img - img_min) / (img_max - img_min + 1e-5)
        img_sam = img_sam.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        return img_sam.astype(np.uint8)

    def getPred(self, fts, prototype):
        sim = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        return sim

    def getFeatures(self, fts, mask):
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        return masked_fts

    def getPrototype(self, fg_fts):
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in fg_fts]
        return fg_prototypes

    def preprocess_batched_image_tensor(self, batched_image_tensor, sam_model):
        device = batched_image_tensor.device
        batch_size = batched_image_tensor.shape[0]
        processed_images = []
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        
        for i in range(batch_size):
            single_image_tensor = batched_image_tensor[i]
            single_image_numpy = single_image_tensor.cpu().permute(1, 2, 0).numpy()
            single_image_numpy_uint8 = (single_image_numpy * 255).clip(0, 255).astype(np.uint8)
            resized_image_numpy = transform.apply_image(single_image_numpy_uint8)
            resized_image_tensor = torch.as_tensor(resized_image_numpy, device=device).permute(2, 0, 1)
            processed_images.append(resized_image_tensor)
            
        processed_batched_tensor = torch.stack(processed_images, dim=0)
        pixel_mean = torch.Tensor(sam_model.pixel_mean).to(device)
        pixel_std = torch.Tensor(sam_model.pixel_std).to(device)
        processed_batched_tensor = (processed_batched_tensor - pixel_mean) / pixel_std
        return processed_batched_tensor
