# the training-free model key file
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
    """
    MAUP: Training-free Multi-center Adaptive Uncertainty-aware Prompting
    for Cross-domain Few-shot Medical Image Segmentation
    
    Based on the paper by Zhu et al.
    """
    
    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()
        # Initialize device and parameters
        self.device = torch.device('cuda')
        self.scaler = 5.0  # Scaling factor for similarity computation
        self.fg_sampler = np.random.RandomState(1289)  # Random state for foreground sampling
        
        # MAUP parameters (from paper)
        self.Nf = 30  # Number of foreground regions for RPG (Section 2.2)
        self.Nmin = 3  # Minimum number of prompts (Section 2.3)
        self.Nmax = 10  # Maximum number of prompts (Section 2.3)
        self.gamma = 0.5  # Scaling factor for complexity (Section 2.3)
        self.tau_percentile = 95  # Percentile threshold for selection (Section 2.3)
        self.n_uncertainty_points = 2  # Number of uncertainty-based prompts (Section 2.3)
        self.dilation_radius = 8  # Radius for morphological dilation (Section 2.2)

        # Load and setup SAM model
        self.SAM = sam_model_registry['vit_h'](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
        self.SAM = self.SAM.eval()  # Set SAM to evaluation mode
        self.SAM_Encoder = self.SAM.image_encoder.eval()  # SAM image encoder
        self.SAM_Predictor = SamPredictor(self.SAM)  # SAM predictor
        
        # Load and setup DINOv2 model (DINOv2-ViT-L/14 as per paper)
        self.dinov2 = Dinov2Model.from_pretrained('./dinov2_weights')
        self.dinov2.eval()  # Set DINOv2 to evaluation mode

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        """
        Forward pass for few-shot segmentation using MAUP strategy.

        Args:
            supp_imgs: Support images, list of lists with shape way x shot x [B x 3 x H x W]
            supp_mask: Support masks, list of lists with shape way x shot x [B x H x W]
            qry_imgs: Query images, list with shape N x [B x 3 x H x W]
            qry_mask: Query masks (not used in this implementation)
            train: Training flag (not used - training-free method)
            t_loss_scaler: Loss scaler (not used)
            n_iters: Number of iterations (not used)
        Returns:
            output: Segmentation predictions, shape [B x 2 x H x W]
        """
        # Set dimensions and validate inputs
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.img_size = supp_imgs[0][0].shape[-2:]  # Image height and width
        assert self.n_ways == 1  # Currently only supports one-way
        assert self.n_queries == 1  # Currently only supports one query

        qry_bs = qry_imgs[0].shape[0]  # Query batch size
        supp_bs = supp_imgs[0][0].shape[0]  # Support batch size
        img_size = supp_imgs[0][0].shape[-2:]  # Image size
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask], dim=0).view(
            supp_bs, self.n_ways, self.n_shots, *img_size)

        # Concatenate support and query images for processing
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)
        
        # Extract features using DINOv2 (Section 2.2)
        outputs_dinov2 = self.dinov2(pixel_values=imgs_concat)
        fts_dinov2 = outputs_dinov2.last_hidden_state[:, 1:]  # Remove CLS token
        B, N, C = fts_dinov2.shape
        H = W = int(N ** 0.5)
        fts_dinov2 = fts_dinov2.transpose(1, 2).reshape(B, C, H, W)
        fts_dinov2_upsampled = F.interpolate(fts_dinov2, size=(64, 64), mode='bilinear', align_corners=True)
        
        # Split features into support and query
        supp_embedding_dinov2 = fts_dinov2_upsampled[:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *fts_dinov2_upsampled.shape[-2:])
        qry_embedding_dinov2 = fts_dinov2_upsampled[self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *fts_dinov2_upsampled.shape[-2:])

        # Compute periphery mask using morphological operations (Section 2.2, Eq. 3)
        kernel = np.ones((self.dilation_radius, self.dilation_radius), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_.astype(np.uint8), kernel, iterations=1)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots, *img_size))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots, *img_size))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()

        outputs = []
        for epi in range(supp_bs):
            # Use DINOv2 embeddings
            supp_embedding = supp_embedding_dinov2
            qry_embedding = qry_embedding_dinov2

            # Regional Prototype Generation (RPG) - Section 2.2, Fig. 2
            # Generate Nf regional prototypes using Voronoi-based partition
            regional_prototypes = self.regional_prototype_generation(
                supp_embedding[[epi], 0, 0],  # [1, C, H, W]
                supp_mask[[epi], 0, 0]  # [1, H, W]
            )  # Returns [1, Nf, C]

            # Compute multi-center positive similarity maps (Section 2.2, Eq. 2)
            similarity_maps = self.compute_similarity_maps(
                qry_embedding[epi, 0],  # [C, H, W]
                regional_prototypes[0]  # [Nf, C]
            )  # Returns [Nf, H, W]

            # Compute periphery similarity map for negative prompts (Section 2.2)
            periphery_prototype = self.compute_periphery_prototype(
                supp_embedding[[epi], 0, 0],
                supp_periphery_mask[[epi], 0, 0]
            )  # [1, C]
            periphery_sim_map = self.compute_single_similarity_map(
                qry_embedding[epi, 0],
                periphery_prototype[0]
            )  # [H, W]

            # MAUP: Multi-center Adaptive Uncertainty-aware Prompting (Section 2.3)
            input_points, point_labels = self.MAUP(
                similarity_maps,
                periphery_sim_map
            )

            # SAM prediction (Section 2.4)
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

            # Convert mask to tensor and prepare output
            best_mask_t = torch.from_numpy(best_mask.astype(bool)).float().unsqueeze(0).cuda()
            preds = torch.cat((1.0 - best_mask_t, best_mask_t), dim=1)
            outputs.append(preds)

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *outputs[0].shape[1:])
        return output

    def regional_prototype_generation(self, supp_fts, supp_mask):
        """
        Regional Prototype Generation (RPG) module - Section 2.2, Fig. 2
        
        Divides foreground into Nf regions using Voronoi-based partition and
        generates regional prototypes via Masked Average Pooling (MAP).
        
        Args:
            supp_fts: Support features [B, C, H, W]
            supp_mask: Support mask [B, H, W]
        Returns:
            regional_prototypes: [B, Nf, C]
        """
        B, C, h, w = supp_fts.shape
        
        # Resize mask to feature size
        fg_mask = F.interpolate(
            supp_mask.unsqueeze(1).float(), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=True
        ).squeeze(1) > 0.5
        
        batch_prototypes = []
        
        for b in range(B):
            fg_mask_b = fg_mask[b]  # [H, W]
            fts_b = supp_fts[b]  # [C, H, W]
            
            # Get foreground pixel coordinates
            fg_coords = torch.stack(torch.where(fg_mask_b), dim=1)  # [N_fg, 2]
            
            if fg_coords.shape[0] < self.Nf:
                # If too few foreground pixels, duplicate some
                n_repeat = (self.Nf // fg_coords.shape[0]) + 1
                fg_coords = fg_coords.repeat(n_repeat, 1)[:self.Nf]
            
            # Voronoi-based partition using Farthest Point Sampling
            centers = self._farthest_point_sampling(fg_coords, self.Nf)
            
            # Assign each pixel to nearest center (Voronoi regions)
            dist = fg_coords.float().reshape(-1, 1, 2) - centers.float().reshape(1, -1, 2)
            region_labels = torch.argmin((dist ** 2).sum(-1), dim=1)  # [N_fg]
            
            # Compute regional prototypes via Masked Average Pooling (Eq. 1)
            fg_features = fts_b.permute(1, 2, 0)[fg_mask_b]  # [N_fg, C]
            
            prototypes = []
            for i in range(self.Nf):
                region_mask = (region_labels == i)
                if region_mask.sum() > 0:
                    proto = fg_features[region_mask].mean(0)  # [C]
                else:
                    # If region is empty, use global foreground prototype
                    proto = fg_features.mean(0)
                prototypes.append(proto)
            
            prototypes = torch.stack(prototypes, dim=0)  # [Nf, C]
            batch_prototypes.append(prototypes)
        
        return torch.stack(batch_prototypes, dim=0)  # [B, Nf, C]

    def _farthest_point_sampling(self, points, n_samples):
        """
        Farthest Point Sampling for Voronoi center selection.
        
        Args:
            points: [N, 2] tensor of point coordinates
            n_samples: number of samples to select
        Returns:
            centers: [n_samples, 2] tensor of selected center coordinates
        """
        N = points.shape[0]
        centers = []
        
        # Start with a random point
        idx = self.fg_sampler.choice(N)
        centers.append(points[idx])
        
        for _ in range(n_samples - 1):
            # Compute distances to all existing centers
            dist = points.float().reshape(-1, 1, 2) - torch.stack(centers, dim=0).float().reshape(1, -1, 2)
            min_dist = (dist ** 2).sum(-1).min(1)[0]  # [N]
            
            # Select point with maximum minimum distance
            idx = torch.argmax(min_dist)
            centers.append(points[idx])
        
        return torch.stack(centers, dim=0)  # [n_samples, 2]

    def compute_similarity_maps(self, qry_fts, prototypes):
        """
        Compute similarity maps between query features and regional prototypes.
        Section 2.2, Eq. 2: S_n = cos(F_q, p_n)
        
        Args:
            qry_fts: Query features [C, H, W]
            prototypes: Regional prototypes [Nf, C]
        Returns:
            similarity_maps: [Nf, H, W]
        """
        C, H, W = qry_fts.shape
        Nf = prototypes.shape[0]
        
        # Normalize features
        qry_fts_norm = F.normalize(qry_fts.reshape(C, -1), p=2, dim=0)  # [C, H*W]
        proto_norm = F.normalize(prototypes, p=2, dim=1)  # [Nf, C]
        
        # Compute cosine similarity
        sim_maps = torch.mm(proto_norm, qry_fts_norm)  # [Nf, H*W]
        sim_maps = sim_maps.reshape(Nf, H, W)
        
        # Upsample to original image size
        sim_maps = F.interpolate(
            sim_maps.unsqueeze(0), 
            size=self.img_size, 
            mode='bilinear', 
            align_corners=True
        ).squeeze(0)
        
        return sim_maps * self.scaler

    def compute_single_similarity_map(self, qry_fts, prototype):
        """
        Compute single similarity map for a prototype.
        
        Args:
            qry_fts: Query features [C, H, W]
            prototype: Single prototype [C]
        Returns:
            sim_map: [H, W]
        """
        sim = F.cosine_similarity(
            qry_fts, 
            prototype.view(-1, 1, 1), 
            dim=0
        ) * self.scaler
        
        # Upsample to original image size
        sim = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0),
            size=self.img_size,
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        return sim

    def compute_periphery_prototype(self, supp_fts, periphery_mask):
        """
        Compute periphery prototype for negative prompts.
        Section 2.2: P_tilde = MAP(F_s, M_s_tilde)
        
        Args:
            supp_fts: Support features [B, C, H, W]
            periphery_mask: Periphery mask [B, H, W]
        Returns:
            periphery_proto: [B, C]
        """
        B, C, h, w = supp_fts.shape
        
        # Resize mask to feature size
        mask_resized = F.interpolate(
            periphery_mask.unsqueeze(1).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)
        
        # Masked Average Pooling
        masked_fts = supp_fts * mask_resized.unsqueeze(1)
        proto = masked_fts.sum(dim=(-2, -1)) / (mask_resized.sum(dim=(-2, -1), keepdim=True) + 1e-5)
        
        return proto  # [B, C]

    def MAUP(self, similarity_maps, periphery_sim_map):
        """
        Multi-center Adaptive Uncertainty-aware Prompting (MAUP) - Section 2.3
        
        Generates optimal point prompts for SAM through:
        1. Mean similarity map based prompting with K-means clustering
        2. Uncertainty map based prompting for challenging regions
        3. Periphery-based negative prompting
        
        Args:
            similarity_maps: Multi-center similarity maps [Nf, H, W]
            periphery_sim_map: Periphery similarity map [H, W]
        Returns:
            input_points: Prompt point coordinates [N_points, 2]
            point_labels: Prompt point labels [N_points] (1=positive, 0=negative)
        """
        H, W = self.img_size
        
        # ==================== Positive Prompting ====================
        
        # 1. Mean Similarity Map (Eq. 4)
        mean_sim_map = similarity_maps.mean(dim=0)  # [H, W]
        
        # 2. Uncertainty Map (Eq. 8) - variance across similarity maps
        uncertainty_map = similarity_maps.var(dim=0)  # [H, W]
        
        # -------------------- Mean Similarity Path --------------------
        # Select pixels with highest similarity (Eq. 5)
        tau_mean = torch.quantile(mean_sim_map.view(-1), self.tau_percentile / 100.0)
        Q_mean_mask = mean_sim_map >= tau_mean
        Q_mean_coords = torch.stack(torch.where(Q_mean_mask), dim=1)  # [N, 2] - (y, x)
        
        if Q_mean_coords.shape[0] > 0:
            # Compute target complexity (Eq. 6)
            complexity = self._compute_complexity(mean_sim_map)
            
            # Determine number of clusters k (Eq. 7)
            k = max(self.Nmin, min(self.Nmax, int(self.gamma * complexity)))
            k = min(k, Q_mean_coords.shape[0])
            
            # K-means clustering for spatial diversity
            if k > 1 and Q_mean_coords.shape[0] > k:
                Q_mean_coords_np = Q_mean_coords.cpu().numpy()
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Q_mean_coords_np)
                mean_positive_points = kmeans.cluster_centers_  # [k, 2] - (y, x)
            else:
                mean_positive_points = Q_mean_coords[:k].cpu().numpy()
        else:
            # Fallback: use center of image
            mean_positive_points = np.array([[H // 2, W // 2]])
        
        # -------------------- Uncertainty Map Path --------------------
        # Select pixels with highest uncertainty (Eq. 9)
        tau_uncert = torch.quantile(uncertainty_map.view(-1), self.tau_percentile / 100.0)
        Q_uncert_mask = uncertainty_map >= tau_uncert
        Q_uncert_coords = torch.stack(torch.where(Q_uncert_mask), dim=1)  # [N, 2]
        
        if Q_uncert_coords.shape[0] >= self.n_uncertainty_points:
            # Randomly select n_uncertainty_points from high uncertainty regions
            indices = np.random.choice(
                Q_uncert_coords.shape[0], 
                size=self.n_uncertainty_points, 
                replace=False
            )
            uncert_positive_points = Q_uncert_coords[indices].cpu().numpy()
        elif Q_uncert_coords.shape[0] > 0:
            uncert_positive_points = Q_uncert_coords.cpu().numpy()
        else:
            uncert_positive_points = np.array([]).reshape(0, 2)
        
        # Merge positive prompts (Eq. 10): Q_pos = Q_mean ∪ Q_uncert
        positive_points = np.concatenate([mean_positive_points, uncert_positive_points], axis=0)
        
        # ==================== Negative Prompting (Eq. 11) ====================
        # Select from periphery similarity map
        tau_neg = torch.quantile(periphery_sim_map.view(-1), self.tau_percentile / 100.0)
        Q_neg_mask = periphery_sim_map >= tau_neg
        Q_neg_coords = torch.stack(torch.where(Q_neg_mask), dim=1)
        
        if Q_neg_coords.shape[0] > 0:
            # Select a subset of negative points
            n_neg = min(len(positive_points), Q_neg_coords.shape[0])
            if n_neg > 0:
                indices = np.random.choice(Q_neg_coords.shape[0], size=n_neg, replace=False)
                negative_points = Q_neg_coords[indices].cpu().numpy()
            else:
                negative_points = np.array([]).reshape(0, 2)
        else:
            negative_points = np.array([]).reshape(0, 2)
        
        # ==================== Combine Prompts ====================
        # Convert from (y, x) to (x, y) for SAM
        positive_points_xy = positive_points[:, ::-1].copy() if len(positive_points) > 0 else np.array([]).reshape(0, 2)
        negative_points_xy = negative_points[:, ::-1].copy() if len(negative_points) > 0 else np.array([]).reshape(0, 2)
        
        if len(positive_points_xy) == 0:
            # Ensure at least one positive point
            positive_points_xy = np.array([[W // 2, H // 2]])
        
        input_points = np.concatenate([positive_points_xy, negative_points_xy], axis=0).astype(np.float32)
        point_labels = np.concatenate([
            np.ones(len(positive_points_xy)),
            np.zeros(len(negative_points_xy))
        ], axis=0).astype(np.int32)
        
        return input_points, point_labels

    def _compute_complexity(self, sim_map):
        """
        Compute target region complexity (Eq. 6).
        C = Area(υ) + Perimeter(υ)
        
        Args:
            sim_map: Similarity map [H, W]
        Returns:
            complexity: float
        """
        # Threshold to get binary mask
        threshold = sim_map.median()
        binary_mask = (sim_map > threshold).float().cpu().numpy().astype(np.uint8)
        
        # Compute area (normalized)
        area = binary_mask.sum() / binary_mask.size
        
        # Compute perimeter using contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        perimeter = perimeter / (2 * (binary_mask.shape[0] + binary_mask.shape[1]))  # Normalize
        
        complexity = area + perimeter
        return complexity

    # ==================== Legacy Methods (kept for compatibility) ====================
    
    def AMCPG(self, similarity_maps, supp_embedding, supp_mask, qry_embedding):
        """
        Legacy AMCPG method - redirects to MAUP for backward compatibility.
        """
        # Compute periphery prototype and similarity map
        kernel = np.ones((self.dilation_radius, self.dilation_radius), np.uint8)
        supp_mask_np = supp_mask[0, 0].cpu().numpy().astype(np.uint8)
        dilated = cv2.dilate(supp_mask_np, kernel, iterations=1)
        periphery_mask = torch.tensor(dilated - supp_mask_np).cuda().unsqueeze(0).unsqueeze(0)
        
        periphery_proto = self.compute_periphery_prototype(
            supp_embedding[0:1, 0],
            periphery_mask[0, 0]
        )
        periphery_sim = self.compute_single_similarity_map(
            qry_embedding[0],
            periphery_proto[0]
        )
        
        return self.MAUP(similarity_maps, periphery_sim)

    def getPred(self, fts, prototype):
        """Compute cosine similarity between features and prototype."""
        sim = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        return sim

    def cluster_and_select_prototypes(self, prototypes, n_select=5):
        """Cluster prototypes and select the top n_select based on similarity."""
        assert len(prototypes.shape) == 3
        batch_size, n_prototypes, n_features = prototypes.shape
        assert batch_size == 1

        prototypes_flat = prototypes.squeeze(0)
        normalized_prototypes = F.normalize(prototypes_flat, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_prototypes, normalized_prototypes.t())
        mean_similarities = similarity_matrix.mean(dim=1)
        _, selected_indices = torch.topk(mean_similarities, k=min(n_select, n_prototypes))
        selected_prototypes = prototypes[:, selected_indices]
        return selected_prototypes

    def getmultiPred(self, fts, prototypes, n_select):
        """Compute multi-prototype similarity maps."""
        n = min(n_select, prototypes.shape[1])
        similarity_maps = []
        for i in range(n):
            sub_prototype = prototypes[0, i, :].unsqueeze(0)
            sub_sim = F.cosine_similarity(fts, sub_prototype[..., None, None], dim=1) * self.scaler
            similarity_maps.append(sub_sim)
        maps = torch.stack(similarity_maps, dim=0)
        maps = F.interpolate(maps, size=self.img_size, mode='bilinear', align_corners=True)
        return maps

    def getFeatures(self, fts, mask):
        """Extract features from masked regions."""
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        return masked_fts

    def getPrototype(self, fg_fts):
        """Compute prototype by averaging features."""
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in fg_fts]
        return fg_prototypes

    def preprocess_batched_image_tensor(self, batched_image_tensor, sam_model):
        """Preprocess batched image tensor for SAM."""
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

    def preprocess_image_for_sam(self, img):
        """Simplified preprocessing for SAM input."""
        img_max, img_min = img.max(), img.min()
        img_sam = (img - img_min) / (img_max - img_min + 1e-5)
        img_sam = img_sam.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        return img_sam.astype(np.uint8)

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """
        Compute multiple foreground prototypes (legacy method).
        Now uses RPG internally.
        """
        B, C, h, w = sup_fts.shape
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear').squeeze(0).bool()
        batch_fg_protos = []

        for b in range(B):
            fg_protos = []
            fg_mask_i = fg_mask[b]
            if fg_mask_i.sum() < fg_num:
                fg_mask_i = fg_mask[b].clone()
                fg_mask_i.view(-1)[:fg_num] = True

            all_centers = []
            first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)
            for _ in range(fg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]
                all_centers.append(pt)

            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)
            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]
            for i in range(fg_num):
                proto = fg_feats[fg_labels == i].mean(0)
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)
            batch_fg_protos.append(fg_protos)
        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)
        return fg_proto
    

