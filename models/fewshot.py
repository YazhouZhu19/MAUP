import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from transformers import Dinov2Model

class FewShotSeg(nn.Module):
    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()
        # Initialize device and parameters
        self.device = torch.device('cuda')
        self.scaler = 5.0  # Scaling factor for similarity computation
        self.fg_sampler = np.random.RandomState(1289)  # Random state for foreground sampling
        self.fg_num = 40  # Number of foreground partitions

        # Load and setup SAM model
        self.SAM = sam_model_registry['vit_h'](checkpoint=".../sam_vit_h_4b8939.pth")
        self.SAM = self.SAM.eval()  # Set SAM to evaluation mode
        self.SAM_Encoder = self.SAM.image_encoder.eval()  # SAM image encoder
        self.SAM_Predictor = SamPredictor(self.SAM)  # SAM predictor
        
        # Load and setup DINOv2 model
        self.dinov2 = Dinov2Model.from_pretrained('./dinov2_weights')
        self.dinov2.eval()  # Set DINOv2 to evaluation mode

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        """
        Forward pass for few-shot segmentation.

        Args:
            supp_imgs: Support images, list of lists with shape way x shot x [B x 3 x H x W]
            supp_mask: Support masks, list of lists with shape way x shot x [B x H x W]
            qry_imgs: Query images, list with shape N x [B x 3 x H x W]
            qry_mask: Query masks (not used in this implementation)
            train: Training flag (not used)
            t_loss_scaler: Loss scaler (not used)
            n_iters: Number of iterations (not used)
        Returns:
            output: Segmentation predictions, shape [B x 2 x H x W]
        """
        # Set dimensions and validate inputs
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.iter = 3  # Number of iterations (fixed)
        self.img_size = supp_imgs[0][0].shape[-2:]  # Image height and width
        assert self.n_ways == 1  # Currently only supports one-way
        assert self.n_queries == 1  # Currently only supports one query

        qry_bs = qry_imgs[0].shape[0]  # Query batch size
        supp_bs = supp_imgs[0][0].shape[0]  # Support batch size
        img_size = supp_imgs[0][0].shape[-2:]  # Image size
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask], dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        # Concatenate support and query images for processing
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)
        imgs_concat_sam = self.preprocess_batched_image_tensor(imgs_concat, self.SAM)
        
        # Extract features using SAM encoder
        img_emdeddings = self.SAM_Encoder(imgs_concat_sam)
        supp_embedding = img_emdeddings[:self.n_ways * self.n_shots * supp_bs].view(supp_bs, self.n_ways, self.n_shots, -1, *img_emdeddings.shape[-2:])
        qry_embedding = img_emdeddings[self.n_ways * self.n_shots * supp_bs:].view(qry_bs, self.n_queries, -1, *img_emdeddings.shape[-2:])

        # Extract features using DINOv2
        outputs_dinov2 = self.dinov2(pixel_values=imgs_concat)
        fts_dinov2 = outputs_dinov2.last_hidden_state[:, 1:]  # Remove CLS token
        B, N, C = fts_dinov2.shape
        H = W = int(N ** 0.5)
        fts_dinov2 = fts_dinov2.transpose(1, 2).reshape(B, C, H, W)
        fts_dinov2_upsampled = F.interpolate(fts_dinov2, size=(64, 64), mode='bilinear', align_corners=True)
        supp_embedding_dinov2 = fts_dinov2_upsampled[:self.n_ways * self.n_shots * supp_bs].view(supp_bs, self.n_ways, self.n_shots, -1, *fts_dinov2_upsampled.shape[-2:])
        qry_embedding_dinov2 = fts_dinov2_upsampled[self.n_ways * self.n_shots * supp_bs:].view(qry_bs, self.n_queries, -1, *fts_dinov2_upsampled.shape[-2:])

        # Compute periphery mask
        kernel = np.ones((8, 8), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_, kernel, iterations=1)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots, *img_size))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots, *img_size))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()

        outputs = []
        for epi in range(supp_bs):
            # Use DINOv2 embeddings for prompt generation
            supp_embedding, qry_embedding = supp_embedding_dinov2, qry_embedding_dinov2

            # Compute multi-prototype similarity maps
            fg_partition_prototypes = [[self.compute_multiple_prototypes(self.fg_num, supp_embedding[[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_partition_prototypes = [[self.cluster_and_select_prototypes(fg_partition_prototypes[way][shot], n_select=10)
                                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
            multi_similarities = [[self.getmultiPred(qry_embedding[epi], fg_partition_prototypes[way][shot], n_select=10)
                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]
            similarity_maps = multi_similarities[0][0].squeeze(1)

            # Generate prompts using AMCPG
            input_points, point_labels = self.AMCPG(similarity_maps, supp_embedding[epi], supp_mask[epi], qry_embedding[epi])

            # SAM prediction
            qry_img = qry_imgs[0]
            qry_img_sam = self.preprocess_image_for_sam(qry_img)
            with torch.no_grad():
                self.SAM_Predictor.set_image(qry_img_sam)
                mask, score, logit = self.SAM_Predictor.predict(point_coords=input_points, point_labels=point_labels, multimask_output=False)
            best_mask = mask

            # Convert mask to tensor and prepare output
            best_mask_t = torch.from_numpy(best_mask.astype(bool)).float().unsqueeze(0).cuda()
            preds = torch.cat((1.0 - best_mask_t, best_mask_t), dim=1)
            outputs.append(preds)

        output = torch.stack(outputs, dim=1).view(-1, *output.shape[2:])
        return output

    def AMCPG(self, similarity_maps, supp_embedding, supp_mask, qry_embedding):
        """
        Adaptive Multi-Cue Prompt Generation (AMCPG) method.

        Args:
            similarity_maps: Multi-prototype similarity maps, shape (n_protos, H, W)
            supp_embedding: Support image embeddings, shape (n_ways, n_shots, C, H', W')
            supp_mask: Support image masks, shape (n_ways, n_shots, H, W)
            qry_embedding: Query image embeddings, shape (n_queries, C, H', W')
        Returns:
            input_points: Prompt point coordinates, shape (n_points, 2)
            point_labels: Prompt point labels, 1 for positive, 0 for negative, shape (n_points,)
        """
        # 1. Compute uncertainty map
        variance_map = torch.var(similarity_maps, dim=0)  # Variance across prototypes, shape (H, W)
        similarity_mean = torch.mean(similarity_maps, dim=0)  # Mean similarity, shape (H, W)
        uncertainty_map = variance_map / (similarity_mean.max() - similarity_mean.min() + 1e-5)  # Normalized uncertainty

        # 2. Extract high similarity and high uncertainty points
        n_pixels = similarity_mean.numel()
        n_high_sim = int(0.1 * n_pixels)  # Top 10% high similarity points
        n_high_uncertainty = int(0.2 * n_pixels)  # Top 20% high uncertainty points

        _, high_sim_indices = torch.topk(similarity_mean.view(-1), n_high_sim)
        _, high_uncertainty_indices = torch.topk(uncertainty_map.view(-1), n_high_uncertainty)

        high_sim_coords = torch.stack([high_sim_indices % self.img_size[1], high_sim_indices // self.img_size[1]], dim=1).cpu().numpy()
        high_uncertainty_coords = torch.stack([high_uncertainty_indices % self.img_size[1], high_uncertainty_indices // self.img_size[1]], dim=1).cpu().numpy()

        # 3. Positive prompts: Clustering selection
        n_clusters = min(5, len(high_sim_coords))  # Adaptive cluster number, max 5
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(high_sim_coords)
            positive_points = kmeans.cluster_centers_  # Use cluster centers as positive prompts
        else:
            positive_points = high_sim_coords[:1]  # If too few points, take the first one

        # Add positive prompts from high uncertainty regions
        uncertainty_sim_values = similarity_mean.view(-1)[high_uncertainty_indices]
        uncertainty_positive_mask = uncertainty_sim_values > similarity_mean.median()
        uncertainty_positive_coords = high_uncertainty_coords[uncertainty_positive_mask.cpu().numpy()]
        if len(uncertainty_positive_coords) > 0:
            positive_points = np.concatenate([positive_points, uncertainty_positive_coords[:min(3, len(uncertainty_positive_coords))]], axis=0)

        # 4. Negative prompts: Select from low similarity regions
        n_neg = int(0.05 * n_pixels)  # Top 5% low similarity points as negative prompts
        _, low_sim_indices = torch.topk(-similarity_mean.view(-1), n_neg)
        negative_points = torch.stack([low_sim_indices % self.img_size[1], low_sim_indices // self.img_size[1]], dim=1).cpu().numpy()

        # 5. Support mask-guided optimization
        supp_mask_resized = F.interpolate(supp_mask.float(), size=(64, 64), mode='bilinear', align_corners=True)[0, 0]
        supp_embedding_resized = F.interpolate(supp_embedding, size=(64, 64), mode='bilinear', align_corners=True)[0]
        qry_embedding_resized = F.interpolate(qry_embedding, size=(64, 64), mode='bilinear', align_corners=True)[0]

        supp_features = supp_embedding_resized * supp_mask_resized[None, :, :]
        supp_proto = supp_features.sum(dim=(-2, -1)) / (supp_mask_resized.sum() + 1e-5)
        sim_map_supp = F.cosine_similarity(qry_embedding_resized, supp_proto[..., None, None], dim=0)
        _, supp_guided_idx = torch.topk(sim_map_supp.view(-1), min(2, n_high_sim // 2))
        supp_guided_points = torch.stack([supp_guided_idx % 64, supp_guided_idx // 64], dim=1).cpu().numpy()
        positive_points = np.concatenate([positive_points, supp_guided_points], axis=0)

        # 6. Adaptive prompt quantity adjustment
        target_complexity = torch.sum(similarity_mean > similarity_mean.median()).item() / n_pixels
        max_points = int(10 * target_complexity) + 5  # Adjust max prompts based on target complexity
        positive_points = positive_points[:min(len(positive_points), max_points // 2)]
        negative_points = negative_points[:min(len(negative_points), max_points // 2)]

        # 7. Combine prompts
        input_points = np.concatenate([positive_points, negative_points], axis=0)
        point_labels = np.concatenate([np.ones(len(positive_points)), np.zeros(len(negative_points))], axis=0)

        return input_points, point_labels
        
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
        _, selected_indices = torch.topk(mean_similarities, k=n_select)
        selected_prototypes = prototypes[:, selected_indices]
        return selected_prototypes

    def getmultiPred(self, fts, prototypes, n_select):
        """Compute multi-prototype similarity maps."""
        n = n_select
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
        """Compute multiple foreground prototypes."""
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

# Example usage for testing
if __name__ == "__main__":
    model = FewShotSeg().cuda()
    supp_imgs = [torch.randn(1, 1, 3, 256, 256).cuda()]  # Example support images
    supp_mask = [torch.ones(1, 1, 256, 256).cuda()]      # Example support masks
    qry_imgs = [torch.randn(1, 3, 256, 256).cuda()]      # Example query images
    qry_mask = [torch.ones(1, 256, 256).cuda()]          # Example query masks
    output = model(supp_imgs, supp_mask, qry_imgs, qry_mask)
    print(output.shape)  # Expected output: torch.Size([1, 2, 256, 256])
