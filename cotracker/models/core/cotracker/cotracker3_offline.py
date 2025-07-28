# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeBase, posenc

torch.manual_seed(0)

def project_points_monotonic(coords, confidence):
    B, T, N, D = coords.shape
    # Reshape to combine batch and time: [B*T, N, D]
    coords_flat = coords.reshape(B * T, N, D)
    confidence_flat = confidence.reshape(B * T, N).unsqueeze(-1)  # Shape: [B*T, N, 1]

    # Compute the mean and center the data
    weighted_mean = (coords_flat * confidence_flat).sum(dim=1, keepdim=True) / (confidence_flat.sum(dim=1, keepdim=True) + 1e-8)  # shape: [B*T, 1, D]
    centered = coords_flat - weighted_mean                 # shape: [B*T, N, D]
    
    # Reshape confidence to match the centered data shape
    # Perform SVD on the centered data (differentiable) -> U: [B*T, N, D], S: [B*T, D], V: [B*T, D, D]
    U, S, Vh = torch.svd(centered, full_matrices=False)
    v = Vh[:,0,:] # v has shape (B*T, 2) and is a unit vector representing the best-fit line direction.
    
    v_dir = v/torch.norm(v)
    proj_scalar = (centered * v.unsqueeze(1)).sum(dim=-1)  # Shape: (B*T, N)


def project_points(coords, confidence, projection):
    """
    Projects each 2D point onto the best-fit line for that frame.
    Parameters:
    -----------
    data : torch.Tensor
        A tensor of shape (B, T, N, 2) where:
          - B is the batch size,
          - T is the number of time samples,
          - N is the number of points per sample,
          - 2 corresponds to the (x, y) coordinates of each point.
    confidence : torch.Tensor
        A tensor of shape (B, T, N) containing confidence scores for each point.
    Returns:
    --------
    torch.Tensor
        A tensor of shape (B, T, N, 2) containing the projected points.
    """
    assert projection in set(['unweighted-svd', 'weighted-svd'])
    B, T, N, D = coords.shape
    # Reshape to combine batch and time: [B*T, N, D]
    coords_flat = coords.reshape(B * T, N, D)
    confidence_flat = confidence.reshape(B * T, N).unsqueeze(-1)  # Shape: [B*T, N, 1]
    
    mean = None
    # Compute the mean and center the data
    if projection=='weighted-svd':
        mean = (coords_flat * confidence_flat).sum(dim=1, keepdim=True) / (confidence_flat.sum(dim=1, keepdim=True) + 1e-8)  # shape: [B*T, 1, D]
    elif projection=='unweighted-svd':
        mean = coords_flat.mean(dim=1,keepdim=True)
    assert mean is not None
    centered = coords_flat - mean                 # shape: [B*T, N, D]
    
    # Reshape confidence to match the centered data shape
    # Perform SVD on the centered data (differentiable) -> U: [B*T, N, D], S: [B*T, D], V: [B*T, D, D]
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    v = Vh[:,0,:] # v has shape (B*T, 2) and is a unit vector representing the best-fit line direction.
    
    # Compute the scalar projection of each point onto v.
    proj_scalar = (centered * v.unsqueeze(1)).sum(dim=-1)  # Shape: (B*T, N)

    # Multiply the scalar projections by the direction vector to get the projected vectors.
    proj_vectors = proj_scalar.unsqueeze(-1) * v.unsqueeze(1)  # Shape: (B*T, N, 2)

    # Add the mean back to obtain the final projected points.
    projected_points = mean + proj_vectors  # Shape: (B*T, N, 2)

    # Reshape back to the original shape: [B, T, N, D]
    projected_points = projected_points.reshape(B, T, N, D)
    # print("Projected points:", projected_points[:,:,:projected_points.shape[2]/2,:])
    return projected_points

class CoTrackerThreeOffline(CoTrackerThreeBase):
    def __init__(self, **args):
        super(CoTrackerThreeOffline, self).__init__(**args)
        
    def _build_mask(self, video, queries, num_attend=6):
        '''
        video: tensor with shape B, T, C, H, W
        queries: tensor with shape B, N, 3
        '''
        # Build mask into shape of B*T, num_heads, N_virtual, N_point 
        B, T, C, H, W = video.shape
        _, N, _ = queries.shape
        point_labels_set = set(t.item() for t in self.point_labels)   # Get unique point labels (0 = points, 1-n = lines)
        device = queries.device
        
        # mask = torch.tensor([[]], dtype=torch.bool).to(device)
        mask_rows = []
        print("point_labels_set", point_labels_set)
        for label in point_labels_set:
            # if label != 0:      # Exclude point label 0
            filtered_labels = self.point_labels == label
            # filtered_labels = filtered_labels*2-1       # Range between -1 and 1
            
            # Add padding for the grid support query points if there are any
            padding = torch.zeros(max(N - filtered_labels.shape[0], 0), dtype=torch.bool).to(device)
            padded = torch.cat([filtered_labels, padding], dim=0)      # filtered_labels shape: (N,)
            # mask = torch.cat([mask, filtered_labels], dim=0)
            for _ in range(num_attend):
                mask_rows.append(padded)
                
                
        # mask = torch.unsqueeze(mask, 0)
        mask = torch.stack(mask_rows, dim=0)
        padding = torch.zeros((64-len(point_labels_set)*num_attend, mask.shape[1]), dtype=torch.bool).to(device)        # padding shape = (64-queries, queries)
        mask = torch.cat([mask, padding], dim=0)
        
        virtual2point_mask = mask.repeat(B*T, self.num_heads, 1, 1)
        point2virtual_mask = torch.transpose(virtual2point_mask, 2, 3)
        return virtual2point_mask, point2virtual_mask

    def forward(
        self,
        video,
        queries,
        iters=4,
        is_train=False,
        add_space_attn=True,
        fmaps_chunk_size=200,
        build_mask=False
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        print("CoTrackerThreeOffline forward")
        virtual2point_mask, point2virtual_mask = None, None
        B, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0
        B, N, __ = queries.shape
        # B = batch size
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B T N 2
        # vis_init = B T N 1
        if build_mask:
            squish_queries = queries.reshape(B*N, 3)  # [B*N, 3]
            
            squish_queries, self.point_labels = self.sam.build_labels(video, squish_queries)
        
            queries = squish_queries.reshape(B, N, 3).to(device)  # [B, N, 3]
            self.point_labels = self.point_labels.to(device)  # [B, N]
        assert T >= 1  # A tracker needs at least two frames to track something

        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        all_coords_predictions, all_vis_predictions, all_confidence_predictions = (
            [],
            [],
            [],
        )
        C_ = C
        H4, W4 = H // self.stride, W // self.stride
        # Compute convolutional features for the video or for the current chunk in case of online mode
        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))                      # Paper Label: Feature Network to extract features from video (Feature CNN)
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C_, H, W))
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        # We compute track features
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        D_coords = 2

        coord_preds, vis_preds, confidence_preds = [], [], []


        # Paper Label: Initialize coordinates (P), visibility (V), and confidence (C)
        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()

        r = 2 * self.corr_radius + 1


        for it in range(iters):                     # Paper Label: Repeat m iterations
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(B * T, N, 2)
            corr_embs = []                          # Paper Label: Correlation Embedding (Corr(a,b))
            corr_feats = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * T * N, r * r * r * r))
                corr_embs.append(corr_emb)
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, T, N, corr_embs.shape[-1])

            transformer_input = [vis[..., None], confidence[..., None], corr_embs]      # Paper Label: Transformer input has Correlation Embedding (Corr(a,b)), Confidence (C(i)), and Visibility (V(i))

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )
            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            transformer_input.append(rel_pos_emb_input)                             # Paper Label: Relative Position Embedding P(i)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )

            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)  # (B N) T D -> B N T D
            if build_mask:
                virtual2point_mask, point2virtual_mask = self._build_mask(video, queries)
            
            delta = self.updateformer(              # Paper Label: Transformer update
                x,
                virtual2point_mask=virtual2point_mask,
                point2virtual_mask=point2virtual_mask,
                add_space_attn=add_space_attn,
            )

            delta_coords = delta[..., :D_coords].permute(0, 2, 1, 3)
            delta_vis = delta[..., D_coords].permute(0, 2, 1)
            delta_confidence = delta[..., D_coords + 1].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_confidence

            coords = coords + delta_coords
            if it >= iters-2:
                if self.projection != 'non-svd':
                    if self.point_labels is not None:
                        point_labels_set = set(self.point_labels)   # Get unique point labels (0 = points, 1-n = lines)
                        for label in point_labels_set:
                            if label != 0:      # Exclude point label 0
                                filtered_labels = self.point_labels == label
                                padding = torch.zeros(max(N - filtered_labels.shape[0], 0), dtype=torch.bool).to(device)
                                filtered_labels = torch.cat([filtered_labels, padding], dim=0)
                                filtered_confidence = confidence[:, :, filtered_labels]
                                coords[:,:,filtered_labels] = project_points(coords[:,:,filtered_labels, :], filtered_confidence, projection=self.projection)
            coords_append = coords.clone()
            coords_append[..., :2] = coords_append[..., :2] * float(self.stride)
            coord_preds.append(coords_append)
            vis_preds.append(torch.sigmoid(vis))
            confidence_preds.append(torch.sigmoid(confidence))

        if is_train:
            all_coords_predictions.append([coord[..., :2] for coord in coord_preds])
            all_vis_predictions.append(vis_preds)
            all_confidence_predictions.append(confidence_preds)

        if is_train:
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                torch.ones_like(vis_preds[-1], device=vis_preds[-1].device),
            )
        else:
            train_data = None
        return coord_preds[-1][..., :2], vis_preds[-1], confidence_preds[-1], train_data
