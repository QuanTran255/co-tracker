"""
Modified from cotracker_online.py
Author: Bhargav Ghanekar
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from cotracker.models.core.model_utils import sample_features5d, bilinear_sampler
from cotracker.models.core.embeddings import get_1d_sincos_pos_embed_from_grid
from cotracker.models.core.cotracker.blocks import Mlp, BasicEncoder
from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer

torch.manual_seed(0)

def posenc(x, min_deg, max_deg):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.
    Returns:
      encoded: torch.Tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)

class CoTrackerThreeModified(nn.Module):
    def __init__(self, window_len, stride, 
                 corr_radius=3, corr_levels=4, 
                 num_virtual_tracks=64, 
                 model_resolution=(384, 512),
                 add_space_attn=True, 
                 linear_layer_for_vis_conf=True):
        super(CoTrackerThreeModified, self).__init__()
        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.hidden_dim = 256
        self.latent_dim = 128 
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim, stride=stride)

        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution

        self.input_dim = 1110
        self.updateformer = EfficientUpdateFormer(space_depth=3, time_depth=3, 
                                                  input_dim=self.input_dim, 
                                                  hidden_size=384, output_dim=4, mlp_ratio=4.0, 
                                                  num_virtual_tracks=num_virtual_tracks,
                                                  add_space_attn=add_space_attn,
                                                  linear_layer_for_vis_conf=linear_layer_for_vis_conf)
        self.corr_mlp = Mlp(in_features=49*49, hidden_features=384, out_features=256)
        
        time_grid = torch.linspace(0, window_len-1, window_len).reshape(1, window_len, 1)
        self.register_buffer("time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0]))

    def get_support_points(self, coords, r, reshape_back=True):
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)
        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)
        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl
        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl
    
    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):
        """
        fmaps: B x T x latent_dim x H4 x W4
        queried_frames: B x N
        queried_coords: B x N x 2
        """
        sample_frames = queried_frames[:,None,:,None]   # B x 1 x N x 1
        sample_coords = torch.cat([sample_frames, queried_coords[:,None],],dim=-1)  # B x 1 x N x 3
        support_points = self.get_support_points(sample_coords, support_radius)     # B x (2r+1)^2 x N x 3
        support_track_feats = sample_features5d(fmaps, support_points)              # B x (2r+1)^2 x N x latent_dim
        return (support_track_feats[:,None,support_track_feats.shape[1]//2], support_track_feats)

    def get_correlation_feat(self, fmaps, queried_coords): 
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius 
        sample_coords = torch.cat([torch.zeros_like(queried_coords[...,:1]), queried_coords], dim=-1)[:,None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(fmaps.reshape(B*T,D,1,H_,W_), support_points)
        return correlation_feat.view(B,T,D,N,(2*r+1),(2*r+1)).permute(0,1,3,4,5,2)
    
    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]
        if t == T: return self.time_emb
        time_emb = self.time_emb.float()
        time_emb = F.interpolate(time_emb.permute(0, 2, 1), size=t, mode="linear").permute(0, 2, 1)
        return time_emb.to(previous_dtype)
    
class CoTrackerThreeOfflineModified(CoTrackerThreeModified):
    def __init__(self, **args):
        super(CoTrackerThreeOfflineModified, self).__init__(**args)
    
    def forward(self, video, queries, iters=4, is_train=False, add_space_attn=True, fmaps_chunk_size=200):
        """Predict tracks
        Args:   video (FloatTensor[B, T, 3]): input videos.
                queries (FloatTensor[B, N, 3]): point queries.
                iters (int, optional): number of updates. Defaults to 4.
                is_train (bool, optional): enables training mode. Defaults to False.
        Returns:    - coords_predicted (FloatTensor[B, T, N, 2]):
                    - vis_predicted (FloatTensor[B, T, N]):
                    - train_data: `None` if `is_train` is false, otherwise:
                    - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                    - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                    - mask (BoolTensor[B, T, N]):
        """
        B, T, C, H, W = video.shape
        device = queries.device 
        assert H % self.stride == 0 and W % self.stride == 0
        assert T >= 2

        B, N, __ = queries.shape
        
        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype 
        queried_frames = queries[:,:,0].long()  # B x N
        queried_coords = queries[:,:,1:3]       # B x N x 2
        queried_coords = queried_coords / self.stride

        all_coords_predictions, all_vis_predictions, all_confidence_predictions = [], [], []
        
        C_ = C 
        H4, W4 = H//self.stride, W//self.stride

        # compute convolutional features for the video 
        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:,t:t+fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1,C_,H,W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B,T_chunk,C_chunk,H_chunk,W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1,C_chunk,H_chunk,W_chunk)
        else: 
            fmaps = self.fnet(video.reshape(-1,C_,H,W))
        fmaps = fmaps.permute(0,2,3,1)  # B*T x C_chunk x H4 x W4 ---> B*T x H4 x W4 x C_chunk
        # normalize the features
        fmaps = fmaps / torch.sqrt(
            torch.maximum(torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                            torch.tensor(1e-12, device=fmaps.device),))
        fmaps = fmaps.permute(0,3,1,2)  # B*T x H4 x W4 x C_chunk ---> B*T x C_chunk x H4 x W4
        fmaps = fmaps.reshape(B,-1,self.latent_dim,H//self.stride,W//self.stride) # B x T x C_chunk x H4 x W4
        fmaps = fmaps.to(dtype)

        # compute track features 
        fmaps_pyramid = []
        track_feat_pyramid = [] 
        track_feat_support_pyramid = []

        # save a feature pyramid with successive 2x2 avg pooling at each level 
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels-1):
            fmaps_ = fmaps.reshape(B*T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1])
            fmaps_ = F.avg_pool2d(fmaps_, kernel_size=2, stride=2)
            fmaps = fmaps_.reshape(B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1])
            fmaps_pyramid.append(fmaps)
        
        # get track features for every level of the feature pyramid 
        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i], queried_frames, queried_coords/2**i, support_radius=self.corr_radius
            )   # B x 1 x N x latent_dim, B x (2r+1)^2 x N x latent_dim
            track_feat_pyramid.append(track_feat.repeat(1,T,1,1))   # B x T x N x latent_dim
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1)) # B x 1 x (2r+1)^2 x N x latent_dim
        
        D_coords = 2
        coord_preds, vis_preds, confidence_preds = [], [], [] 
        vis = torch.zeros((B,T,N), device=device).float()
        confidence = torch.zeros((B,T,N), device=device).float()
        coords = queried_coords.reshape(B,1,N,2).expand(B,T,N,2).float()    # B T N 2

        r = 2*self.corr_radius + 1
        for it in range(iters):
            coords = coords.detach()            # B T N 2
            coords_init = coords.view(B*T,N,2)  # B*T N 2
            corr_embs = [] 
            corr_feats = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(fmaps_pyramid[i], coords_init/2**i)   # B T N 2r+1 2r+1 latent_dim
                track_feat_support = track_feat_support_pyramid[i].view(B,1,r,r,N,self.latent_dim) # B 1 2r+1 2r+1 N latent_dim
                track_feat_support = track_feat_support.squeeze(1) # B 2r+1 2r+1 N latent_dim
                track_feat_support = track_feat_support.permute(0,3,1,2,4) # B N 2r+1 2r+1 latent_dim
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )  # B T N 2r+1 2r+1 2r+1 2r+1
                corr_emb = self.corr_mlp(corr_volume.reshape(B*T*N, r*r*r*r)) # B*T*N (2r+1)^4
                corr_embs.append(corr_emb)
            corr_embs = torch.cat(corr_embs, dim=-1)  # B*T*N (2r+1)^4*corr_levels
            corr_embs = corr_embs.view(B,T,N,-1)        # B T N (2r+1)^4*corr_levels

            transformer_input = [vis[...,None], confidence[...,None], corr_embs]

            rel_coords_forward = coords[:,:-1] - coords[:,1:]       # B T-1 N 2
            rel_coords_forward = torch.nn.functional.pad(rel_coords_forward, (0,0,0,0,0,1)) # B T N 2
            rel_coords_backward = coords[:,1:] - coords[:,:-1]
            rel_coords_backward = torch.nn.functional.pad(rel_coords_backward, (0,0,0,0,1,0)) # B T N 2
            scale = (torch.tensor([self.model_resolution[1], self.model_resolution[0]], device=coords.device)/self.stride)
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                                       min_deg=0,max_deg=10) # B T N 84 ?? 
            transformer_input.append(rel_pos_emb_input)

            x = torch.cat(transformer_input, dim=-1).permute(0,2,1,3)   # B N T 84+1+1+corr_levels*(2r+1)^4
            x = x.reshape(B*N,T,-1)                                     # B*N T 84+1+1+corr_levels*(2r+1)^4
            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)                                     # B N T 84+1+1+corr_levels*(2r+1)^4

            delta = self.updateformer(x, add_space_attn=add_space_attn) # B N T 4
            
            delta_coords = delta[...,:D_coords].permute(0,2,1,3)        # B T N 2
            delta_vis = delta[...,D_coords].permute(0,2,1)              # B T N
            delta_confidence = delta[...,D_coords+1].permute(0,2,1)     # B T N

            vis = vis + delta_vis
            confidence = confidence + delta_confidence
            coords = coords + delta_coords
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

