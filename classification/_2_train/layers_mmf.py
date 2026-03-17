import torch
import torch.nn as nn
import torch.nn.functional as F

################ MatMulFree Linear layer ################
class MMFLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        # Ternary weights: -1, 0, +1 [C_out, C_in]
        self.weight = nn.Parameter(torch.randint(-1, 2, (out_features, in_features)).float())

        # Bias [C_out]
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable scaling factor (very important!) [scalar]
        self.scale = nn.Parameter(torch.tensor(scale)) 

    def forward(self, x):
        # x: [B, C_in], weight: [C_out, C_in], out: [B, C_out] = [B, C_in] * [C_out, C_in].t()
        B, I = x.shape
        O = self.weight.shape[0]

        # Instead of x @ W.t() we do element-wise addition/subtraction
        
        # Create masks for +1 and -1 weights: [1 where +1, 0 elsewhere] and [1 where -1, 0 elsewhere]
        # [C_out, C_in]

        # Positive indices: where weight == +1
        pos_mask = self.weight == 1                  # [O, I] bool
        pos_indices = pos_mask.nonzero(as_tuple=True)  # tuple of two tensors: (rows, cols) indices from weight [O, I]
        

        # Negative indices: where weight == -1
        neg_mask = self.weight == -1                 # [O, I] bool
        neg_indices = neg_mask.nonzero(as_tuple=True) # tuple of two tensors: (rows, cols) indices from weight [O, I]

        # Sum positive contributions (pure addition) [B, O]
        out_pos = torch.zeros(B, O, device=x.device)
        if pos_indices[0].numel() > 0:
            # x [B, I], out_pos [B, O]
            # pos_indices[0]: row indices (output channels) where weight == +1 [num_pos]
            # pos_indices[1]: column indices (input features) where weight == +1 [num_pos]
            # x[:, pos_indices[1]]: [B, num_pos] — selects only the columns (input features) that have +1 weight
            # tensor.index_add_(dim, index, source) adds values from source into tensor at positions given by index along dim
            out_pos.index_add_(1, pos_indices[0], x[:, pos_indices[1]])

        # Sum negative contributions (pure subtraction) [B, O]
        out_neg = torch.zeros(B, O, device=x.device)
        if neg_indices[0].numel() > 0:
            out_neg.index_add_(1, neg_indices[0], x[:, neg_indices[1]])

        # Final: scale * (positive_sum - negative_sum) + bias
        # [B, O] = scalar * ([B, O] - [B, O]) + [B, O]
        out = self.scale * (out_pos - out_neg) + self.bias[None, :]

        return out

################ Sample MatMulFree Conv2d layer ################
class MMFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_init=1.0, chunk_size=32):
        super().__init__()
        # Ternary weights: -1, 0, +1 [C_out, C_in, kH, kW]
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randint(-1, 2, weight_shape).float())

        # Bias [C_out]
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Learnable scaling factor (very important!) [scalar]
        self.scale = nn.Parameter(torch.tensor(scale_init)) 

        # Stride and padding from Conv2d
        self.stride = stride
        self.padding = padding
        self.chunk_size = chunk_size

    def forward(self, x):
        B, C_in, H_in, W_in = x.shape
        C_out, _, kH, kW = self.weight.shape

        # Compute out shape values
        H_out = (H_in + 2*self.padding - kH) // self.stride + 1
        W_out = (W_in + 2*self.padding - kW) // self.stride + 1
        N_out = H_out * W_out

        # Snap weights to ternary {-1, 0, +1} without storing extra tensors
        # Using STE (straight-through estimator): forward uses snapped, backward flows through
        w = self.weight - (self.weight - self.weight.detach().sign()).detach()

        # (C_out, C_in, kH, kW) indices
        pos_indices = (w == 1).nonzero(as_tuple=True) # tuple of four tensors: (C_out, C_in, kH, kW) indices from weights [num_pos]
        neg_indices = (w == -1).nonzero(as_tuple=True) # tuple of four tensors: (C_out, C_in, kH, kW) indices from weights [num_neg]

        # Unfold input to [B, C_in * kH * kW, N_out]
        # unfolded = F.unfold(x, kernel_size=(kH, kW), stride=self.stride, padding=self.padding)

        # Precompute flat channel indices for +1 and -1: = c_in * spatial_size + kh * kW + kw
        # pos_flat = (pos_indices[1] * (kH*kW) + pos_indices[2]*kW + pos_indices[3])  # [num_pos]
        # neg_flat = (neg_indices[1] * (kH*kW) + neg_indices[2]*kW + neg_indices[3])  # [num_neg]

        out = torch.zeros(B, C_out, N_out, device=x.device, dtype=x.dtype)

        # Process output channels in chunks to limit intermediate tensor sizes
        for start in range(0, C_out, self.chunk_size):
            end = min(start + self.chunk_size, C_out)

            # --- Positive weights for this chunk ---
            pos_mask = (pos_indices[0] >= start) & (pos_indices[0] < end)
            if pos_mask.any():
                pos_cout = pos_indices[0][pos_mask] - start          # local c_out index [num_pos_chunk]
                pos_cin  = pos_indices[1][pos_mask]                   # [num_pos_chunk]
                pos_kh   = pos_indices[2][pos_mask]                   # [num_pos_chunk]
                pos_kw   = pos_indices[3][pos_mask]                   # [num_pos_chunk]

                # Only unfold the C_in channels actually needed by this chunk
                needed_cin = torch.unique(pos_cin)
                cin_to_local = torch.full((C_in,), -1, dtype=torch.long, device=x.device)
                cin_to_local[needed_cin] = torch.arange(needed_cin.numel(), device=x.device)

                # [B, len(needed_cin)*kH*kW, N_out] — much smaller unfold
                unfolded_pos = F.unfold(
                    x[:, needed_cin, :, :],
                    kernel_size=(kH, kW),
                    stride=self.stride,
                    padding=self.padding
                )

                # Flat indices into the partial unfold
                local_cin = cin_to_local[pos_cin]
                pos_flat = local_cin * (kH * kW) + pos_kh * kW + pos_kw  # [num_pos_chunk]

                # Scatter-add directly into the output slice (in-place)
                chunk_size_actual = end - start
                idx = pos_cout.unsqueeze(0).unsqueeze(-1).expand(B, -1, N_out)
                out[:, start:end, :].scatter_add_(1, idx, unfolded_pos[:, pos_flat, :])

            # --- Negative weights for this chunk ---
            neg_mask = (neg_indices[0] >= start) & (neg_indices[0] < end)
            if neg_mask.any():
                neg_cout = neg_indices[0][neg_mask] - start
                neg_cin  = neg_indices[1][neg_mask]
                neg_kh   = neg_indices[2][neg_mask]
                neg_kw   = neg_indices[3][neg_mask]

                needed_cin = torch.unique(neg_cin)
                cin_to_local = torch.full((C_in,), -1, dtype=torch.long, device=x.device)
                cin_to_local[needed_cin] = torch.arange(needed_cin.numel(), device=x.device)

                unfolded_neg = F.unfold(
                    x[:, needed_cin, :, :],
                    kernel_size=(kH, kW),
                    stride=self.stride,
                    padding=self.padding
                )

                local_cin = cin_to_local[neg_cin]
                neg_flat = local_cin * (kH * kW) + neg_kh * kW + neg_kw

                idx = neg_cout.unsqueeze(0).unsqueeze(-1).expand(B, -1, N_out)
                # Negate directly — no separate out_neg buffer
                out[:, start:end, :].scatter_add_(1, idx, -unfolded_neg[:, neg_flat, :])


        """
        # Positive contributions (pure addition) [B, C_out, N_out]
        # out_pos = torch.zeros(B, C_out, N_out, device=x.device)
        if pos_indices[0].numel() > 0:
            # x [B, C_in, H_in, W_in], out_pos [B, C_out, N_out]
            # pos_indices = (c_out_idx, c_in_idx, kh_idx, kw_idx) each of them a tensor of shape [num_pos]
            # unfolded[:, pos_flat, :]: [B, num_pos, N_out] — selects only the indices (input features) that have +1 weight
            # tensor.index_add_(dim, index, source) adds values from source into tensor at positions given by index along dim
            # out_pos.index_add_(1, pos_indices[0], unfolded[:, pos_flat, :])
            out.scatter_add_(
                1, 
                pos_indices[0].unsqueeze(0).unsqueeze(-1).expand(B, -1, N_out),  # broadcast c_out indices
                unfolded[:, pos_flat, :]                             # [B, num_pos, N_out]
            )

        # Negative contributions (pure addition) [B, C_out, N_out]
        # out_neg = torch.zeros(B, C_out, N_out, device=x.device)
        if neg_indices[0].numel() > 0:
            # out_neg.index_add_(1, neg_indices[0], unfolded[:, neg_flat, :])
            out.scatter_add_(
                1, 
                neg_indices[0].unsqueeze(0).unsqueeze(-1).expand(B, -1, N_out),  # broadcast c_out indices
                -unfolded[:, neg_flat, :]                             # [B, num_neg, N_out]
            )
        """    

        # Final: scale * (pos - neg) + bias
        # [B, C_out, N_out] = scalar * ([B, C_out, N_out] - [B, C_out, N_out]) + [C_out]
        out = self.scale * out + self.bias.view(1, -1, 1)

        # Reshape out to: [B, C_out, H_out, W_out]
        out = out.view(B, C_out, H_out, W_out)


        # Instead of unfold + index_add_
        """
        out = torch.zeros(B, C_out, H_out, W_out, device=x.device) # [B, C_out, H_out, W_out]

        for c_out_idx, c_in_idx, kh_idx, kw_idx in zip(*pos_indices):
            # Slide the kernel position over the output grid
            for oh in range(H_out):
                for ow in range(W_out):
                    ih = oh * self.stride + kh_idx - self.padding
                    iw = ow * self.stride + kw_idx - self.padding
                    if 0 <= ih < H_in and 0 <= iw < W_in:
                        patch_val = x[:, c_in_idx, ih, iw]  # [B]
                        out[:, c_out_idx, oh, ow] += patch_val

        for c_out_idx, c_in_idx, kh_idx, kw_idx in zip(*neg_indices):
            # Slide the kernel position over the output grid
            for oh in range(H_out):
                for ow in range(W_out):
                    ih = oh * self.stride + kh_idx - self.padding
                    iw = ow * self.stride + kw_idx - self.padding
                    if 0 <= ih < H_in and 0 <= iw < W_in:
                        patch_val = x[:, c_in_idx, ih, iw]  # [B]
                        out[:, c_out_idx, oh, ow] -= patch_val

        # Final: scale * (pos - neg) + bias
        # [B, C_out, H_out, W_out] = scalar * [B, C_out, H_out, W_out] + [1, C_out, 1, 1]
        out = self.scale * out + self.bias.view(1, -1, 1, 1)

        """

        # Safety clamp, to prevent inf/Nan
        out = torch.clamp(out, min=-1e4, max=1e4)

        return out


################ Tests ################

print("Testing MMFLinear...")
x = torch.randn(4, 64)          # batch 4, dim 64
lin_mmf = MMFLinear(64, 128)
out_mmf = lin_mmf(x)
print("MMF output shape:", out_mmf.shape)
print("\n") 

print("Testing MMFConv2d...")
x = torch.randn(2, 3, 32, 32)  # batch 2, 3 channels, 32×32
conv_mmf = MMFConv2d(3, 64, 3, padding=1)
out_mmf = conv_mmf(x)
print("MMF output shape:", out_mmf.shape)  # [2, 64, 32, 32]