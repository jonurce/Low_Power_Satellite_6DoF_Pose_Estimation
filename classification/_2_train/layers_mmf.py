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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_init=1.0):
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

    def forward(self, x):
        B, C_in, H_in, W_in = x.shape
        C_out = self.weight.shape[0]
        kH, kW = self.weight.shape[2], self.weight.shape[3]

        # Unfold input to [B, C_in * kH * kW, H_out * W_out]
        unfolded = F.unfold(x, kernel_size=(kH, kW), stride=self.stride, padding=self.padding)

        # Compute out shape values
        H_out = (H_in + 2*self.padding - kH) // self.stride + 1
        W_out = (W_in + 2*self.padding - kW) // self.stride + 1
        N_out = H_out * W_out

        # (C_out, C_in, kH, kW) indices
        pos_indices = (self.weight == 1).nonzero(as_tuple=True) # tuple of four tensors: (C_out, C_in, kH, kW) indices from weights [num_pos]
        neg_indices = (self.weight == -1).nonzero(as_tuple=True) # tuple of four tensors: (C_out, C_in, kH, kW) indices from weights [num_neg]

        # Positive contributions (pure addition) [B, C_out, N_out]
        out_pos = torch.zeros(B, C_out, N_out, device=x.device)
        if pos_indices[0].numel() > 0:
            # x [B, C_in, H_in, W_in], out_pos [B, C_out, N_out]
            # pos_indices[0]: C_out indices (output channels) where weight == +1 [num_pos]
            # pos_indices[1]: C_in indices (input features) where weight == +1 [num_pos]
            # unfolded[:, pos_indices[1], :]: [B, num_pos, N_out] — selects only the indices (input features) that have +1 weight
            # tensor.index_add_(dim, index, source) adds values from source into tensor at positions given by index along dim
            out_pos.index_add_(1, pos_indices[0], unfolded[:, pos_indices[1], :])

        # Negative contributions
        out_neg = torch.zeros(B, C_out, N_out, device=x.device)
        if neg_indices[0].numel() > 0:
            out_neg.index_add_(1, neg_indices[0], unfolded[:, neg_indices[1], :])

        # Final: scale * (pos - neg) + bias
        # [B, C_out, N_out] = scalar * ([B, C_out, N_out] - [B, C_out, N_out]) + [C_out]
        out = self.scale * (out_pos - out_neg) + self.bias.view(1, -1, 1)

        # Reshape out to: [B, C_out, H_out, W_out]
        out = out.view(B, C_out, H_out, W_out)

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