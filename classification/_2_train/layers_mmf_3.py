import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


################ MMF Linear Function ################
class MMFLinearFunction(torch.autograd.Function):
    # Theory: X ∈ R^{M×N}, W ∈ R^{N×K}, b ∈ R^K
    # Code: X [M,N], W [K, N], b [K] 
    # Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b):
        # Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X - mu)                           # [M, N]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, N]

        # Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean()                          # scalar
        
        # w_tilde [K, N] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        # Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = y_tilde @ w_tilde.t() + b    # [M, K] <- [M, N] @ [N, K] + [K]

        # Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)

        return O # [M, K]

    # Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors

        # Step 2. On chip: dY <- dO × W^T
        dY = dO @ W # [M, N] <- [M, K] @ [K, N] 

        # Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        y_norm  = r * (X - mu)
        s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values
        Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act

        dsigma2 = (dY * (X - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        N  = X.shape[-1]
        dX = r * dY + 2 * dsigma2 * (X - mu) / N + dmu / N

        # Step 4: On chip: dW <- dO^T × Y_tilde
        dW = dO.t() @ Y_tilde   # [K, N] = [K, M] @ [M, N] 

        # Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=0)    # [K]

        # Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b)
        return dX, dW, db

################ MMF Linear Layer ################
class MMFLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # [K, N]
        self.bias = nn.Parameter(torch.zeros(out_features)) # [K]

    def forward(self, x):
        return MMFLinearFunction.apply(x, self.weight, self.bias)







################ MMF Conv2d Function ################
class MMFConv2dFunction(torch.autograd.Function):
    # Theory: X ∈ R^{B×C_in×H_in×W_in}, W ∈ R^{C_out×C_in×kH×kW}, b ∈ R^{C_out}
    # Code: X [B, C_in, H_in, W_in], W [C_out, C_in, kH, kW], b [C_out]
    ### Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b, stride, padding):

        # Before nothing, get shapes
        B, C_in, H_in, W_in = X.shape

        # Normalize over C_in per spatial position — reshape to [B*H_in*W_in, C_in]
        # so that dim=-1 normalization operates over C_in exactly as in Linear
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in] where M=B*H_in*W_in

        ### Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X_flat.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X_flat.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X_flat - mu)                           # [M, C_in]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, C_in]

        # Reshape y_tilde back to [B, C_in, H_in, W_in] for conv2d
        y_tilde_4d = y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean() # scalar
        
        # w_tilde [C_out, C_in, kH, kW] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        ### Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = F.conv2d(y_tilde_4d, w_tilde, b, stride, padding)  # [B, C_out, H_out, W_out]


        ### Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r, stride, padding)

        return O # [B, C_out, H_out, W_out]

    ### Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r, stride, padding = ctx.saved_tensors

        B, C_in, H_in, W_in = X.shape

        ### Step 2. On chip: dY <- dO × W^T
        # In conv2d, W^T means transposed convolution
        dY_4d = torch.nn.grad.conv2d_input(X.shape, W, dO, stride=stride, padding=padding)  # [B, C_in, H_in, W_in]

        # Reshape dY to [M, C_in] to match RMSNorm dimension: M = B * H_in * W_in
        dY = dY_4d.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in]

        ### Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)              # [M, C_in]

        y_norm  = r * (X_flat - mu)
        s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values
        Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act # [M, C_in]

        dsigma2 = (dY * (X_flat - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X_flat - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        # normalization dimension is C_in
        dX_flat = r * dY + 2 * dsigma2 * (X_flat - mu) / C_in + dmu / C_in # [M, C_in]

        # Reshape dX back to [B, C_in, H_in, W_in]
        dX = dX_flat.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 4: On chip: dW <- dO^T × Y_tilde
        # Reshape Y_tilde back to 4d for conv2d_weight
        Y_tilde_4d = Y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        dW = torch.nn.grad.conv2d_weight(Y_tilde_4d, W.shape, dO, stride=stride, padding=padding)  # [C_out, C_in, kH, kW]

        ### Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=(0, 2, 3))    # [C_out]

        ### Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b, stride, padding)
        return dX, dW, db, None, None   # None for stride, padding

################ MMF Conv2d Layer ################
class MMFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) 

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return MMFConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)








################ Tests ################
if __name__ == "__main__":
    print("Testing MMFLinear...")
    x = torch.randn(32, 128, device="cuda")  # batch 32, dim 128
    lin_mmf = MMFLinear(128, 64).cuda()
    out_mmf = lin_mmf(x)
    print("MMF output shape:", out_mmf.shape) # [32, 64]

    print("\n") 

    print("Testing MMFConv2d...")
    x = torch.randn(16, 3, 32, 32, device="cuda")  # batch 16, 3 channels, 32×32
    conv_mmf = MMFConv2d(3, 64, kernel_size=3, stride=1, padding=1).cuda()
    out_mmf = conv_mmf(x)
    print("MMF output shape:", out_mmf.shape)  # [16, 64, 32, 32]

    print("\n") 

    layer = MMFConv2d(3, 64, kernel_size=3, padding=1).cuda()
    bn = nn.BatchNorm2d(64).cuda()
    fc = nn.Linear(64, 10).cuda()

    x = torch.randn(4, 3, 32, 32, device='cuda')
    labels = torch.randint(0, 10, (4,), device='cuda')

    out = layer(x)           # [4, 64, 32, 32]
    out = bn(out)
    out = out.mean(dim=[2,3]) # [4, 64]
    out = fc(out)             # [4, 10]
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()

    print("grad norm:", layer.weight.grad.norm().item())
    print("grad std:", layer.weight.grad.std().item())
    # Check if all filters are still identical
    print("filter 0 vs filter 1 identical:", 
        layer.weight.grad[0].equal(layer.weight.grad[1]))