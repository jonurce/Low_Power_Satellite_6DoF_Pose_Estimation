import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


################ Load MMF Linear Kernel ################
mmf_linear_ext = load(
    name="mmf_linear_ext",
    sources=["classification/_2_train/cuda_kernels/mmf_linear_1.cu"],
    verbose=False
)

################ MMF Linear Function ################
class MMFLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, scale):
        ctx.save_for_backward(x, w, bias)
        ctx.scale = scale
        return mmf_linear_ext.mmf_linear(x, w, bias, scale)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, bias = ctx.saved_tensors

        grad_output_scaled = grad_output * ctx.scale

        grad_x    = grad_output_scaled @ w                       # [B, in_features]
        grad_w    = grad_output_scaled.t() @ x                   # [out_features, in_features]
        grad_bias = grad_output_scaled.sum(dim=0)                # [out_features]

        return grad_x, grad_w, grad_bias, None  # None for scale

################ MMF Linear Layer ################
class MMFLinear(nn.Module):
    def __init__(self, in_features, out_features, scale_init=1.0):
        super().__init__()
        # Ternary weights: -1, 0, +1 [C_out, C_in]
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Bias [C_out]
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable scaling factor (very important!) [scalar]
        # self.scale = nn.Parameter(torch.tensor(scale_init)) 

    def forward(self, x):
        # Snap weights to ternary via STE
        w = self.weight - (self.weight - self.weight.detach().sign()).detach()
        return MMFLinearFunction.apply(x, w, self.bias, 1.0)







################ Load MMF Conv2d Kernel ################
mmf_conv2d_ext = load(
    name="mmf_conv2d_ext",
    sources=["classification/_2_train/cuda_kernels/mmf_conv2d_1.cu"],
    verbose=False
)

################ MMF Conv2d Function ################
class MMFConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, scale, stride, padding):
        # Save what we need for backward
        ctx.save_for_backward(x, w, bias)
        ctx.scale   = scale
        ctx.stride  = stride
        ctx.padding = padding
        return mmf_conv2d_ext.mmf_conv2d(x, w, bias, scale, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, bias = ctx.saved_tensors
        # Fall back to PyTorch autograd for the backward pass
        # using F.conv2d which has a registered backward
        grad_output = grad_output * ctx.scale

        grad_x    = torch.nn.grad.conv2d_input(x.shape, w, grad_output, stride=ctx.stride, padding=ctx.padding)
        grad_w    = torch.nn.grad.conv2d_weight(x, w.shape, grad_output, stride=ctx.stride, padding=ctx.padding)
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        grad_scale = None  # handled by STE in the module

        return grad_x, grad_w, grad_bias, grad_scale, None, None
    
################ MMF Conv2d Layer ################
class MMFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_init=1.0):
        super().__init__()
        # Ternary weights: -1, 0, +1 [C_out, C_in, kH, kW]
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))

        # Bias [C_out]
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Learnable scaling factor (very important!) [scalar]
        # self.scale = nn.Parameter(torch.tensor(scale_init)) 

        # Stride and padding from Conv2d
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # STE: forward sees snapped ternary, gradients flow through self.weight
        w = self.weight - (self.weight - self.weight.detach().sign()).detach()
        return MMFConv2dFunction.apply(x, w, self.bias, 1.0, self.stride, self.padding)







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