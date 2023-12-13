import torch
import torch.nn as nn

class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""
    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift
    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones

if __name__ == '__main__':
    from torchinfo import summary

    batch_size = 1
    in_channel = 1
    fs = int(16e3)
    T = 256
    L = 512
    # N, C, H, W = [batch_size, in_channel, 5/0.05=100, 1000]
    model = TorchOLA(frame_shift=256)
    summary(model, input_size=[batch_size, in_channel, T, L])  # 输出网络结构 [B, C, N]