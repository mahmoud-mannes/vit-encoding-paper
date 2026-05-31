import torch

# split q,k into 4 chunks
def rotate_half(a, b):
    return -b, a

def rope_split(x, quarter):
    # x: [B, heads, T, d]
    x_cls = x[:, :, :1, :]
    x_patch = x[:, :, 1:, :]

    x1, x2, x3, x4 = torch.split(x_patch, quarter, dim=-1)
    return x_cls, x1, x2, x3, x4

def rope_combine(x_cls, x1, x2, x3, x4):
    x_patch = torch.cat([x1, x2, x3, x4], dim=-1)
    return torch.cat([x_cls, x_patch], dim=2)

def rope_apply_tensor(x, sin_x, cos_x, sin_y, cos_y, quarter):
    x_cls, x1, x2, x3, x4 = rope_split(x, quarter)

    # X rotation
    rx1, rx2 = rotate_half(x1, x2)
    x1 = x1 * cos_x + rx1 * sin_x
    x2 = x2 * cos_x + rx2 * sin_x

    # Y rotation
    rx3, rx4 = rotate_half(x3, x4)
    x3 = x3 * cos_y + rx3 * sin_y
    x4 = x4 * cos_y + rx4 * sin_y

    return rope_combine(x_cls, x1, x2, x3, x4)

def apply_2d_rope(q, k, H, W, dim):
    device = q.device
    B, n_heads, T, d = q.shape

    assert d % 4 == 0, "d_head must be divisible by 4"

    quarter = d // 4

    # positions
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    # frequencies
    freqs = torch.arange(quarter, device=device) / quarter
    freqs = 1.0 / (10000 ** freqs)

    # angles
    angles_x = torch.einsum("t,d->td", grid_x, freqs)
    angles_y = torch.einsum("t,d->td", grid_y, freqs)

    sin_x = angles_x.sin()[None, None, :, :]
    cos_x = angles_x.cos()[None, None, :, :]

    sin_y = angles_y.sin()[None, None, :, :]
    cos_y = angles_y.cos()[None, None, :, :]

    # apply to q and k
    q = rope_apply_tensor(q, sin_x, cos_x, sin_y, cos_y, quarter)
    k = rope_apply_tensor(k, sin_x, cos_x, sin_y, cos_y, quarter)

    return q, k