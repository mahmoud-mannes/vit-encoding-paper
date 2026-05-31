import torch
def build_2d_sincos_pe(H, W, D, device):
    """
    Returns: [1, H*W, D]
    """

    assert D % 4 == 0, "D must be divisible by 4"

    d_half = D // 2
    d_quarter = D // 4

    # grid
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

    grid_y = grid_y.reshape(-1)  # [HW]
    grid_x = grid_x.reshape(-1)

    # frequencies
    omega = torch.arange(d_quarter, device=device) / d_quarter
    omega = 1.0 / (10000 ** omega)  # [d_quarter]

    # outer product → [HW, d_quarter]
    out_x = torch.einsum("n,d->nd", grid_x, omega)
    out_y = torch.einsum("n,d->nd", grid_y, omega)

    # sin/cos
    pe_x = torch.cat([out_x.sin(), out_x.cos()], dim=1)  # [HW, d_half]
    pe_y = torch.cat([out_y.sin(), out_y.cos()], dim=1)  # [HW, d_half]

    pe = torch.cat([pe_x, pe_y], dim=1)  # [HW, D]

    return pe.unsqueeze(0)  # [1, HW, D]