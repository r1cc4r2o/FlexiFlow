
import torch
import numpy as np


def batch_rotation_matrices(alphas, betas, gammas):
    """Generate a batch of 3D rotation matrices for each batch in PyTorch."""
    cos = torch.cos
    sin = torch.sin

    R_x = torch.stack([
        torch.tensor([[1, 0, 0], 
                      [0, cos(a), -sin(a)], 
                      [0, sin(a), cos(a)]]) 
        for a in alphas
    ])

    R_y = torch.stack([
        torch.tensor([[cos(b), 0, sin(b)], 
                      [0, 1, 0], 
                      [-sin(b), 0, cos(b)]]) 
        for b in betas
    ])

    R_z = torch.stack([
        torch.tensor([[cos(g), -sin(g), 0], 
                      [sin(g), cos(g), 0], 
                      [0, 0, 1]]) 
        for g in gammas
    ])

    # Compute the final rotation matrix R = R_z @ R_y @ R_x (batch-wise)
    R = torch.bmm(R_z, torch.bmm(R_y, R_x))  # Shape: (4, 3, 3)
    return R


def random_rotate(batch):
    x = batch["x"]
    y = batch["y"]
    m = batch["mask"]

    alphas = torch.rand(len(x)) * 2.0*np.pi
    betas  = torch.rand(len(x)) * 2.0*np.pi
    gammas = torch.rand(len(x)) * 2.0*np.pi
    
    R = batch_rotation_matrices(alphas, betas, gammas)  # (N, 3, 3)
    
    # Apply rotation: Use batch matrix multiplication
    xr = torch.bmm(x, R.transpose(1, 2))  # (N, C, 3) @ (N, 3, 3)^T -> (N, C, 3)
    yr = torch.bmm(y, R.transpose(1, 2))  # (N, C, 3) @ (N, 3, 3)^T -> (N, C, 3)
    xr = xr * (~m).float().unsqueeze(-1)
    yr = yr * (~m).float().unsqueeze(-1)
    batch["x"] = xr


def prepare_input(batch, t, vocab):
    h = torch.nn.functional.one_hot(batch["c"][...,0], num_classes=len(vocab.atom_types)+1).to("cuda" if torch.cuda.is_available() else "cpu")
    h2 = torch.nn.functional.one_hot(batch["c"][...,1], num_classes=len(vocab.charge_types)+1).to("cuda" if torch.cuda.is_available() else "cpu")
    e = torch.nn.functional.one_hot(batch["a"], num_classes=len(vocab.bond_types)).to("cuda" if torch.cuda.is_available() else "cpu")
    t = t.view(-1, 1, 1).expand(-1, h.size(1), -1).to("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.cat((t, h.float()), dim=2)
    return batch["x"], features, h2, e, batch["mask"], batch["y"]


def nll_loss(y_hat, y, mask=None, ignore_index=-100, logits=True):
    y_hat_shape = y_hat.shape
    y_shape = y.shape
    y = y.ravel()
    y_hat = y_hat.view(-1, y_hat_shape[-1])
    if logits:
        y_hat_log = torch.nn.functional.log_softmax(y_hat, -1)
    else:
        y_hat_log = y_hat
    
    _nlls = torch.nn.functional.nll_loss(y_hat_log, y, ignore_index=ignore_index, reduction="none")
    
    if mask is None:
        _nlls = _nlls.view(y_hat_shape[:-1])
        _nlls = _nlls.mean(dim=tuple(range(1, len(_nlls.shape))))
        return _nlls

    _nlls = _nlls * (~mask).view(-1).float()
    _nlls = _nlls.view(y_hat_shape[:-1])
    _nlls = _nlls.sum(dim=tuple(range(1, len(_nlls.shape))))
    _den = (~mask).sum(dim=tuple(range(1, len(mask.shape))))
    _den = torch.clip(_den.float(), 1.)
    return _nlls / _den