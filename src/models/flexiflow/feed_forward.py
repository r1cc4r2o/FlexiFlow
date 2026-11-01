import torch

from src.models.flexiflow.layers import EquiNorm
from src.models.flexiflow.layers import InvNorm

"""
Masks are always boolean tensors, where a value of 1 indicates masking, and a value of 0 indicates 
no masking.
"""

class InvariantMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()
        
        d_ff = d_model * 4 if d_ff is None else d_ff

        self.Phi = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_coord_sets, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_model)
        )

    def forward(self, h, x_norm, h_norm):
        """
        Args:
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            x_norm (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h_norm (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated node features, shape [batch_size, n_nodes, d_model]
        """

        lengths = torch.linalg.vector_norm(x_norm, dim=-1).transpose(-2, -1)
        phi_input = torch.cat((h_norm, lengths), dim=-1)
        h_ff = h + self.Phi(phi_input)
        return h_ff


class EquivariantMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        super().__init__()
        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.Psi = torch.nn.Sequential(
            torch.nn.Linear(d_model, proj_sets),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(proj_sets, proj_sets)
        )
        self.W1 = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.W2 = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, x, x_norm, h_norm):
        """
        Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            x_norm (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h_norm (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated coord_sets, shape [batch_size, n_sets, n_nodes, 3]
        """
        # B: batch_size
        # S: n_sets
        # P: proj_sets
        # N: n_nodes

        psi = self.Psi(h_norm).unsqueeze(1)
        x_tilde = self.W1(x_norm.transpose(1, -1))
        # inv_feats shape [B, 1, N, P]
        # proj_sets shape [B, 3, N, P]

        attentions = x_tilde.unsqueeze(-2) * psi.unsqueeze(-1)
        # attentions shape [B, 1, N, P, 1] * [B, 3, N, 1, P] = [B, 3, N, P, P]

        attentions = attentions.sum(-1) # attentions shape [B, 3, N, P]
        x_ff = x + self.W2(attentions).transpose(1, -1) # attentions shape [B, N, P, 3]
        return x_ff
    


class pFeedForward(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, remove_com, d_ff=None, proj_sets=None, coord_norm="length"):
        super().__init__()
        self.phi_equi = EquiNorm(n_coord_sets, norm=coord_norm, remove_com=remove_com)
        self.phi_inv = InvNorm(d_model)
        self.invariant_mlp = InvariantMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(d_model, n_coord_sets, proj_sets=proj_sets)
        
    def forward(self, x, h, mask):
        """
         Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """
        x_norm = self.phi_equi(x, mask)
        h_norm = self.phi_inv(h)
        x_ff = self.equivariant_mlp(x, h, x_norm, h_norm)
        h_ff = self.invariant_mlp(x, h, x_norm, h_norm)
        return x_ff, h_ff



class FeedForward(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, remove_com, d_ff=None, proj_sets=None, coord_norm="length"):
        super().__init__()
        self.phi_equi = EquiNorm(n_coord_sets, norm=coord_norm, remove_com=remove_com)
        self.phi_inv = InvNorm(d_model)
        self.invariant_mlp = InvariantMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(d_model, n_coord_sets, proj_sets=proj_sets)
        
    def forward(self, x, y, h, hy, mask):
        """
         Args:
            x, y (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h, hy (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """
        x_norm = self.phi_equi(x, mask)
        y_norm = self.phi_equi(y, mask)
        h_norm = self.phi_inv(h)
        hy_norm = self.phi_inv(hy)
        x_ff = self.equivariant_mlp(x, x_norm, h_norm)
        y_ff = self.equivariant_mlp(y, y_norm, hy_norm)
        h_ff = self.invariant_mlp(h, x_norm, h_norm)
        hy_ff = self.invariant_mlp(hy, y_norm, hy_norm)
        return x_ff, y_ff, h_ff, hy_ff
    