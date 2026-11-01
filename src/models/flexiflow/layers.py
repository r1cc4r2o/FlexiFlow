import torch


"""
Masks are always boolean tensors, where a value of 1 indicates masking, and a value of 0 indicates 
no masking.
"""

class EquiNorm(torch.nn.Module):
    """Coordinate normalisation layer for coordinate sets with inductive bias towards molecules

    This layer allows 3 different types of coordinate normalisation (defined in the norm argument):
        1. 'none' - The coordinates are zero-centred and multiplied by learnable weights
        2. 'gvp' - Coords are zero-centred, scaled by learnable weights and each is scaled by 
           sqrt(n_sets) / ||x_i||_2
        3. 'length' - Coords are zero-centred, multiplied by learnable weights and scaled by 
           1 / avg vector length

    Note that 'length' provides the same coordinate normalisation that is commonly used in current 
    models but adapted to multiple coordinate sets, thereby allowing easier comparison to existing 
    approaches.
    """

    def __init__(self, n_coord_sets, remove_com, norm="length", eps=1e-6):
        super().__init__()

        norm = "none" if norm is None else norm
        if norm not in ["none", "gvp", "length"]:
            raise ValueError(f"Unknown normalisation type '{norm}'")

        self.n_coord_sets = n_coord_sets
        self.norm = norm
        self.eps = eps
        self.remove_com = remove_com
        self.set_weights = torch.nn.Parameter(torch.ones((1, n_coord_sets, 1, 1)))

    def forward(self, x, mask):
        """Apply coordinate normlisation layer

        Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep

        Returns:
            torch.Tensor: Normalised coords, shape [batch_size, n_sets, n_nodes, 3]
        """

        inv_mask = (~mask).unsqueeze(-1).float()
        n_atoms = inv_mask.sum(dim=-2, keepdim=True)
        n_atoms = torch.clamp(n_atoms, min=1.)
        
        if self.remove_com:
            com = (x * inv_mask).sum(dim=-2, keepdim=True) / n_atoms
            x = (x - com) * inv_mask
        else:
            x = x * inv_mask
        # END COM
        
        lengths = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

        if self.norm == "length":
            scaled_lengths = lengths.sum(dim=-2, keepdim=True) / n_atoms
            coord_div = scaled_lengths + self.eps

        elif self.norm == "gvp":
            coord_div = (lengths + self.eps) / self.n_coord_sets**0.5

        else:
            coord_div = torch.ones_like(x)

        x = (x * self.set_weights) / coord_div
        x = x * inv_mask
        return x

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

  
class InvNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = torch.nn.Sequential(torch.nn.LayerNorm(d_model),)
        
    def forward(self, x):
        return self.norm(x)


class BondRefine(torch.nn.Module):
    def __init__(self, d_model, d_message, d_edge, d_ff, remove_com):
        super().__init__()

        d_ff = d_message if d_ff is None else d_ff
        in_feats = (2 * d_message) + d_edge + 2

        self.coord_norm = EquiNorm(1, norm="none", remove_com=remove_com)
        self.node_norm = InvNorm(d_model)
        self.edge_norm = InvNorm(d_edge)

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_edge)
        )

    def forward(self, x, h, e, mask):
        """Refine the bond predictions with a message passing layer that only updates bonds

        Args:
            x (torch.Tensor): Coordinate tensor without coord sets, shape [batch_size, n_nodes, 3] # [B, S, N, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            e (torch.Tensor): Current edge features, shape [batch_size, n_nodes, n_nodes, d_edge]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_nodes], 1 mask, 0 keep

        Returns:
            torch.Tensor: Bond predictions tensor, shape [batch_size, n_nodes, n_nodes, n_bond_types]
        """
        assert len(x.shape) == 3

        batch_size, n_nodes, _ = tuple(h.shape)

        x_norm = self.coord_norm(x.unsqueeze(1), mask.unsqueeze(1)).squeeze(1)
        diffs = x_norm.unsqueeze(2) - x_norm.unsqueeze(1)
        dists = (diffs * diffs).sum(dim=-1).unsqueeze(-1)
        dotprods = torch.bmm(x_norm, x_norm.transpose(1, 2)).unsqueeze(-1)

        h_norm = self.node_proj(self.node_norm(h))
        h_norm_i = h_norm.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        h_norm_j = h_norm.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        h_norm_ij = torch.cat((h_norm_i, h_norm_j), dim=-1) 

        e_norm = self.edge_norm(e)
        in_feats = torch.cat((h_norm_ij, dists, dotprods, e_norm), dim=3)
        return self.message_mlp(in_feats)