import torch
import copy

from src.models.flexiflow.layers import EquiNorm
from src.models.flexiflow.layers import InvNorm
from src.models.flexiflow.layers import BondRefine
from src.models.flexiflow.feed_forward import FeedForward, pFeedForward
from src.models.flexiflow.graph_attention import GraphAttention
from src.models.flexiflow.utils import edges_from_nodes


class FFLayer(torch.nn.Module):
    def __init__(self, d_model, 
                       d_message,
                       n_coord_sets, 
                       d_ff,
                       d_message_hidden,
                       d_edge,
                       edge_in,
                       edge_out,
                       n_attn_heads,
                       proj_sets,
                       coord_norm, 
                       eps,
                       remove_com,
                       protein_conditioning,
                       gramian):
        super().__init__()

        self.feed_forward = FeedForward(d_model, n_coord_sets, 
                                        d_ff=d_ff, proj_sets=d_message, 
                                        coord_norm=coord_norm, remove_com=remove_com)
        if protein_conditioning:
            self.pfeed_forward = pFeedForward(d_model, n_coord_sets, 
                                            d_ff=d_ff, proj_sets=d_message, 
                                            coord_norm=coord_norm, remove_com=remove_com)
        self.graph_attention = GraphAttention(d_model,
                                    d_message,
                                    n_coord_sets,
                                    proj_sets,
                                    remove_com=remove_com,
                                    n_attn_heads=n_attn_heads,
                                    d_message_hidden=d_message_hidden,
                                    d_edge=d_edge,
                                    edge_in=edge_in,
                                    edge_out=edge_out,
                                    coord_norm=coord_norm,
                                    eps=eps,
                                    protein_conditioning=protein_conditioning,
                                    gramian=gramian)
        self.protein_conditioning = protein_conditioning


    def forward(self, x, y, h, adj_matrix, y_adj_matrix, e, ey, mask, px=None, ph=None, pmask=None, hy=None):
        """
         Args:
            x, y (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h, hy (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            adj_matrix, y_adj_matrix (torch.Tensor): Adjacency matrices for the graphs, shape [batch_size, n_nodes, n_nodes]
            e (torch.Tensor): Edge feature tensor, shape [batch_size, n_nodes, n_nodes, d_edge]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep
            px, ph, pmask (torch.Tensor, optional): Protein conditioning inputs, default None

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """
        x_ff, y_ff, h_ff, hy_ff = self.feed_forward(x, y, h, h, mask) if hy is None else self.feed_forward(x, y, h, hy, mask)
        px_ff, hp_ff = [None, None]
        if (self.protein_conditioning) and (px is not None):
            px_ff, hp_ff  = self.pfeed_forward(px, ph, pmask)
        x_ff, y_ff, h_ff, hy_ff, e_ff, ey_ff = self.graph_attention(x_ff, y_ff, h_ff, hy_ff, adj_matrix, 
                                                                    y_adj_matrix, e, ey, mask, 
                                                                    px_ff, hp_ff, pmask)
        return x_ff, y_ff, h_ff, hy_ff, e_ff, ey_ff
        

class FlexiFlow(torch.nn.Module):
    def __init__(self, n_layers,
                       d_model, 
                       d_message,
                       n_coord_sets,
                       atom_vocab_size,
                       charge_vocab_size,
                       gramian,
                       edge_vocab_size=None,
                       size_emb=64,
                       max_atoms=256,
                       d_ff=None,
                       d_message_hidden=None,
                       d_edge=None,
                       n_attn_heads=None,
                       proj_sets=None,
                       coord_norm="length", 
                       eps=1e-6,
                       bond_refine=True,
                       self_conditioning=False,
                       strict=True, 
                       remove_com=True,
                       protein_conditioning=False,
                       patom_vocab_size=None,
                       max_patoms=256):
        self.params = {}
        for key, value in locals().items():
            if key not in ["self", "__class__"]:
                self.params[key] = value
        super().__init__()

        extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("`n_layers` is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        self.params["n_attn_heads"] = n_attn_heads

        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            if strict:
                raise ValueError(f"`n_attn_heads` must exactly divide `d_model`, got {n_attn_heads} and {d_model}")
            else:
                d_model = (d_model // n_attn_heads) * n_attn_heads
                self.params["d_model"] = d_model


        if d_edge is not None or edge_vocab_size is not None:
            if None in [d_edge, edge_vocab_size]:
                raise ValueError("If either d_edge or edge_vocab_size are given both must be provided.")

            edge_in_feats = edge_vocab_size * 2 if self_conditioning else edge_vocab_size

            self.edge_in_proj = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, d_edge)
            )
            self.edge_out_proj = torch.nn.Sequential(
                torch.nn.Linear(d_edge, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, edge_vocab_size)
            )

        in_feats = atom_vocab_size * 2 + 1 if self_conditioning else atom_vocab_size + 1
        in_feats = in_feats + size_emb
        
        self.size_emb = torch.nn.Embedding(max_atoms, size_emb)
        self.feat_proj = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, d_model)
        )
        ff_layers = []
        for i in range(n_layers):
            if ((i == 0) or (i == n_layers -1)) and d_edge is not None:
                d_m_h = None
                d_e = d_edge
                edge_in = i == 0
                edge_out = i == n_layers -1

            else:
                d_m_h = d_message_hidden
                d_e = None
                edge_in = False
                edge_out = False

            ff_layer = FFLayer(d_model=d_model,
                                    d_message=d_message,
                                    n_coord_sets=n_coord_sets, 
                                    d_ff=d_ff,
                                    d_message_hidden=d_m_h,
                                    d_edge=d_e, 
                                    edge_in=edge_in,
                                    edge_out=edge_out,
                                    n_attn_heads=n_attn_heads,
                                    proj_sets=proj_sets,
                                    coord_norm=coord_norm, 
                                    eps=eps,
                                    remove_com=remove_com,
                                    protein_conditioning=protein_conditioning,
                                    gramian=gramian)
            ff_layers.append(ff_layer)

        self.ff_layers = torch.nn.ModuleList(ff_layers)

        self.final_ff_block = FeedForward(d_model, n_coord_sets, d_ff=d_ff, 
                                          proj_sets=proj_sets, coord_norm=coord_norm,
                                          remove_com=remove_com)
        self.coord_norm_x = EquiNorm(n_coord_sets, remove_com=remove_com, norm=coord_norm, eps=eps)
        self.feat_norm = InvNorm(d_model)

        in_coord_sets = 2 if self_conditioning else 1
        self.coord_proj_x = torch.nn.Linear(in_coord_sets, n_coord_sets, bias=False)            
        self.coord_head_x = torch.nn.Linear(n_coord_sets, 1, bias=False)

        if d_edge is not None:
            self.bond_norm = InvNorm(d_edge)

        if bond_refine:
            self.refine_layer = BondRefine(d_model, d_message, d_edge, d_ff, remove_com=remove_com)

        self.atom_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, atom_vocab_size)
        )
        self.charge_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, charge_vocab_size)
        )
        if protein_conditioning:
            self.psize_emb = torch.nn.Embedding(max_patoms, size_emb)
            if self_conditioning:
                self.pcoord_proj = torch.nn.Linear(1, n_coord_sets, bias=False)
                in_pfeats = patom_vocab_size + size_emb
                self.pfeat_proj = torch.nn.Sequential(
                    torch.nn.Linear(in_pfeats, d_model),
                    torch.nn.SiLU(inplace=False),
                    torch.nn.Linear(d_model, d_model)
                )
            else:
                print(patom_vocab_size ,size_emb, in_feats, size_emb+in_feats)
                self.pcoord_proj = copy.deepcopy(self.coord_proj_x)
                in_pfeats = patom_vocab_size + size_emb
                self.pfeat_proj = torch.nn.Sequential(
                    torch.nn.Linear(in_pfeats, d_model),
                    torch.nn.SiLU(inplace=False),
                    torch.nn.Linear(d_model, d_model)
                )


    def forward(
        self,
        x,
        y,
        h,
        e=None,
        mask=None,
        given_x=None,
        given_h=None,
        given_e=None,
        px = None,
        ph = None,
        pmask = None
    ):
        """Predict molecular coordinates and atom types

        Args:
            x, y (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            h (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, n_feats] 
            e (torch.Tensor): In edge features, shape [batch_size, n_atoms, n_atoms, edge_vocab_size]
            mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 mask, 0 keep
            given_x (torch.Tensor): Conditional coords, shape [batch_size, n_atoms, 3]
            given_h (torch.Tensor): Conditional atom type logits, shape [batch_size, n_atoms, n_feats]
            given_e (torch.Tensor): Cond bond type logits, shape [batch_size, n_atoms, n_atoms, edge_vocab_size]
            px (torch.Tensor): Input coordinates, shape [batch_size, n_protein_atoms, 3]
            ph (torch.Tensor): Invariant atom features, shape [batch_size, n_protein_atoms, n_feats] 
            pmask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_protein_atoms], 1 mask, 0 keep

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charges)
            All torch.Tensor, shapes:
                Coordinates: [batch_size, n_atoms, 3]
                Type logits: [batch_size, n_atoms, vocab_size],
                Bond logits: [batch_size, n_atoms, n_atoms, edge_vocab_size]
                Charge logits: [batch_size, n_atoms, 7] #not good!
        """
        self_cond = self.params["self_conditioning"]
        remove_com = self.params["remove_com"]

        if e is not None and self.params["d_edge"] is None:
            raise ValueError("`e` was provided but the model was initialised with `d_edge` as None.")

        if e is None and self.params["d_edge"] is not None:
            raise ValueError("The model was initialised with `d_edge` but no edge feats `e` were provided to forward.")

        if given_x is not None and not self_cond:
            raise ValueError("`given_x` was provided but the model was initialised with `self_conditioning` as False.")

        if given_x is None and self_cond:
            raise ValueError("The model was initialsed with `self_conditioning` but `given_x` was not provided.")

        if e is None and given_e is not None:
            raise ValueError("`e` must be provided if using bond conditioning.")

        mask = torch.zeros_like(x[..., 0], dtype=torch.bool) if mask is None else mask
        inv_mask = (~mask.unsqueeze(-1)).float()
        
        if (self.params["protein_conditioning"]) and (px is not None):
            x_and_px = torch.cat([x, px], dim = -2)
            inv_pmask = (~pmask.unsqueeze(-1)).float()
            inv_mask_x_and_px = torch.cat([inv_mask, inv_pmask], dim = -2)
            n_ligand_atoms = x.shape[1]
            adj_full = edges_from_nodes(x_and_px, node_mask=inv_mask_x_and_px[...,0])
            adj_matrix = adj_full[:,:n_ligand_atoms,:]
            px = self.pcoord_proj(px.unsqueeze(0).movedim(0, -1)).movedim(-1, 1)
            px = px * inv_pmask[:, None]
            pmask = pmask.unsqueeze(1)
            n_patoms = inv_pmask.sum(dim=-2)
            n_patoms = torch.clamp(n_patoms, min=1.)
            psize_emb = self.psize_emb(n_patoms.long()).expand(-1, ph.size(1), -1)
            ph = torch.cat((ph, psize_emb), dim=-1)
            ph = self.pfeat_proj(ph)
        else:
            adj_matrix = edges_from_nodes(x, node_mask=inv_mask[...,0])
        
        n_atoms = inv_mask.sum(dim=-2)
        n_atoms = torch.clamp(n_atoms, min=1.)
        size_emb = self.size_emb(n_atoms.long()).expand(-1, h.size(1), -1)

        h = torch.cat((h, size_emb), dim=-1)
        if given_h is not None:
            h = torch.cat((h, given_h), dim=-1)
        h = self.feat_proj(h)

        if e is not None:
            e = e.float()
            e = torch.cat((e, given_e), dim=-1) if given_e is not None else e
            e = self.edge_in_proj(e)

        x = torch.stack((x, given_x)) if self_cond else x.unsqueeze(0)
        y = torch.stack((y, given_x)) if self_cond else y.unsqueeze(0)
        x = self.coord_proj_x(x.movedim(0, -1)).movedim(-1, 1)
        y = self.coord_proj_x(y.movedim(0, -1)).movedim(-1, 1)
        x = x * inv_mask[:, None]
        y = y * inv_mask[:, None]

        mask = mask.unsqueeze(1)
        for _id_ff_layer, ff_layer in enumerate(self.ff_layers):
            x, y, h, hy, e, ey = ff_layer(x, y, h, adj_matrix, adj_matrix, e, e, mask, px, ph, pmask, hy=h+h) if _id_ff_layer == 0 \
                else ff_layer(x, y, h, adj_matrix, adj_matrix, e, ey, mask, px, ph, pmask, hy=hy)

        xc, yc, hc, hyc = self.final_ff_block(x, y, h, hy, mask)
        x = xc - x
        y = yc - y
        h = hc - h
        hy = hyc - hy
        y = self.coord_norm_x(y, mask)
        x = self.coord_norm_x(x, mask)
        x = self.coord_head_x(x.transpose(1, -1))
        y = self.coord_head_x(y.transpose(1, -1))
        x = x.transpose(1, -1).squeeze(1) 
        y = y.transpose(1, -1).squeeze(1) 

        x_hat = x
        y_hat = y
        
        if self.params["bond_refine"]:
            e = self.refine_layer(x, h, e, mask[:, 0, :]) 
            ey = self.refine_layer(y, hy, ey, mask[:, 0, :])

        h_hat = self.feat_norm(h)
        hy_hat = self.feat_norm(hy)

        if self.params["d_edge"] is not None:
            e_hat = self.bond_norm(e)
            ey_hat = self.bond_norm(ey) 
        else:
            e_hat = None
            ey_hat = None

        atom_logits = self.atom_classifier_head(h_hat)
        atom_logits_y = self.atom_classifier_head(hy_hat)

        charge_logits = self.charge_classifier_head(h_hat)
        charge_logits_y = self.charge_classifier_head(hy_hat)

        if e_hat is not None:
            e_hat = e_hat + e_hat.transpose(1, 2)
            e_logits = self.edge_out_proj(e_hat)
            ey_hat = ey_hat + ey_hat.transpose(1, 2)
            ey_logits = self.edge_out_proj(ey_hat)
        else:
            e_logits = None
            ey_logits = None

        if remove_com:
            com_x = (x_hat * inv_mask).sum(dim=-2, keepdim=True) / n_atoms.unsqueeze(-1)
            com_y = (y_hat * inv_mask).sum(dim=-2, keepdim=True) / n_atoms.unsqueeze(-1)
            x_hat = (x_hat - com_x) * inv_mask
            y_hat = (y_hat - com_y) * inv_mask
        else:
            x_hat = x_hat * inv_mask
            y_hat = y_hat * inv_mask


        return x_hat, y_hat, atom_logits, atom_logits_y, charge_logits, charge_logits_y, e_logits, ey_logits


    def get_state_dict(self):
        return {"class_name": self.__class__.__name__,
                "class_module": self.__class__.__module__,
                "params": self.params,
                "state": self.state_dict()}


    @classmethod
    def load_from_file(cls, file, device, instance=None):
        save_dict = torch.load(file, map_location="cpu", weights_only=False)
        model = cls.load_from_state_dict(save_dict, device, instance=instance)
        return model
