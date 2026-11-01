import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
from pathlib import Path
from src.train.checkpoint import CheckpointManager
from src.data.interpolate import NoiseSampler
from src.data.vocabularyp import Vocabulary
from src.data.protein_molecule import ProteinLigand
from src.inference.utils import cloud2mol
from src.inference.utils import prepare_protein_input
from src.inference.utils import filter_based_on_atomtype
from src.inference.utils import _uniform_sample_step_ensembles
from src.inference.utils import to_rdkit
from src.utils.torch_forcefield import MMFF94

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from openbabel import pybel
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import numpy as np

pybel.ob.obErrorLog.StopLogging()
RDLogger.DisableLog("rdApp.*")


class DefaultParams:
    def __init__(self):
        self.n_layers = 11
        self.d_model = 384
        self.d_message = 128
        self.d_edge = 128
        self.n_coord_sets = 64
        self.n_attn_heads = 32
        self.d_message_hidden = 128
        self.coord_norm = "length"
        self.size_emb = 64
        self.max_atoms = 256
        self.atom_vocab_size = 13
        self.charge_vocab_size = 4
        self.edge_vocab_size = 5
        self.self_conditioning = False
        self.remove_com = False
        self.protein_conditioning = True
        self.max_patoms = 501
        self.patom_vocab_size = 6
        self.gramian = False

    @property
    def kwargs(self):
        return self.__dict__


DEFAULT_TRAINING_PARAMS = DefaultParams()


def set_conformer(ligand, x):
    conf = Chem.Conformer(len(x))
    for i in range(len(x)):
        p = Point3D(*x[i].tolist()) 
        conf.SetAtomPosition(i, p)
    ligand.RemoveAllConformers()
    ligand.AddConformer(conf, assignId=True)
    return ligand

def prepare_input(batch, t, vocab, device=None):
    h = torch.nn.functional.one_hot(batch["c"][...,0], num_classes=len(vocab.atom_types)+1).to(device)
    h2 = torch.nn.functional.one_hot(batch["c"][...,1], num_classes=len(vocab.charge_types)+1).to(device)
    # h = torch.cat((h1, h2), -1)
    e = torch.nn.functional.one_hot(batch["a"], num_classes=len(vocab.bond_types)).to(device)
    t = t.view(-1, 1, 1).expand(-1, h.size(1), -1).to(device)
    features = torch.cat((t, h.float()), dim=2).to(device)
    return batch["x"].to(device), batch["y"].to(device), features, h2, e, batch["mask"].to(device)


def conditional_generate(pldataset, lvocab, pvocab, save_dir, steps, batch_size, n_samples_per_pocket, device, verbose, no_hs, same_molecule):
    time_points = (1 - np.geomspace(0.01, 1.0, steps + 1))[::-1]
    step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
    noise_sampler = NoiseSampler(lvocab, scale_ot=True, scale_ot_factor=SCALE_OT_FACTOR, sample_charges=False, symmetric_adj=False)
    pbar = range(len(pldataset))
    if verbose:
        pbar = tqdm(pbar,total=len(pldataset))    
    inference_iterations = int(np.ceil(n_samples_per_pocket / batch_size))    
    all_x, all_y, all_c, all_a = {}, {}, {}, {}
    for i in pbar:
        native_x, _, _, px, pc, _ = pldataset[i]
        pdb_id = pldataset.idx_to_key[i]
        if no_hs:
            native_lp = NATIVE_L_DIR / pdb_id / 'ligand.sdf'
            mols = [mol for mol in Chem.SDMolSupplier(native_lp, removeHs=False, sanitize=False) if mol is not None]
            assert len(mols) == 1
            _n_before = mols[0].GetNumAtoms()
            native_mol = Chem.RemoveHs(mols[0])
            num_ligand_atoms = native_mol.GetNumAtoms()
            print(f'before: {_n_before}, after: {num_ligand_atoms}')
        else:
            num_ligand_atoms = len(native_x)
        if (save_dir/pdb_id/f'generated_ligands.sdf').exists():
            print(f'skipping {pdb_id}')
            continue
        print(f'generating ligands for pocket {pdb_id}:')
        if num_ligand_atoms > DEFAULT_TRAINING_PARAMS.max_atoms:
            num_ligand_atoms = DEFAULT_TRAINING_PARAMS.max_atoms -1
        px_den = len(px)
        batch_protein = {"x": (px.unsqueeze(0)).to("cpu"), "c": pc[None].to("cpu"), "a": None, "mask": torch.zeros_like(px[...,0])[None].bool().to("cpu")} 
        batch_protein_filtered = filter_based_on_atomtype(batch_protein, pvocab)
        enc_batch_protein = pvocab.encode(batch_protein_filtered)  
        px_num = enc_batch_protein['x'].sum(1)
        px_com = px_num / px_den
        enc_batch_protein['x'] = enc_batch_protein['x'] - px_com
        all_x[pdb_id] = []
        all_y[pdb_id] = []
        all_c[pdb_id] = []
        all_a[pdb_id] = []
        for j in tqdm(range(inference_iterations)):
            start = j*batch_size
            end = min((j+1)*batch_size, n_samples_per_pocket)
            bs = end - start
            times = torch.zeros(bs, device=device)
            enc_noise = noise_sampler.sample_batch([num_ligand_atoms]*bs)

            if same_molecule:
                print("Using same molecule for all samples")
                enc_noise['x'][1:] = enc_noise['x'][0][None,:].to(device)
                enc_noise['c'][1:] = enc_noise['c'][0][None,:].to(device)
                enc_noise['a'][1:] = enc_noise['a'][0][None,:].to(device)
                enc_noise['mask'] = enc_noise['mask'][0][None,:].repeat(batch_size,1).to(device)

            for k in enc_noise:
                enc_noise[k] = enc_noise[k].to(device)
            x, y, h, _, e, mask = prepare_input(enc_noise, times, lvocab, device=device)
            enc_batch_protein_expanded = {}
            for k in enc_batch_protein:
                if  enc_batch_protein[k] is None:
                    enc_batch_protein_expanded[k] = None
                elif enc_batch_protein[k].dim() == 2:
                    enc_batch_protein_expanded[k] = enc_batch_protein[k].repeat(bs,1).to(device)
                else:
                    enc_batch_protein_expanded[k] = enc_batch_protein[k].repeat(bs,1,1).to(device)
            px, ph, _, _, pmask = prepare_protein_input(enc_batch_protein_expanded, pvocab)

            for step_size in step_sizes:
                with torch.no_grad():
                    x_p, y_p, h_p, _, c_p, _, e_p, _ = model(x=x, y=y, h=h, e=e, mask=mask, px=px, ph=ph, pmask=pmask)
                h_probs = torch.softmax(h_p, -1)
                c_probs = torch.softmax(c_p, -1)
                e_probs = torch.softmax(e_p, -1)
                
                xv = (x_p - x) / (1 - times.view(-1, 1, 1))
                yv = (y_p - y) / (1 - times.view(-1, 1, 1))

                x = x + step_size * xv
                y = y + step_size * yv

                h[...,1:] = _uniform_sample_step_ensembles(h[...,1:], h_probs, times, step_size, cat_noise=1, same_samples=same_molecule)
                e = _uniform_sample_step_ensembles(e, e_probs, times, step_size, cat_noise=1, same_samples=same_molecule)

                times = times + float(step_size)
                h[...,0] = times.view(-1, 1).repeat((1,h.shape[1]))

            mol = {
                "x": x_p.cpu() + px_com,
                "y": y_p.cpu() + px_com,
                "c": torch.cat((h_probs.argmax(-1, keepdims=True), c_probs.argmax(-1, keepdims=True)), -1).cpu(),
                "a": e_probs.argmax(-1).cpu(),
                "mask": mask.cpu()
            }
            out = lvocab.decode(mol)
            all_x[pdb_id].append(out["x"])
            all_y[pdb_id].append(out["y"])
            all_c[pdb_id].append(out["c"])
            all_a[pdb_id].append(out["a"]) 
        
        mols, egs = to_rdkit(all_x[pdb_id], all_c[pdb_id], all_a[pdb_id])  
        molsy, egsy = to_rdkit(all_y[pdb_id], all_c[pdb_id], all_a[pdb_id])
        print(f'mean energy for generated ligands: (x) {np.nanmean(egs)}, (y) {np.nanmean(egsy)}')

        pdb_dir = save_dir/pdb_id
        pdb_dir.mkdir(exist_ok=True)

        save_fp = str(pdb_dir/f'generated_ligands_x.sdf')
        save_fp_y = str(pdb_dir/f'generated_ligands_y.sdf')

        writer = Chem.SDWriter(save_fp)
        writer_y = Chem.SDWriter(save_fp_y)

        counter = 0
        for i, (mol, moly) in enumerate(zip(mols, molsy)):
            if mol is None:
                continue
            try:
                mol.SetProp("_Name", str(i))
                mol.SetProp("_Set", "x")
                writer.write(mol)
                moly.SetProp("_Name", str(i))
                moly.SetProp("_Set", "y")
                writer_y.write(moly)
                counter += 1
            except Exception as err:
                print(err)
                pass

        writer.close()
        writer_y.close()

        data_test_dir = Path('data/pdbbind/testset')/pdb_id

        save_fp = str(pdb_dir/f'generated_ligands+opt_x.sdf')
        save_fp_y = str(pdb_dir/f'generated_ligands+opt_y.sdf')

        writer = Chem.SDWriter(save_fp)
        writer_y = Chem.SDWriter(save_fp_y)

        counter = 0
        for i, (mol, moly) in enumerate(zip(mols, molsy)):

            if mol is None:
                continue

            try:

                tmp_protein = Chem.MolFromPDBFile(f"{str(data_test_dir)}/protein.pdb", sanitize=False, removeHs=False)
                tmp_protein = Chem.AddHs(tmp_protein, addCoords=True)
                Chem.GetSSSR(tmp_protein)

                ff_x = MMFF94(protein=tmp_protein, device=device)
                ff_y = MMFF94(protein=tmp_protein, device=device)
                Chem.GetSSSR(mol)
                Chem.GetSSSR(moly)
                ff_x.setup(mol)
                ff_y.setup(moly)

                x = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
                x = x.requires_grad_(True)
                y = torch.tensor(moly.GetConformer().GetPositions(), dtype=torch.float32, device=device)
                y = y.requires_grad_(True)

                opt_x = torch.optim.Adam([x], lr=1e-2)
                opt_y = torch.optim.Adam([y], lr=1e-2)

                for _ in range(1000):

                    protein_ligand_energy_x, _ = ff_x.forward(x)
                    protein_ligand_energy_y, _ = ff_y.forward(y)

                    protein_ligand_energy_x.backward()
                    protein_ligand_energy_y.backward()

                    opt_x.step()
                    opt_y.step()
                    opt_x.zero_grad()
                    opt_y.zero_grad()


                mol = set_conformer(mol, x.detach().cpu().numpy())
                moly = set_conformer(moly, y.detach().cpu().numpy())

                mol.SetProp("_Name", str(i))
                mol.SetProp("_Set", "x")
                writer.write(mol)

                moly.SetProp("_Name", str(i))
                moly.SetProp("_Set", "y")
                writer_y.write(moly)
                counter += 1

            except Exception as err:
                print(err)
                pass

        writer.close()
        writer_y.close()


def arguments_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_job', type=int, default=1)
    parser.add_argument('--total_jobs', type=int, default=1)
    _args = parser.parse_args()
    return _args

if __name__ == '__main__':

    args = arguments_parser()

    save_dir = Path("")

    NATIVE_L_DIR = Path('data/pdbbind')
    SCALE_OT_FACTOR = 0.25645364536453646
    save_dir.mkdir(exist_ok=True)
    device = "cuda"
    NO_HS = False
    NO_HS = False
    same_molecule = True
    if NO_HS:
        result_dir = ''
    else:
        result_dir = '../'
    model, _, _, _, _ = CheckpointManager.restore_last_checkpoint(result_dir)
    model = model.eval().to(device)
    
    ligand_vocabulary = Vocabulary()
    protein_vocabulary = Vocabulary({"H", "C", "N", "O", "S"})
    protein_ligand_dataset = ProteinLigand(
            ["data/pdbbind/test/pdbbind_test.npy"], 
            min_qed=0,
            min_pocket_size=120,
            max_pocket_size=500, n_split=args.total_jobs,
            split_id=args.nr_job
        )
    conditional_generate(protein_ligand_dataset, ligand_vocabulary, protein_vocabulary, save_dir, steps = 100, batch_size=42, n_samples_per_pocket=420, device=device, verbose=True, no_hs = NO_HS, same_molecule=same_molecule)