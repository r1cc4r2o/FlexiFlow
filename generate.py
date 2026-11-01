import random
import warnings

from zipp import Path
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import copy
import pandas as pd


from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
import torch

import importlib

from src.data.interpolate import NoiseSampler
from src.data.vocabulary import Vocabulary
from src.train.utils import prepare_input
from src.inference.utils import cloud2mol_paired, _uniform_sample_step, _uniform_sample_step_ensembles
from src.utils.metrics import *

from src.models.flexiflow.ff import FlexiFlow

import os


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Disables optimization that may introduce randomness

def flatten(_list):
    return [item for sublist in _list for item in sublist]


def get_energy(mol_x, mol_y):
    """Calculate the energy of a molecule using MMFF94. For all the conformers."""
    n_atoms = mol_x.GetNumAtoms()
    eg, em = [], []
    for tmp_mol in [copy.deepcopy(mol_x), copy.deepcopy(mol_y)]:
        try:
            conf = tmp_mol.GetConformer()
            mmprop = AllChem.MMFFGetMoleculeProperties(tmp_mol)
            mmff94 = AllChem.MMFFGetMoleculeForceField(tmp_mol, mmprop, confId=conf.GetId())
            e = mmff94.CalcEnergy()
            mmff94.Minimize()
            eg.append(e/n_atoms)
            em.append(mmff94.CalcEnergy()/n_atoms)
        except:
            continue
    return eg, em


def argparse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate molecules using SemlaFlow")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Path to the model checkpoint")
    parser.add_argument("--epoch", type=int, default=5, help="Epoch to load the model from")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size for generation")
    parser.add_argument("--n_mol2sample", type=int, default=1, help="Number of molecules to sample")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of steps for generation")
    parser.add_argument("--same_molecule", default=False, action='store_true', help="Use the same molecule for all samples")
    parser.add_argument("--experiment", default="", help="Experiment ID for loading the model")
    parser.add_argument("--trainingset", default="qm9", help="Training set to use for generation (geom or qm9)")
    parser.add_argument("--verbose", default=True, help="Verbose output")
    parser.add_argument("--seed", type=int, default=32, help="Random seed for reproducibility")
    parser.add_argument("--model_size", type=int, default=2, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":

    args = argparse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _df = []

    seed = args.seed
    epoch = args.epoch
    nsteps = args.nsteps
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint
    n_mol2sample = args.n_mol2sample
    same_molecule = args.same_molecule
    _training_data = args.trainingset.lower()

    print(f"Seed: {seed}, Epoch: {epoch}, N steps: {nsteps}, Batch size: {batch_size}, N mols to sample: {n_mol2sample}, Same molecule: {same_molecule}")

    verbose = args.verbose
    scale_ot_factor = 0.25645364536453646

    vocab = Vocabulary() if _training_data == "geom" else Vocabulary(atom_types={"H", "C", "N", "O", "F"})

    set_seed(seed)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = FlexiFlow(**state_dict["params"])
    model.load_state_dict(state_dict["state"])
    
    model = model.eval()
    model = model.to(device)
    print("Model loaded")

    for steps in [nsteps]:

        sdf_filename = f"results/generatedmols_epoch={epoch}_nsteps={nsteps}_nmol2sample={n_mol2sample}_batchsize={batch_size}_samemol={same_molecule}_seed={seed}.v2.sdf"
        sdf_writer = Chem.SDWriter(sdf_filename)
        noise_sampler = NoiseSampler(vocab ,scale_ot=True, scale_ot_factor=scale_ot_factor, sample_charges=False, symmetric_adj=False)

        if _training_data == "geom":
            np_atom_dist = np.load("cache/geom_atom_num_distribution.npy")
            mols_to_sample = np.random.multinomial(n_mol2sample, pvals=np_atom_dist)
            atoms = np.where(mols_to_sample > 0)[0]
            examples = mols_to_sample[atoms]
            atoms = flatten([[int(_a)] * _n for _a, _n in zip(atoms, examples)])
            examples = flatten([[int(_n)] * _n for _n in examples])
            
        elif _training_data == "qm9":
            np_atom_dist = np.load("cache/qm9_natoms.npy")
            unique, counts = np.unique(np_atom_dist, return_counts=True)
            mols_to_sample = np.random.multinomial(n_mol2sample, pvals=counts / counts.sum())
            atoms = flatten([[int(_u)] * _n for _u, _n in zip(unique, mols_to_sample) if _n > 0])
            examples = flatten([[int(_n)] * _n for _n in mols_to_sample])
        else:
            raise ValueError("Unknown experiment type. Use 'geom' or 'qm9'.")

        pbar = zip(atoms, examples)
        if verbose:
            pbar = tqdm(pbar,total=len(atoms))       

        all_x, all_y, all_c, all_a = {}, {}, {}, {}
        for num_atoms, num_examples in pbar:
            
            num_atoms = int(num_atoms)

            if num_atoms not in all_x:
                all_x[num_atoms] = []
                all_y[num_atoms] = []
                all_c[num_atoms] = []
                all_a[num_atoms] = []

            inference_iterations = 1
            time_points = (1 - np.geomspace(0.01, 1.0, steps + 1))[::-1]
            step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

            for ii in tqdm(range(inference_iterations)):
                bs = batch_size
                times = torch.tensor([0.0] * bs).to(device).to(torch.float32)

                enc_noise = noise_sampler.sample_batch([num_atoms]*bs)

                if same_molecule:
                    enc_noise['x'][1:] = enc_noise['x'][0][None,:].to(device)
                    enc_noise['c'][1:] = enc_noise['c'][0][None,:].to(device)
                    enc_noise['a'][1:] = enc_noise['a'][0][None,:].to(device)
                    enc_noise['mask'] = enc_noise['mask'][0][None,:].repeat(batch_size,1).to(device)

                x, h, _, e, mask, y = prepare_input(enc_noise, times, vocab)

                x = x.to(device)
                h = h.to(device)
                e = e.to(device)
                y = y.to(device)
                mask = mask.to(device)
                
                for step_size in step_sizes:
                    with torch.no_grad():
                        x_p, y_p, h_p, _, c_p, _, e_p, _ = model(x, y, h, e, mask)

                    h_probs = torch.softmax(h_p, -1)
                    c_probs = torch.softmax(c_p, -1)
                    e_probs = torch.softmax(e_p, -1)
                    
                    given_x = x_p.detach()
                    given_h = h_probs.detach()
                    given_e = e_probs.detach()
            
                    xv = (x_p - x) / (1 - times.view(-1, 1, 1))
                    x = x + step_size * xv

                    yv = (y_p - y) / (1 - times.view(-1, 1, 1))
                    y = y + step_size * yv 

                    if same_molecule:
                        h[...,1:] = _uniform_sample_step_ensembles(h[...,1:], h_probs, times, step_size, same_samples=same_molecule)
                        e = _uniform_sample_step_ensembles(e, e_probs, times, step_size, same_samples=same_molecule)
                    else:
                        h[...,1:] = _uniform_sample_step(h[...,1:], h_probs, times, step_size)
                        e = _uniform_sample_step(e, e_probs, times, step_size)
            
                    times = times + float(step_size)
                    h[...,0] = times.view(-1, 1).repeat((1,h.shape[1]))
                
                mol = {
                    "x": x_p.cpu(),
                    "y": y_p.cpu(),
                    "c": torch.cat((h_probs.argmax(-1, keepdims=True), c_probs.argmax(-1, keepdims=True)), -1).cpu(),
                    "a": e_probs.argmax(-1).cpu(),
                    "mask": mask.cpu()
                }
                out = vocab.decode(mol)
                all_x[num_atoms].append(out["x"])
                all_y[num_atoms].append(out["y"])
                all_c[num_atoms].append(out["c"])
                all_a[num_atoms].append(out["a"])


        atm_stab = AtomStability()
        mol_stab = MoleculeStability()
        validity = Validity()
        uniqueness = Uniqueness()
        novelty = Novelty(dataset="qm9" if _training_data == "qm9" else "geomdrugs")
        energy_validity = EnergyValidity()
        avg_strain_energy_x, avg_strain_energy_y = AverageStrainEnergy(), AverageStrainEnergy()

        mols = []
        eg, egm = [], []

        counter = 0
        molecule_counter = 0
        for k in all_x:
            X = np.vstack(all_x[k])
            Y = np.vstack(all_y[k])
            C = np.vstack(all_c[k])
            A = np.vstack(all_a[k])
            for _idx, (x, y, c, a) in enumerate(zip(X, Y, C, A)):

                counter += 1
                mol_x, mol_y = cloud2mol_paired(x, y, c, a)

                validity.update([mol_x])
                uniqueness.update([mol_x])

                try:
                    mol_x.UpdatePropertyCache()
                    Chem.GetSSSR(mol_x)
                    mol_y.UpdatePropertyCache()
                    Chem.GetSSSR(mol_y)
                    
                    if mol_x is not None:
                        mol_x = Chem.Mol(mol_x)
                        Chem.SanitizeMol(mol_x)
                    if mol_y is not None:
                        mol_y = Chem.Mol(mol_y)
                        Chem.SanitizeMol(mol_y)
                except:
                    mol_x = None
                    mol_y = None

                
                if mol_x is not None and mol_y is not None:
                    mols.append((mol_x, mol_y))
                    tmp_atm_stab = calc_atom_stabilities(mol_x)
                    atm_stab.update([tmp_atm_stab])
                    mol_stab.update([tmp_atm_stab])
                    novelty.update([mol_x])

                    mol_x.SetProp("molecule_id", f"mol_x_{molecule_counter}")
                    mol_x.SetProp("molecule_type", "x")
                    mol_x.SetProp("steps", str(steps))
                    mol_x.SetProp("epoch", str(epoch))
                    mol_x.SetProp("counter", str(counter))

                    mol_y.SetProp("molecule_id", f"mol_y_{molecule_counter}")
                    mol_y.SetProp("molecule_type", "y")
                    mol_y.SetProp("steps", str(steps))
                    mol_y.SetProp("epoch", str(epoch))
                    mol_y.SetProp("counter", str(counter))

                    molecule_counter += 1
                else:
                    continue

                e, em = get_energy(mol_x, mol_y)
                if len(e) > 0:

                    eg.append(e)
                    egm.append(em)

                    mol_x.SetProp("energy", str(e[0]) if len(e) > 0 else "")
                    mol_x.SetProp("min_energy", str(em[0]) if len(em) > 0 else "")
                    sdf_writer.write(mol_x)

                    mol_y.SetProp("energy", str(e[1]) if len(e) > 0 else "")
                    mol_y.SetProp("min_energy", str(em[1]) if len(em) > 0 else "")

                    sdf_writer.write(mol_y)
                else:
                    if mol_x is not None:
                        sdf_writer.write(mol_x)
                        sdf_writer.write(mol_y)
                    continue

                
        sdf_writer.close()
        print(f"Saved {molecule_counter} molecules to {sdf_filename}")