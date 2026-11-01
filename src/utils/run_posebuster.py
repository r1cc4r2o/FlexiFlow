import argparse
import pandas as pd
import logging
from pathlib import Path
from posebusters import PoseBusters

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def main(gen_dir, ref_dir):
    data = []
    gen_dir = Path(gen_dir)
    ref_dir= Path(ref_dir)
    pred_path = gen_dir / "generated_ligands.sdf"
    true_path = ref_dir / "ligand.sdf"
    cond_path = ref_dir / "protein.pdb"
    out_path = gen_dir / "posebuster.csv"
    pdb_id = gen_dir.name
    if out_path.exists():
        logging.info(f'{out_path} exists, skipping {pdb_id}')
        return None
    if not all(f.exists() for f in [pred_path, true_path, cond_path]):
        logging.debug(f"Missing file(s) for {pdb_id}, skipping.")

    data.append({
        "mol_pred": pred_path,
        "mol_true": true_path,
        "mol_cond": cond_path
    })

    mol_table = pd.DataFrame(data)

    buster = PoseBusters()
    results = buster.bust_table(mol_table, full_report=False)

    results.to_csv(out_path, index=False)
    logging.info(f"PoseBuster analysis complete for {pdb_id}. Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PoseBuster on a batch of predicted molecules.")
    parser.add_argument("--gen_dir", required=True, help="Directory containing generated_molecules/pdb_id/generated_ligands.sdf")
    parser.add_argument("--ref_dir", required=True, help="Directory containing reference_structures/pdb_id/ligand.sdf and protein.pdb")
    args = parser.parse_args()

    main(args.gen_dir, args.ref_dir)
