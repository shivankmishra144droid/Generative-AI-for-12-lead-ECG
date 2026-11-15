"""Build latent_diffusion/colab/ecg_colab_bundle.zip for training C-M on Google Colab.

Contents mirror the repository layout so the code runs unchanged:
  processed_data/   X_train.npy, y_train.npy, train_metadata.csv   (TRAIN SPLIT ONLY: no val, no test)
  synthetic_generation/  __init__.py, model.py, s4.py, utils.py   (frozen SSSD/S4 code, import only)
  latent_diffusion/    all top-level .py and latent/*.py, the instructions file (hashed by register()),
                    artifacts/{partition,train_conditions}.npz,
                    artifacts/v2/{chosen_ae.json, ae_r1/best.pt, latents_chosen_ae.pt, rpeaks_train.npz}
Run from the repository root:  py -3.14 -m latent_diffusion.colab.build_bundle
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT/'latent_diffusion'
OUT = HOME/'colab'/'ecg_colab_bundle.zip'

FILES = [ROOT/'processed_data'/f for f in ('X_train.npy', 'y_train.npy', 'train_metadata.csv')]
FILES += [ROOT/'synthetic_generation'/f for f in ('__init__.py', 'model.py', 's4.py', 'utils.py')]
FILES += sorted(HOME.glob('*.py')) + sorted((HOME/'latent').glob('*.py'))
FILES += [HOME/'STUDY9_V2_INSTRUCTIONS.md', HOME/'artifacts'/'partition.npz', HOME/'artifacts'/'train_conditions.npz']
FILES += [HOME/'artifacts'/'v2'/f for f in ('chosen_ae.json', 'ae_r1/best.pt', 'latents_chosen_ae.pt', 'rpeaks_train.npz')]


def main():
    banned = ('X_test', 'y_test', 'test_metadata', 'X_val', 'y_val', 'val_metadata')
    assert not any(b in p.name for p in FILES for b in banned), 'held-out split in bundle'
    (HOME/'colab').mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in FILES:
            z.write(p, p.relative_to(ROOT).as_posix())
            print(p.relative_to(ROOT).as_posix())
    print(f'{OUT}  {OUT.stat().st_size/2**20:.0f} MiB, {len(FILES)} files')


if __name__ == '__main__':
    main()
