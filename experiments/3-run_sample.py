from fflame.sample.selection import selector
from glob import glob
from ase.io import read, write

if __name__ == "__main__":
    traj_files = glob("data/mof/md_nvt/*.xyz")
    all_frames = []
    for fi in traj_files:
        all_frames += read(fi, index=":")

    train, idx = selector(
        all_frames, number=20, sort=True, species=all_frames[0].get_chemical_symbols(), seed=6666
    )

    write("data/mof/train.xyz", train)