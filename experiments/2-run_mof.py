from fflame.simulation.md import run_md_nvt, run_md_npt
from mace.calculators import MACECalculator
from fflame.sample.selection import selector
from ase.io import read, write
from collections import Counter
import os
import numpy as np
from glob import glob
import subprocess

# Download MACE-MP-0b2
url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model"
output = "data/mace-medium-density-agnesi-stress.model"
subprocess.run(["wget", url, "-O", output], check=True)


if __name__ == "__main__":

    incif = "data/example.cif"
    cifname = incif.split('/')[-1].split('.')[0]
    mof_folder = "data/mof"
    os.makedirs(mof_folder, exist_ok=True)
    ensemble = "npt"  # or "npt"

    # Set calculator
    model_path = "data/mace-medium-density-agnesi-stress.model" # replace the model path
    mace_calc = MACECalculator(model_paths=[model_path], device="cuda", cu_eq=True)

    # Run MD simulation     
    md_folder = f"{mof_folder}/md_{ensemble}"
    os.makedirs(md_folder, exist_ok=True)

    for k in [100, 300, 500, 600]:
        print(f" 🚀 [INFO] Running MD simulation at {k}K")
        init_conf = read(incif)
        outf = os.path.join(md_folder, f"{cifname}_{k}K.xyz")
        
        if ensemble == 'nvt':
            run_md_nvt(
                init_conf,
                calculator=mace_calc,
                temperature=k,
                fout=outf,
                nvt_time_fs=100,
                write_interval=10,
            )
        elif ensemble == 'npt':
            p_in_bar = 1.0
            run_md_npt(
                init_conf,
                calculator=mace_calc,
                temperature=k,
                output_prefix=os.path.join(md_folder, f"{cifname}_{k}K_{p_in_bar}bar"),
                nvt_time_fs=10,
                npt_time_fs=100,
                write_interval=10,
                pressure_bar=p_in_bar
            )
