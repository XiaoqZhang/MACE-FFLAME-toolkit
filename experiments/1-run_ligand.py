from fflame.cell import LigandBuilder, LigandScissor
from fflame.simulation.md import run_md_nvt
from fflame.simulation.mol import GeometryOptimization
from fflame.sample.selection import selector
from ase.io import read, write
from xtb.ase.calculator import XTB
from collections import Counter
import os
import numpy as np
from glob import glob

if __name__ == "__main__":

    # Extract ligand from CIF file
    incif = "data/example.cif"
    cifname = incif.split('/')[-1]
    ligand_folder = "data/ligand"
    os.makedirs(ligand_folder, exist_ok=True)
    fm = read(incif)
    lgs = LigandScissor(fm)
   
    print(f"🚀 [INFO] {lgs.num_linkers()} linkers found in the structure.")

    for i in range(2):
        extracted = lgs.extract_linkers(i, 4, "ase")
        fname_extracted = os.path.join(ligand_folder, cifname.replace('.cif', f'_lk_{i}.xyz'))
        write(fname_extracted, extracted)

        # Add hydrogens
        flag, optimized, addh = lgs.optimize_linker(extracted, True, 'vesta', scaling=1.04)
        fname_addh = os.path.join(ligand_folder, cifname.replace('.cif', f'_lk_{i}_addh.xyz'))
        write(fname_addh, addh)
        # Optimize ligand
        fname_good = os.path.join(ligand_folder, cifname.replace('.cif', f'_lk_{i}_opt.xyz'))
        write(fname_good, optimized)
    

        # Run MD simulation
        run_md = True
        if run_md:
            md_files = []
            ligand_md_folder = "data/ligand/md"
            os.makedirs(ligand_md_folder, exist_ok=True)
            for k in [100, 300, 500, 600]:
                print(f" 🚀 [LOG] Running MD simulation at {k}K")
                init_conf = read(fname_good)
                xtb_calc = XTB(method = 'GFN1-xTB')
                outf = f"{ligand_md_folder}/lk{i}_{k}K.xyz"
                md_files.append(outf)
                run_md_nvt(
                    init_conf, 
                    calculator=xtb_calc, 
                    temperature=k, 
                    fout=outf, 
                    nvt_time_fs=100, 
                    write_interval=10
                )

            md_frames = []
            for f in md_files:
                traj = read(f, index=':')[1:]
                md_frames += traj
            # Run geometry optimization
            num_geo_opt = 10
            frames_to_go, idx_go = selector(md_frames, number=num_geo_opt, sort=False, species=optimized.get_chemical_symbols(), seed=6666)
            print(f" 🚀 [LOG]  {num_geo_opt} frames selected for geometry optimization. ")
            flgs = []
            opt_files = []
            ligand_opt_folder = "data/ligand/opt"
            os.makedirs(ligand_opt_folder, exist_ok=True)
            for idx, frame in zip(idx_go, frames_to_go):
                outf = f"{ligand_opt_folder}/lk{i}_{idx}_opt.xyz"
                logf = f"{ligand_opt_folder}/lk{i}_{idx}_opt.log"
                flg = GeometryOptimization(frame, outf, logf, 5, 0.01, 200)
                flgs.append(flg)
                opt_files.append(outf)
            print(f"Geometry optimization finished! {Counter(flgs)}")