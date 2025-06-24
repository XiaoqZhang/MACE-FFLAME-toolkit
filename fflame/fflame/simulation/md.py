from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
import time
from ase.md import MDLogger
from ase.md.npt import NPT

def run_md_nvt(
    init_conf,
    calculator,
    temperature,
    fout,
    nvt_time_fs,
    write_interval=10,
):
    # 1. setup
    init_conf.set_calculator(calculator)

    # 2. Initialize velocities
    MaxwellBoltzmannDistribution(init_conf, temperature_K=temperature)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    # 3. === NVT equilibration ===
    timestep = 1.0 * units.fs
    nvt_steps = int(nvt_time_fs)

    dyn_nvt = Langevin(init_conf, timestep, temperature_K=temperature, friction=0.1)
    dyn_nvt.attach(lambda: init_conf.write(f"{fout}", append=True), interval=write_interval)

    fout_log = fout.replace('.xyz', '.log')
    log_nvt = open(f"{fout_log}", 'w')
    logger_nvt = MDLogger(dyn_nvt, init_conf, log_nvt, header=True, stress=False, peratom=False)
    dyn_nvt.attach(logger_nvt, interval=write_interval)

    print(f"Running NVT for {nvt_steps} steps...")
    t0 = time.time()
    dyn_nvt.run(nvt_steps)
    t1 = time.time()
    print(f"NVT finished in {(t1 - t0) / 60:.2f} minutes.")

def run_md_npt(
    init_conf,
    calculator,
    temperature,
    output_prefix,
    nvt_time_fs,
    npt_time_fs,
    write_interval=10,
    pressure_bar=1.0
):  
    # 1. Setup 
    init_conf.set_calculator(calculator)

    # 2. Initialize velocities
    MaxwellBoltzmannDistribution(init_conf, temperature_K=temperature)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    # 3. === NVT equilibration ===
    timestep = 1.0 * units.fs
    nvt_steps = int(nvt_time_fs)

    dyn_nvt = Langevin(init_conf, timestep, temperature_K=temperature, friction=0.1)
    dyn_nvt.attach(lambda: init_conf.write(f"{output_prefix}_nvt.xyz", append=True), interval=write_interval)

    log_nvt = open(f"{output_prefix}_nvt.log", 'w')
    logger_nvt = MDLogger(dyn_nvt, init_conf, log_nvt, header=True, stress=False, peratom=False)
    dyn_nvt.attach(logger_nvt, interval=write_interval)

    print(f"Running NVT for {nvt_steps} steps...")
    t0 = time.time()
    dyn_nvt.run(nvt_steps)
    t1 = time.time()
    print(f"NVT finished in {(t1 - t0) / 60:.2f} minutes.")

    # 4. === NPT production ===
    npt_steps = int(npt_time_fs)

    dyn_npt = NPT(
        init_conf,
        timestep,
        temperature_K=temperature,
        externalstress=pressure_bar * units.bar,
        ttime=100.0 * units.fs,
        pfactor=3000.0 * units.fs**2 * units.eV
    )
    dyn_npt.attach(lambda: init_conf.write(f"{output_prefix}_npt.xyz", append=True), interval=write_interval)

    log_npt = open(f"{output_prefix}_npt.log", 'w')
    logger_npt = MDLogger(dyn_npt, init_conf, log_npt, header=True, stress=True, peratom=False)
    dyn_npt.attach(logger_npt, interval=write_interval)

    print(f"Running NPT for {npt_steps} steps...")
    t0 = time.time()
    dyn_npt.run(npt_steps)
    t1 = time.time()
    print(f"NPT finished in {(t1 - t0) / 60:.2f} minutes.")

    