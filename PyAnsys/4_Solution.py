#!/usr/bin/env python
# =============================================================================
# Ansys Fluent - batch transient CFD (PyFluent)
# Kernel: Python (Ansys 2024R2)
#
# For every geometry (1..40) and every inlet profile (1..5):
#   launch Fluent -> read mesh -> set up laminar transient blood flow ->
#   apply velocity + temperature profile at the inlet -> initialize ->
#   run the transient solve -> auto-export results -> save case+data -> exit
# =============================================================================

import os
import time
import tracemalloc
from pathlib import Path

import ansys.fluent.core as pyfluent

# ---- HPC / Ansys environment ------------------------------------------------
# InfiniBand + MPI settings so Fluent runs in parallel on the cluster.
os.environ["UCX_NET_DEVICES"] = "mlx5_0:1"
os.environ["UCX_TLS"]         = "dc,ud,self"
os.environ["FI_PROVIDER"]     = "verbs"
os.environ["I_MPI_DEBUG"]     = "5"
os.environ["AWP_ROOT242"]     = "/uufs/chpc.utah.edu/sys/installdir/ansys/2024R2/v242"

# ---- settings ---------------------------------------------------------------
GEO_START,     GEO_END     = 1, 40     # geometry index range
PROFILE_START, PROFILE_END = 1, 5      # inlet-profile index range

PROCESSORS           = 16              # parallel processes for Fluent
TURBULENCE           = False           # False = laminar, True = k-omega SST
TIME_STEP_SIZE       = 0.001           # seconds
NUMBER_OF_TIME_STEPS = 4000
MAX_ITER_PER_STEP    = 40
SAVE_EVERY_STEPS     = 50              # export results every N time steps

# Zone names as they appear in the mesh.
INLET_ZONE  = "blood-inlet"
OUTLET_ZONE = "blood-outlet"
FLUID_ZONE  = "solid"

# Input/output locations.
MESH_DIR     = "Mesh"   # meshes from the meshing tutorial: Mesh/output_mesh_<i>.cas
RESULTS_ROOT = "/scratch/general/nfs1/u1462615/Cases/Ansys/idealized_aneurysm_tutorial"

# Blood material properties (CGS units: g, cm, s).
BLOOD_DENSITY   = 1.04   # g/cm^3
BLOOD_VISCOSITY = 0.04   # g/(cm s)


# ---- one simulation ---------------------------------------------------------
def run_one(i, profile_num):
    """Run a single transient simulation for geometry i with profile profile_num."""
    mesh_file    = f"{MESH_DIR}/output_mesh_{i}.cas"
    profile_file = f"profile_{profile_num}.prof"   # the file on disk
    profile_name = f"profile{profile_num}"         # the profile name used in the BC
    run_dir      = f"{RESULTS_ROOT}/geo_{i}_profile_{profile_num}"
    results_pref = f"{run_dir}/Results"            # EnSight export name prefix

    # Check inputs exist before spending a Fluent license launching the solver.
    if not Path(mesh_file).is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    if not Path(profile_file).is_file():
        raise FileNotFoundError(f"Profile file not found: {profile_file}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=== Geometry {i}, profile {profile_num} ===")
    start_time = time.time()

    # 1) Launch the Fluent solver (double precision, parallel, transient).
    session = pyfluent.launch_fluent(
        precision="double",
        processor_count=PROCESSORS,
        mode="solver",
        start_timeout=600,
    )

    try:
        # 2) General settings: pressure-based, transient.
        session.setup.general = {
            "solver": {
                "type": "pressure-based",
                "velocity_formulation": "absolute",
                "time": "transient",
            }
        }

        # 3) Read the mesh and check it.
        session.file.read_case(file_name=mesh_file)
        session.mesh.check()

        # Second-order bounded time advancement.
        session.settings.setup.general.solver.time = "unsteady-2nd-order-bounded"

        # 4) Viscous model: laminar blood flow (or k-omega SST if TURBULENCE).
        if TURBULENCE:
            session.setup.models.viscous.model = "k-omega"
            session.setup.models.viscous.k_omega_model = "sst"
        else:
            session.execute_tui("define models viscous laminar yes")

        # 5) Scale the mesh by 0.001 in each direction, then re-check.
        session.execute_tui("mesh scale 0.001 0.001 0.001")
        session.execute_tui("mesh check")

        # 6) Enable the energy equation (we transport temperature).
        session.setup.models.energy.enabled = True

        # 7) Read the transient inlet profile table (holds "vel" and "temp").
        session.execute_tui(f"file read-transient-table {profile_file}")

        # 8) Set inlet / outlet boundary types.
        session.setup.boundary_conditions.set_zone_type(
            zone_list=[INLET_ZONE], new_type="velocity-inlet"
        )
        session.setup.boundary_conditions.set_zone_type(
            zone_list=[OUTLET_ZONE], new_type="pressure-outlet"
        )

        # 9) Apply the velocity + temperature profile to the inlet.
        #    "vel" / "temp" are the field names defined inside the .prof file.
        session.execute_tui(
            f"define boundary-conditions velocity-inlet {INLET_ZONE} "
            f"no no yes yes yes no {profile_name} vel no 0 yes no {profile_name} temp"
        )

        # 10) Switch to CGS units and define the blood material.
        session.execute_tui("define set-unit-system cgs")
        session.setup.materials.fluid["blood"] = {
            "density":   {"option": "constant", "value": BLOOD_DENSITY},
            "viscosity": {"option": "constant", "value": BLOOD_VISCOSITY},
        }
        session.setup.cell_zone_conditions.fluid[FLUID_ZONE].general.material = "blood"

        # 11) Initialize the flow field (reference temperature 0).
        session.execute_tui("solve initialize set-defaults temperature 0")
        session.execute_tui("solve initialize initialize-flow")

        # 12) Set the transient time-step size.
        session.execute_tui(
            f"solve set transient-controls time-step-size {TIME_STEP_SIZE}"
        )

        # 13) Auto-export results to EnSight Gold every SAVE_EVERY_STEPS steps.
        #     Configure this BEFORE iterating so every interval is captured.
        session.execute_tui(
            f"file transient-export ensight-gold-transient {results_pref} () * () "
            f"wall-shear temperature velocity-magnitude x-velocity y-velocity "
            f"z-velocity total-temperature () no yes export time-step "
            f"{SAVE_EVERY_STEPS} yes"
        )

        # 14) Run the transient simulation (single, complete run).
        print(f"Running {NUMBER_OF_TIME_STEPS} time steps...")
        session.execute_tui(
            f"solve dual-time-iterate {NUMBER_OF_TIME_STEPS} {MAX_ITER_PER_STEP}"
        )

        # 15) Save the final case + data.
        case_path = f"{run_dir}/simulation_results.cas.h5"
        session.file.write(file_type="case-data", file_name=case_path)
        print(f"Saved {case_path}")

    finally:
        session.exit()      # always release the Fluent license

    print(f"Done in {time.time() - start_time:.1f} s")


# ---- batch loop -------------------------------------------------------------
tracemalloc.start()
total_runs = (GEO_END - GEO_START + 1) * (PROFILE_END - PROFILE_START + 1)
done = failed = 0

for i in range(GEO_START, GEO_END + 1):
    for profile_num in range(PROFILE_START, PROFILE_END + 1):
        try:
            run_one(i, profile_num)
            done += 1
        except Exception as error:
            failed += 1
            print(f"FAILED geo {i}, profile {profile_num}: "
                  f"{type(error).__name__}: {error}")
        print(f"Progress: {done + failed}/{total_runs} "
              f"(done={done}, failed={failed})")

current, peak = tracemalloc.get_traced_memory()
print(f"\nAll runs finished. done={done}, failed={failed}, "
      f"peak memory={peak / 1e6:.1f} MB")
tracemalloc.stop()