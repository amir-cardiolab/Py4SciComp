# =============================================================================
# Ansys Prime - polyhedral meshing 
# Kernel: Python (Ansys 2024R2)
#
# Pipeline per geometry:
#   import CAD -> create zones from labels -> surface mesh ->
#   polyhedral volume mesh (+ prism layers) -> print stats -> write .cas
# =============================================================================

import os
import socket
import subprocess
import time
from pathlib import Path

# ---- headless rendering -----------------------------------------------------
# On a compute node there is no screen, so we render off-screen. start_xvfb is
# wrapped in try/except because it was removed in newer PyVista versions.
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import pyvista as pv
try:
    pv.start_xvfb(wait=3)
except Exception:
    pass
pv.OFF_SCREEN = True
try:
    pv.set_jupyter_backend("static")
except Exception:
    pass

# PyPrimeMesh: the Python client. lucid is the beginner-friendly meshing API.
import ansys.meshing.prime as prime
from ansys.meshing.prime import lucid
try:
    from ansys.meshing.prime.graphics import Graphics
except Exception:
    Graphics = None

# ---- settings ---------------------------------------------------------------
START, END   = 1, 40      # geometry index range to process
MIN_SIZE     = 0.1        # smallest surface element edge length
MAX_SIZE     = 0.5        # largest surface element edge length
PRISM_LAYERS = 5          # boundary-layer (prism) layers grown off the walls
MAX_SKEWNESS = 0.9        # skewness threshold used in the quality report
VISUALIZE    = True

IP, PORT = "127.0.0.1", 50055
os.environ["AWP_ROOT242"] = "/uufs/chpc.utah.edu/sys/installdir/ansys/2024R2/v242"
PRIME = os.path.join(os.environ["AWP_ROOT242"], "meshing", "Prime", "runPrime.sh")


# ---- server -----------------------------------------------------------------
def port_open():
    """Return True if something is already listening on IP:PORT."""
    try:
        with socket.create_connection((IP, PORT), timeout=2):
            return True
    except OSError:
        return False


def start_server():
    """Launch the Prime server (runPrime.sh) and wait for its port to open."""
    # A port that is already open means a leftover server from a previous run.
    if port_open():
        raise RuntimeError(
            f"Port {PORT} already in use (stale server). In a terminal run "
            f"'pkill -f Prime', then restart the kernel."
        )
    proc = subprocess.Popen(
        [PRIME, "server", "-np", "1", "--ip", IP, "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for _ in range(24):            # poll for up to ~120 s
        if port_open():
            print("Prime server is up.")
            return proc
        time.sleep(5)
    raise RuntimeError("Prime server did not start in time.")


def show(model, title):
    """Render the current model state off-screen (if Graphics is available)."""
    if VISUALIZE and Graphics is not None:
        print(f"--- {title} ---")
        Graphics(model=model)()


# ---- statistics -------------------------------------------------------------
def print_stats(model):
    """Print labels, zones and cell-type counts, plus volume-mesh quality."""
    for part in model.parts:
        # Labels come from the CAD (e.g. inlet, outlet) and define the named
        # boundaries that the solver later uses for boundary conditions.
        labels = part.get_labels()

        # print_mesh=True adds element counts to the summary, so we can see the
        # tet-vs-poly breakdown (n_poly_cells > 0 confirms a polyhedral mesh).
        summary = part.get_summary(
            prime.PartSummaryParams(model=model, print_id=False, print_mesh=True)
        )

        print(f"\nPart        : {part.name}")
        print(f"Labels      : {labels}")
        print(f"Nodes       : {summary.n_nodes}")
        print(f"Tri faces   : {summary.n_tri_faces}")
        print(f"Tet cells   : {summary.n_tet_cells}")
        print(f"Poly cells  : {summary.n_poly_cells}")
        print(f"Total cells : {summary.n_cells}")

    # Skewness is the standard cell-quality metric (0 = perfect, ~1 = degenerate).
    # We report the worst (maximum) value across the whole volume mesh.
    search = prime.VolumeSearch(model=model)
    params = prime.VolumeQualitySummaryParams(
        model=model,
        scope=prime.ScopeDefinition(model=model, part_expression="*"),
        cell_quality_measures=[prime.CellQualityMeasure.SKEWNESS],
        quality_limit=[MAX_SKEWNESS],
    )
    results = search.get_volume_quality_summary(params=params)
    print(f"Max skewness: {results.quality_results_part[0].max_quality:.4f}")


# ---- mesh one geometry ------------------------------------------------------
def mesh_one(infile, outfile):
    # One client session per geometry. Connecting to an externally started
    # server means client.exit() only disconnects; it does not kill the server.
    client = prime.Client(ip=IP, port=PORT, timeout=60)
    try:
        model = client.model
        mesher = lucid.Mesh(model=model)

        # 1) Read the CAD geometry.
        mesher.read(str(infile))
        show(model, f"Geometry: {infile.name}")

        # 2) Turn the inlet/outlet CAD labels into named face zones.
        mesher.create_zones_from_labels("inlet,outlet")

        # 3) Triangular surface mesh, sized between MIN_SIZE and MAX_SIZE.
        mesher.surface_mesh(min_size=MIN_SIZE, max_size=MAX_SIZE)

        # 4) Polyhedral volume mesh with prism boundary layers on the walls.
        #    "* !inlet !outlet" = grow prisms on every surface except the
        #    inlet and outlet caps.
        mesher.volume_mesh(
            volume_fill_type=prime.VolumeFillType.POLY,
            prism_surface_expression="* !inlet !outlet",
            prism_layers=PRISM_LAYERS,
        )
        show(model, f"Volume mesh: {infile.name}")

        # 5) Report mesh statistics.
        print_stats(model)

        # 6) Export a Fluent .cas file.
        mesher.write(str(outfile))
        print(f"Wrote {outfile}")
    finally:
        client.exit()


# ---- run --------------------------------------------------------------------
server = start_server()
try:
    for i in range(START, END + 1):
        infile = Path("Geometries") / f"geo_{i}.fmd"
        outfile = Path.cwd() / f"output_mesh_{i}.cas"
        print(f"\n=== Geometry {i}/{END}: {infile} ===")
        mesh_one(infile, outfile)
    print("\nAll geometries done.")
finally:
    server.terminate()      # stop the Prime server when the batch is done