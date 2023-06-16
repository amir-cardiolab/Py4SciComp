__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#from oasis.problems import *
from problems import *

# Default parameters NSfracStep solver
NS_parameters.update(
    # Physical constants and solver parameters
    t=0.0,               # Time
    tstep=0,             # Timestep
    T=1.0,               # End time
    dt=0.01,             # Time interval on each timestep

    # Some discretization options
    # Use Adams Bashforth projection as first estimate for pressure on new timestep
    AB_projection_pressure=False,
    solver="IPCS_ABCN",  # "IPCS_ABCN", "IPCS_ABE", "IPCS", "Chorin", "BDFPC", "BDFPC_Fast"

    # Parameters used to tweek solver
    max_iter=1,                 # Number of inner pressure velocity iterations on timestep
    max_error=1e-6,             # Tolerance for inner iterations (pressure velocity iterations)
    iters_on_first_timestep=2,  # Number of iterations on first timestep
    use_krylov_solvers=True,    # Otherwise use LU-solver
    print_intermediate_info=10,
    print_velocity_pressure_convergence=False,

    # Parameters used to tweek output
    plot_interval=10,
    checkpoint=10,       # Overwrite solution in Checkpoint folder each checkpoint
    save_step=10,        # Store solution each save_step
    restart_folder=None, # If restarting solution, set the folder holding the solution to start from here
    output_timeseries_as_vector=True,  # Store velocity as vector in Timeseries
    
    save_start = 0, #starts saving results inf tstep greater than thi value
    #NonNewtonian model
    nonNewtonian_flag = False, #currently only implemented in IPCS_ABCN
    
    #ramping
    flag_ramp = False, #if True, for the first cycle only start from 0 flow rate at the inlet
    initial_time_ramp = -0.05, # should be a negative value. Same value entered at the first time entry in the BC file with a corresponding zero flow.
                     
    #Outlet backflow stabilization
    backflow_flag = False, #currently only implemented in IPCS_ABCN
    beta_backflow = 0.2,
                 
    #R BC
    Resistance_flag = False, #make sure to specify outlet faces in IPCS_ABCN pressure_solve
    Res1 = 24000., #resistance at the first outlet
    Res2 = 24000., #resistance at the second outlet
                     
    #HDF5 support
    flag_H5 = True, #Current version of fenics does not support h5 files in parallel (set to False for MPI)
               
    flag_wss = False,
                     
    #Residence time model (runs with NSfracStep_RTrheo and solver="IPCS_ABCN_RT"
    RT_thresh = 5.0,  #5.0 # higher RT will turn on nonNewtonian model #default value was 5
    RT_filename = '/scratch/aa3878/IA_oasis/PRT/IAbwdRT_RT.',
                     
    # Choose LES model and set default parameters
    # NoModel, Smagorinsky, Wale, DynamicLagrangian, ScaleDepDynamicLagrangian
    les_model='NoModel',

    # LES model parameters
    Smagorinsky=dict(Cs=0.1677),              # Standard Cs, same as OpenFOAM
    Wale=dict(Cw=0.325),
    DynamicSmagorinsky=dict(Cs_comp_step=1),  # Time step interval for Cs to be recomputed
    KineticEnergySGS=dict(Ck=0.08, Ce=1.05),

    # Parameter set when enabling test mode
    testing=False,

    # Solver parameters that will be transferred to dolfins parameters['krylov_solver']
    krylov_solvers=dict(
        monitor_convergence=False,
        report=False,
        error_on_nonconvergence=False,
        nonzero_initial_guess=True,
        maximum_iterations=200,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-8),
    #added by Amir for P:
    krylov_solvers_P=dict(
                     monitor_convergence=False,
                     report=False,
                     error_on_nonconvergence=False,
                     nonzero_initial_guess=True,
                     maximum_iterations=200,
                     relative_tolerance=1e-8,
                     absolute_tolerance=1e-8),

    # Velocity update
    velocity_update_solver=dict(
        method='default',  # "lumping", "gradient_matrix"
        solver_type='cg',
        preconditioner_type='jacobi',
        low_memory_version=False),

    velocity_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi'),
                     
     #default preconditioner was hypre_amg. change to default (Amir)
    pressure_krylov_solver=dict(
        solver_type='gmres',
        preconditioner_type='default'),

    scalar_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi'),

    nut_krylov_solver=dict(
        method='WeightedAverage',  # Or 'default'
        solver_type='cg',
        preconditioner_type='jacobi'),
)


def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass


def pressure_hook(**NS_namespace):
    """Called prior to pressure solve."""
    pass


def start_timestep_hook(**NS_namespace):
    """Called at start of new timestep"""
    pass


def temporal_hook(**NS_namespace):
    """Called at end of a timestep."""
    pass
