__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
#import gc #garbage collector (us gc.collect() after del to prevent memory leak)
from ..NSfracStep import *
from ..NSfracStep import __all__


def setup(u_components,backflow_flag,nonNewtonian_flag, beta_backflow,Resistance_flag,n_normal,facet_domains, u, v, p, q, bcs, les_model, nu, nut_,
          scalar_components, V, Q, x_, p_, u_,u_1, A_cache,
          velocity_update_solver, assemble_matrix, homogenize,
          GradFunction, DivFunction, LESsource,mesh, **NS_namespace):
    """Preassemble mass and diffusion matrices.

    Set up and prepare all equations to be solved. Called once, before
    going into time loop.

    """
    # Mass matrix
    M = assemble_matrix(inner(u, v) * dx)
    


    # Stiffness matrix (without viscosity coefficient)

    #if(backflow_flag):
    #    ds = Measure("ds")[facet_domains]
    #    K = lhs( inner(grad(u), grad(v)) * dx - 2.0/nu*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   )
    #else:
    #    K = assemble_matrix(inner(grad(u), grad(v)) * dx)

    # Allocate stiffness matrix for LES that changes with time
    KT = None if les_model is "NoModel" else (
        Matrix(M), inner(grad(u), grad(v)))

    # Pressure Laplacian.
    if (Resistance_flag):
     Ap2 = inner(grad(q), grad(p)) * dx
     Ap = assemble_matrix(inner(grad(q), grad(p)) * dx, bcs['p'])#Ap=[]
    else:
     Ap = assemble_matrix(inner(grad(q), grad(p)) * dx, bcs['p'])
     Ap2 = []

    # if les_model is "NoModel":
    # if not Ap.id() == K.id():
    # Compress matrix (creates new matrix)
    #Bp = Matrix()
    # Ap.compressed(Bp)
    #Ap = Bp
    # Replace cached matrix with compressed version
    #A_cache[(inner(grad(q), grad(p))*dx, tuple(bcs['p']))] = Ap

    # Allocate coefficient matrix (needs reassembling)
    A = Matrix(M)

    # Allocate Function for holding and computing the velocity divergence on Q
    divu = DivFunction(u_, Q, name='divu',
                       method=velocity_update_solver)

    # Allocate a dictionary of Functions for holding and computing pressure gradients
    gradp = {ui: GradFunction(p_, V, i=i, name='dpd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]),
                              method=velocity_update_solver)
             for i, ui in enumerate(u_components)}

    # Create dictionary to be returned into global NS namespace

    d = dict(A=A, M=M, Ap=Ap,Ap2=Ap2, divu=divu, gradp=gradp) #  d = dict(A=A, M=M, K=K, Ap=Ap, divu=divu, gradp=gradp)

    # Allocate coefficient matrix and work vectors for scalars. Matrix differs
    # from velocity in boundary conditions only
    if len(scalar_components) > 0:
        d.update(Ta=Matrix(M))
        if len(scalar_components) > 1:
            # For more than one scalar we use the same linear algebra solver for all.
            # For this to work we need some additional tensors. The extra matrix
            # is required since different scalars may have different boundary conditions
            Tb = Matrix(M)
            bb = Vector(x_[scalar_components[0]])
            bx = Vector(x_[scalar_components[0]])
            d.update(Tb=Tb, bb=bb, bx=bx)

    # Setup for solving convection
    u_ab = as_vector([Function(V) for i in range(len(u_components))])
    u_ab_prev = as_vector([Function(V) for i in range(len(u_components))]) #used for nonNewtonian Gamma calclulation
    a_conv = inner(v, dot(u_ab, nabla_grad(u))) * dx
    a_scalar = a_conv
    LT = None if les_model is "NoModel" else LESsource(
        nut_, u_ab, V, name='LTd')

    if bcs['p'] == []:
        attach_pressure_nullspace(Ap, x_, Q)

    mu1 = interpolate(Constant(1.0/1.06),Q)
    num1 = interpolate(Constant(0.0345),Q)
    num2 =  interpolate(Constant(0.56),Q)
    num11 = interpolate(Constant(1.0),Q)
    num3 = interpolate(Constant(1.902),Q)
    pow1 = interpolate(Constant(1.25),Q)
    pow2 = interpolate(Constant((0.22-1.)/1.25),Q)
    num0 = interpolate(Constant(0.5),Q)
    num22 = interpolate(Constant(2.0),Q)
    beta_backflow_fn = interpolate(Constant(beta_backflow),Q)
    #one_space = interpolate(Constant(1.0),Q)
    nu_const = interpolate(Constant(nu),Q)
    if(nonNewtonian_flag and backflow_flag):
        #visc_gamma = pow(0.5*inner(grad(u_ab_prev)+transpose(grad(u_ab_prev)), grad(u_ab_prev)+transpose(grad(u_ab_prev))), 0.5)
        visc_gamma = pow(num0*inner(grad(u_ab_prev)+transpose(grad(u_ab_prev)), grad(u_ab_prev)+transpose(grad(u_ab_prev))), num0) # Maybe Faster than above?
        #visc_gamma = pow(0.5*Strain(u_ab_prev), 0.5)
        #Carreau-Yasuda model. Cho and Kensey, 1991
        #visc_nu = (1./1.06) * (0.0345 + (0.56 - 0.0345) * (1. + (1.902*visc_gamma)**1.25 )**( (0.22-1.)/1.25 )  )
        visc_nu = mu1 * ( num1 + ( num2 -  num1) * ( num11 + ( num3*visc_gamma)**pow1 )**( pow2 ) ) # Maybe Faster than above?
        #K_weak = visc_nu * inner(grad(u), grad(v)) * dx #need to assemble again due to viscossity change
        ds = Measure("ds")[facet_domains]
        #K = lhs( visc_nu*inner(grad(u), grad(v)) * dx - 2.0*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   )
        K = lhs( visc_nu*inner(grad(u), grad(v)) * dx - num22*beta_backflow_fn *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/num22 * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   ) # Maybe Faster than above?
    elif(nonNewtonian_flag and not backflow_flag):
         visc_gamma = pow(0.5*inner(grad(u_ab_prev)+transpose(grad(u_ab_prev)), grad(u_ab_prev)+transpose(grad(u_ab_prev))), 0.5)
         #visc_gamma = pow(0.5*Strain(u_ab_prev), 0.5)
         #Carreau-Yasuda model
         visc_nu = (1./1.06) * (0.0345 + (0.56 - 0.0345) * (1. + (1.902*visc_gamma)**1.25 )**( (0.22-1.)/1.25 )  )
         K = visc_nu * inner(grad(u), grad(v)) * dx
    elif(backflow_flag and not nonNewtonian_flag):
         #K_weak = inner(grad(u), grad(v)) * dx - 2.0/nu*beta_backflow *  inner(v, ( dot(u_1,n_normal) - abs( dot(u_1,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains)) #beta_backflow = 2 and 10 works (at least till t=0.44)
         #K_weak_lhs = lhs(K_weak)
         #K_back = assemble_matrix(K_weak_lhs)
         #A.axpy(-0.5*nu, K_back, True)
         #del K_back
         #gc.collect() #still memory leak!
         #del K_weak_lhs
         #gc.collect()
         #del K_weak
         #gc.collect()
         ds = Measure("ds")[facet_domains]
         #K = lhs( nu*inner(grad(u), grad(v)) * dx - 2.0*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   )
         K = lhs( nu_const*inner(grad(u), grad(v)) * dx - num22*beta_backflow_fn *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/num22 * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   )  # Maybe Faster than above?
         visc_nu = nu
    else:
         K = assemble_matrix(inner(grad(u), grad(v)) * dx)
         visc_nu = nu

    #test visc_gamma output (also pass K_test and K_test_rhs in update below)
    #K_test = q*gamma_soln*dx
    #K_test_rhs = q*visc_gamma*dx
    #d.update(u_ab=u_ab,u_ab_prev=u_ab_prev, a_conv=a_conv, d=a_scalar, v=LT, KT=KT,K=K,visc_nu = visc_nu)
    d.update(u_ab=u_ab,u_ab_prev=u_ab_prev, a_conv=a_conv, a_scalar=a_scalar, LT=LT, KT=KT,K=K,visc_nu = visc_nu)
    return d

def get_solvers(use_krylov_solvers, krylov_solvers,krylov_solvers_P, bcs,
                x_, Q, scalar_components, velocity_krylov_solver,
                pressure_krylov_solver, scalar_krylov_solver, **NS_namespace):
    """Return linear solvers.

    We are solving for
       - tentative velocity
       - pressure correction

       and possibly:
       - scalars

    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        u_prec = PETScPreconditioner(
            velocity_krylov_solver['preconditioner_type'])
        u_sol = PETScKrylovSolver(
            velocity_krylov_solver['solver_type'], u_prec)
        u_sol.prec = u_prec  # Keep from going out of scope
        # u_sol = KrylovSolver(velocity_krylov_solver['solver_type'],
        #                     velocity_krylov_solver['preconditioner_type'])
        #u_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
        u_sol.parameters.update(krylov_solvers)

        ## pressure solver ##
        #p_prec = PETScPreconditioner('hypre_amg')
        #p_prec.parameters['report'] = True
        #p_prec.parameters['hypre']['BoomerAMG']['agressive_coarsening_levels'] = 0
        #p_prec.parameters['hypre']['BoomerAMG']['strong_threshold'] = 0.5
        #PETScOptions.set('pc_hypre_boomeramg_truncfactor', 0)
        #PETScOptions.set('pc_hypre_boomeramg_agg_num_paths', 1)
        p_sol = KrylovSolver(pressure_krylov_solver['solver_type'],
                             pressure_krylov_solver['preconditioner_type'])
        #p_sol.parameters['preconditioner']['structure'] = 'same'
        #p_sol.parameters['profile'] = True

        #p_sol.parameters.update(krylov_solvers)
        p_sol.parameters.update(krylov_solvers_P)

        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_prec = PETScPreconditioner(
                scalar_krylov_solver['preconditioner_type'])
            c_sol = PETScKrylovSolver(
                scalar_krylov_solver['solver_type'], c_prec)
            c_sol.prec = c_prec
            # c_sol = KrylovSolver(scalar_krylov_solver['solver_type'],
            # scalar_krylov_solver['preconditioner_type'])
            c_sol.parameters.update(krylov_solvers)
            #c_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver()
        u_sol.parameters['same_nonzero_pattern'] = True
        ## pressure solver ##
        p_sol = LUSolver()
        p_sol.parameters['reuse_factorization'] = True
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver()
            sols.append(c_sol)
        else:
            sols.append(None)

    return sols


def assemble_first_inner_iter(nonNewtonian_flag,backflow_flag, beta_backflow,mesh,facet_domains,n_normal, A, a_conv, dt, M, scalar_components, les_model,
                              a_scalar, K, nu, nut_, u_components, LT, KT,
                              b_tmp, b0, x_1, x_2, u_ab,u_ab_prev,u_,u_1,assemble_matrix, u,v, bcs, **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.

    Assemble convection matrix, compute rhs of tentative velocity and
    reset coefficient matrix for solve.

    """
    t0 = Timer("Assemble first inner iter")
    # Update u_ab used as convecting velocity
    for i, ui in enumerate(u_components):
        u_ab[i].vector().zero()
        u_ab[i].vector().axpy(1.5, x_1[ui])
        u_ab[i].vector().axpy(-0.5, x_2[ui])

    A = assemble(a_conv, tensor=A)
    A._scale(-0.5)            # Negative convection on the rhs
    A.axpy(1. / dt, M, True)  # Add mass


    # Set up scalar matrix for rhs using the same convection as velocity
    if len(scalar_components) > 0:
        Ta = NS_namespace['Ta']
        if a_scalar is a_conv:
            Ta.zero()
            Ta.axpy(1., A, True)
                
    #ds = Measure("ds")[facet_domains]
    # Add diffusion and compute rhs for all velocity components
    if(nonNewtonian_flag):
      for i, ui in enumerate(u_components):
        u_ab_prev[i].vector().zero()
        u_ab_prev[i].vector().axpy(1.0, x_1[ui])

    if (not nonNewtonian_flag and not backflow_flag):
       A.axpy(-0.5*nu, K, True)
    else:
       K2 = assemble(K)   #assemble_matrix(K)  # assemble_matrix will not update variables?
       A.axpy(-0.5, K2, True)

    #file_visc_output = File('/Users/aa3878/data/oasis_2017_changed/AAA_results_nonNewtonian/visc/' + 'visc_mu.pvd')
    #visc_nonN = project(visc_nu, FunctionSpace(mesh, "CG",1),solver_type='cg')
    #file_visc_output << visc_nonN

    if not les_model is "NoModel":
        assemble(nut_ * KT[1] * dx, tensor=KT[0])
        A.axpy(-0.5, KT[0], True)

    for i, ui in enumerate(u_components):
        # Start with body force
        b_tmp[ui].zero()
        b_tmp[ui].axpy(1., b0[ui])
        # Add transient, convection and diffusion
        b_tmp[ui].axpy(1., A * x_1[ui])
        if not les_model is "NoModel":
            LT.assemble_rhs(i)
            b_tmp[ui].axpy(1., LT.vector())

    # Reset matrix for lhs
    A._scale(-1.)
    A.axpy(2. / dt, M, True)
    [bc.apply(A) for bc in bcs['u0']]

def attach_pressure_nullspace(Ap, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm('l2')
    Aa = as_backend_type(Ap)
    null_space = VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space


def velocity_tentative_assemble(ui, b, b_tmp, p_, gradp, **NS_namespace):
    """Add pressure gradient to rhs of tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs)

def velocity_tentative_solve(ui, A, bcs, x_, x_2, u_sol, b, udiff,
                             use_krylov_solvers, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    #if use_krylov_solvers:
        #if ui == 'u0':
            #u_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
        #else:
            #u_sol.parameters['preconditioner']['structure'] = 'same'
    [bc.apply(b[ui]) for bc in bcs[ui]]
    # x_2 only used on inner_iter 1, so use here as work vector
    x_2[ui].zero()
    x_2[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol.solve(A, x_[ui], b[ui])
    t1.stop()
    udiff[0] += norm(x_2[ui] - x_[ui])


def pressure_assemble(Resistance_flag,Res1, Res2,Q,n_normal,facet_domains,mesh,bcs,assemble_matrix, u_,b, x_, dt, Ap,Ap2, divu, **NS_namespace):
    """Assemble rhs of pressure equation."""
    #if (Resistance_flag):
    #   ds = Measure("ds")[facet_domains]
    #   Res_bc1 = Res1 * abs( assemble(dot(u_, n_normal)*ds(3, domain=mesh, subdomain_data=facet_domains) ) )
    #   Res_bc2 = Res2 * abs( assemble(dot(u_, n_normal)*ds(4, domain=mesh, subdomain_data=facet_domains) ) )
    #   print '+++++++++++++++++++++++Res_bc1', Res_bc1
    #   print '+++++++++++++++++++++++Res_bc2', Res_bc2
    #   bcs['p'] = [DirichletBC(Q,  Res_bc1 , facet_domains,3), DirichletBC(Q,  Res_bc2, facet_domains,4)]
    #   Ap = assemble_matrix(Ap2, bcs['p'])
    divu.assemble_rhs()  # Computes div(u_)*q*dx
    b['p'][:] = divu.rhs
    b['p']._scale(-1. / dt)
    b['p'].axpy(1., Ap * x_['p'])


def pressure_solve(Resistance_flag,Res1, Res2,Q,n_normal,facet_domains,mesh,assemble_matrix,u_,dp_, x_, Ap,Ap2, b, p_sol, bcs, **NS_namespace):
    """Solve pressure equation."""
    if (Resistance_flag):
       ds = Measure("ds")[facet_domains]
       flow1 = abs( assemble(dot(u_, n_normal)*ds(3, domain=mesh, subdomain_data=facet_domains) ) )
       flow2 = abs( assemble(dot(u_, n_normal)*ds(4, domain=mesh, subdomain_data=facet_domains) ) )
       Res_bc1 =  Res1 * flow1
       Res_bc2 =  Res2 * flow2
       if MPI.rank(mpi_comm_world()) == 0:
        print '+++++++++++++++++++++++Res_bc: flow1:', flow1
        print '+++++++++++++++++++++++Res_bc: flow2:', flow2
        print '+++++++++++++++++++++++Res_bc: 1/2ratio:', flow1/(flow2 + 1e-16)
       bcs['p'] = [DirichletBC(Q,  Res_bc1 , facet_domains,3), DirichletBC(Q,  Res_bc2, facet_domains,4)]
       #Ap = assemble_matrix(Ap2, bcs['p'])
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(Ap, 'null_space'):
        p_sol.null_space.orthogonalize(b['p'])


    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, x_['p'], b['p'])

    t1.stop()
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(x_['p'])

    dp_.vector().axpy(-1., x_['p'])
    dp_.vector()._scale(-1.)


def velocity_update(u_components, bcs, gradp, dp_, dt, x_, **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    for ui in u_components:
        gradp[ui](dp_)
        x_[ui].axpy(-dt, gradp[ui].vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]

def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, Schmidt, b, K, x_1, b0, les_model, **NS_namespace):
    """Assemble scalar equation."""
    # Just in case you want to use a different scalar convection
    if not a_scalar is a_conv:
        assemble(a_scalar, tensor=Ta)
        Ta._scale(-0.5)            # Negative convection on the rhs
        Ta.axpy(1. / dt, M, True)    # Add mass

    # Compute rhs for all scalars
    for ci in scalar_components:
        # Add diffusion
        Ta.axpy(-0.5 * nu / Schmidt[ci], K, True)
        if not les_model is "NoModel":
            Ta.axpy(-0.5 / Schmidt_T[ci], KT[0], True)

        # Compute rhs
        b[ci].zero()
        b[ci].axpy(1., Ta * x_1[ci])
        b[ci].axpy(1., b0[ci])

        # Subtract diffusion
        Ta.axpy(0.5 * nu / Schmidt[ci], K, True)
        if not les_model is "NoModel":
            Ta.axpy(0.5 / Schmidt_T[ci], KT[0], True)

    # Reset matrix for lhs - Note scalar matrix does not contain diffusion
    Ta._scale(-1.)
    Ta.axpy(2. / dt, M, True)


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""

    Ta.axpy(0.5 * nu / Schmidt[ci], K, True)  # Add diffusion
    if len(scalar_components) > 1:
        # Reuse solver for all scalars. This requires the same matrix and vectors to be used by c_sol.
        Tb, bb, bx = NS_namespace['Tb'], NS_namespace['bb'], NS_namespace['bx']
        Tb.zero()
        Tb.axpy(1., Ta, True)
        bb.zero()
        bb.axpy(1., b[ci])
        bx.zero()
        bx.axpy(1., x_[ci])
        [bc.apply(Tb, bb) for bc in bcs[ci]]
        c_sol.solve(Tb, bx, bb)
        x_[ci].zero()
        x_[ci].axpy(1., bx)

    else:
        [bc.apply(Ta, b[ci]) for bc in bcs[ci]]
        c_sol.solve(Ta, x_[ci], b[ci])
    Ta.axpy(-0.5 * nu / Schmidt[ci], K, True)  # Subtract diffusion
    # x_[ci][x_[ci] < 0] = 0.               # Bounded solution
    #x_[ci].set_local(maximum(0., x_[ci].array()))
    # x_[ci].apply("insert")
