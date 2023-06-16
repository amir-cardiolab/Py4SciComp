__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-07"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
"""This is an optimized implementation of a backwards
differencing solver with pressure correction in rotational form.

"""
from dolfin import *
from .IPCS_ABCN import *  # reuse code from IPCS_ABCN
from .IPCS_ABCN import __all__, attach_pressure_nullspace


def setup(u_components, u, v, p, q, nu, nut_, LESsource,
          bcs, scalar_components, V, Q, x_, u_, p_, q_1, q_2,
          velocity_update_solver, assemble_matrix, les_model,
          DivFunction, GradFunction, homogenize, **NS_namespace):
    """Set up all equations to be solved."""

    # Mass matrix
    M = assemble_matrix(inner(u, v) * dx)

    # Stiffness matrix (without viscosity coefficient)
    K = assemble_matrix(inner(grad(u), grad(v)) * dx)

    # Allocate stiffness matrix for LES that changes with time
    KT = None if les_model is "NoModel" else (
        Matrix(M), inner(grad(u), grad(v)))

    # Pressure Laplacian. Either reuse K or assemble new
    Ap = assemble_matrix(inner(grad(q), grad(p)) * dx, bcs['p'])

    if les_model is "NoModel":
        if not Ap.id() == K.id():
            Bp = Matrix()
            Ap.compressed(Bp)
            Ap = Bp

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

    # Check first if we are starting from two equal velocities (u_1=u_2)
    initial_u1_norm = sum([q_1[ui].vector().norm('l2') for ui in u_components])
    initial_u2_norm = sum([q_2[ui].vector().norm('l2') for ui in u_components])

    # In that case use Euler on first iteration
    beta = Constant(2.0) if abs(initial_u1_norm -
                                initial_u2_norm) > DOLFIN_EPS_LARGE else Constant(3.0)

    # Create dictionary to be returned into global NS namespace
    d = dict(A=A, M=M, K=K, Ap=Ap, divu=divu, gradp=gradp, beta=beta)

    if bcs['p'] == []:
        attach_pressure_nullspace(Ap, x_, Q)

    # Allocate coefficient matrix and work vectors for scalars. Matrix differs from velocity in boundary conditions only
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
    u_convecting = as_vector([Function(V) for i in range(len(u_components))])
    a_conv = inner(v, dot(u_convecting, nabla_grad(u))) * dx  # Faster version
    a_scalar = inner(v, dot(u_, nabla_grad(u))) * dx
    LT = None if les_model is "NoModel" else LESsource(
        (nu + nut_), u_convecting, V, name='LTd')
    d.update(u_convecting=u_convecting, a_conv=a_conv,
             a_scalar=a_scalar, LT=LT, KT=KT)
    return d


def assemble_first_inner_iter(A, a_conv, dt, M, scalar_components, KT, LT,
                              a_scalar, K, nu, u_components, les_model, nut_,
                              b_tmp, b0, x_1, x_2, u_convecting,
                              bcs, beta, **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.

    Assemble convection matrix, compute rhs of tentative velocity and
    reset coefficient matrix for solve.

    """
    t0 = Timer("Assemble first inner iter")
    # Update u_convecting used as convecting velocity
    for i, ui in enumerate(u_components):
        u_convecting[i].vector().zero()
        u_convecting[i].vector().axpy(2.0, x_1[ui])
        u_convecting[i].vector().axpy(-1.0, x_2[ui])

    assemble(a_conv, tensor=A)

    # Set up scalar matrix
    if len(scalar_components) > 0:
        Ta = NS_namespace['Ta']
        if a_scalar is a_conv:
            Ta.zero()
            Ta.axpy(1., A, True)
        else:
            assemble(a_scalar, tensor=Ta)

    # Compute rhs for all velocity components
    for ui in u_components:
        b_tmp[ui].zero()              # start with body force
        b_tmp[ui].axpy(1., b0[ui])
        b_tmp[ui].axpy(4.0 / (beta(0) * dt), M * x_1[ui])
        b_tmp[ui].axpy(-1.0 / (beta(0) * dt), M * x_2[ui])
        if not les_model is "NoModel":
            LT.assemble_rhs(i)
            b_tmp[ui].axpy(1., LT.vector())

    A.axpy(nu, K, True)
    if not les_model is "NoModel":
        assemble(nut_ * KT[1] * dx, tensor=KT[0])
        A.axpy(1., KT[0], True)

    A.axpy(3.0 / beta(0) / dt, M, True)
    [bc.apply(A) for bc in bcs['u0']]


def velocity_tentative_assemble(ui, b, b_tmp, x_, gradp, p_, **NS_namespace):
    """Add pressure gradient to rhs of tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs)


def pressure_assemble(b, dt, divu, beta, Ap, x_, nu, u_, q, **NS_namespace):
    """Assemble rhs of pressure equation."""
    divu()  # Both computes div(u_) and the rhs div(u_)*q*dx
    b['p'][:] = divu.rhs
    b['p']._scale(-3.0 / beta(0) / dt)
    b['p'].axpy(1., Ap * x_['p'])
    # There's a small difference here from BDFPC in the assembling of divu
    b['p'].axpy(-nu, Ap * divu.vector())  # This is fast
    # b['p'].axpy(-nu, assemble(inner(grad(div(u_)), grad(q))*dx)) # This is exact


def pressure_solve(dp_, x_, Ap, b, p_sol, bcs, nu, divu, Q, beta, **NS_namespace):
    """Solve pressure equation."""
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])

    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(Ap, 'null_space'):
        Ap.null_space.orthogonalize(b['p'])

    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, x_['p'], b['p'])
    t1.stop()
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(x_['p'])

    dp_.vector()._scale(-1)
    dp_.vector().axpy(1.0, x_['p'])
    dp_.vector().axpy(nu, divu.vector())
    dp_.vector()._scale(beta(0) / 3.0)  # To reuse code from IPCS_ABCN


def velocity_update(u_components, bcs, dp_, dt, x_, gradp, beta, **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    for ui in u_components:
        gradp[ui](dp_)     # Computes gradient of pressure correction
        x_[ui].axpy(-dt, gradp[ui].vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]
    beta.assign(2.0)

# def scalar_assemble(a_scalar, a_conv, Ta , dt, M, scalar_components,
    # nu, Schmidt, b, K, x_1, b0, **NS_namespace):
    #"""Assemble scalar equation."""
    # Just in case you want to use a different scalar convection
    # if not a_scalar is a_conv:
    #Ta = assemble(a_scalar, tensor=Ta)
    # Ta._scale(-1.)            # Negative convection on the rhs
    # Ta.axpy(1./dt, M, True)   # Add mass

    # Compute rhs for all scalars
    # for ci in scalar_components:
    # Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Add diffusion
    # b[ci].zero()                          # Compute rhs
    #b[ci].axpy(1., Ta*x_1[ci])
    #b[ci].axpy(1., b0[ci])
    # Ta.axpy(0.5*nu/Schmidt[ci], K, True)  # Subtract diffusion
    # Reset matrix for lhs - Note scalar matrix does not contain diffusion
    # Ta._scale(-1.)
    #Ta.axpy(2./dt, M, True)

# def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
    # nu, Schmidt, K, **NS_namespace):
    #"""Solve scalar equation."""

    # Ta.axpy(0.5*nu/Schmidt[ci], K, True) # Add diffusion
    # if len(scalar_components) > 1:
    # Reuse solver for all scalars. This requires the same matrix and vectors to be used by c_sol.
    #Tb, bb, bx = NS_namespace['Tb'], NS_namespace['bb'], NS_namespace['bx']
    # Tb.zero()
    #Tb.axpy(1., Ta, True)
    #bb.zero(); bb.axpy(1., b[ci])
    #bx.zero(); bx.axpy(1., x_[ci])
    #[bc.apply(Tb, bb) for bc in bcs[ci]]
    #c_sol.solve(Tb, bx, bb)
    #x_[ci].zero(); x_[ci].axpy(1., bx)

    # else:
    #[bc.apply(Ta, b[ci]) for bc in bcs[ci]]
    #c_sol.solve(Ta, x_[ci], b[ci])
    # Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Subtract diffusion
    # x_[ci][x_[ci] < 0] = 0.               # Bounded solution
    ##x_[ci].set_local(maximum(0., x_[ci].array()))
    # x_[ci].apply("insert")
