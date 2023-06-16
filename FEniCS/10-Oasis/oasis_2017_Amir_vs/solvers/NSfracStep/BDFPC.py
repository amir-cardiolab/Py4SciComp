__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-07"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
"""This is a simplest possible naive implementation of a backwards
differencing solver with pressure correction in rotational form.

The idea is that this solver can be quickly modified and tested for
alternative implementations. In the end it can be used to validate
the implementations of the more complex optimized solvers.

"""
from dolfin import *
from ..NSfracStep import *
from ..NSfracStep import __all__


def setup(u, q_, q_1, uc_comp, u_components, dt, v, U_AB, u_1, u_2, q_2,
          nu, p_, dp_, mesh, f, fs, q, p, u_, Schmidt, V, bcs, Schmidt_T, les_model, nut_,
          scalar_components, Q, DivFunction, GradFunction, **NS_namespace):
    """Set up all equations to be solved."""
    # Implicit Crank Nicolson velocity at t - dt/2
    #U_CN = dict((ui, 0.5*(u+q_1[ui])) for ui in uc_comp)

    F = {}
    Fu = {}

    # Check first if we are starting from zero velocity
    initial_u1_norm = sum([q_1[ui].vector().norm('l2') for ui in u_components])
    initial_u2_norm = sum([q_2[ui].vector().norm('l2') for ui in u_components])

    # In that case use Euler on first iteration
    beta = Constant(2.0) if abs(initial_u1_norm -
                                initial_u2_norm) > DOLFIN_EPS_LARGE else Constant(3.0)
    for i, ui in enumerate(u_components):
        # Tentative velocity step
        if not les_model is "NoModel":
            F[ui] = ((1. / (beta * dt)) * inner(3 * u - 4 * q_1[ui] + q_2[ui], v) * dx
                     + inner(inner(grad(u), 2 * u_1 - u_2), v) * dx
                     + (nu + nut_) * inner(grad(u), grad(v)) * dx +
                     inner(p_.dx(i), v) * dx - inner(f[i], v) * dx
                     + (nu + nut_) * inner(grad(v), U_AB.dx(i)) * dx)
        else:
            F[ui] = ((1. / (beta * dt)) * inner(3 * u - 4 * q_1[ui] + q_2[ui], v) * dx
                     + inner(inner(grad(u), 2 * u_1 - u_2), v) * dx
                     + nu * inner(grad(u), grad(v)) * dx + inner(p_.dx(i), v) * dx - inner(f[i], v) * dx)

        # Velocity update
        Fu[ui] = (inner(u, v) * dx - inner(q_[ui], v) * dx +
                  beta * dt / 3.0 * inner(dp_.dx(i), v) * dx)

    # Pressure update
    Fp = (inner(grad(q), grad(p) - grad(p_) + nu * grad(div(u_))) *
          dx + 3.0 / beta / dt * div(u_) * q * dx)

    # create Function to hold projection of div(u_) on Q
    divu = DivFunction(u_, Q, name='divu')

    gradp = {ui: GradFunction(p_, V, i=i, name='dpd' + ('x', 'y', 'z')[i])
             for i, ui in enumerate(u_components)}

    # Scalar with SUPG
    h = CellSize(mesh)
    #vw = v + h*inner(grad(v), u_)
    vw = v
    n = FacetNormal(mesh)
    U_CN = dict((ui, 0.5 * (u + q_1[ui])) for ui in uc_comp)
    for ci in scalar_components:
        F[ci] = ((1. / dt) * inner(u - q_1[ci], vw) * dx
                 + inner(dot(grad(U_CN[ci]), U_AB), vw) * dx
                 + (nu / Schmidt[ci] + nut_ / Schmidt_T[ci])
                 * inner(grad(U_CN[ci]), grad(vw)) * dx
                 - inner(fs[ci], vw) * dx)
            #-(nu/Schmidt[ci]+nut_/Schmidt_T[ci])*inner(dot(grad(U_CN[ci]), n), vw)*ds

    return dict(F=F, Fu=Fu, Fp=Fp, divu=divu, beta=beta, gradp=gradp)


def velocity_tentative_solve(ui, F, q_, bcs, x_, b_tmp, udiff, beta, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    b_tmp[ui][:] = x_[ui]
    A, L = system(F[ui])
    solve(A == L, q_[ui], bcs[ui])
    udiff[0] += norm(b_tmp[ui] - x_[ui])


def pressure_solve(Fp, p_, bcs, dp_, x_, nu, divu, Q, beta, **NS_namespace):
    """Solve pressure equation."""
    dp_.vector()[:] = x_['p']
    solve(lhs(Fp) == rhs(Fp), p_, bcs['p'])
    if bcs['p'] == []:
        normalize(p_.vector())
    dp_.vector()._scale(-1)
    dp_.vector().axpy(1.0, x_['p'])
    divu()
    dp_.vector().axpy(nu, divu.vector())


def velocity_update(u_components, q_, bcs, Fu, beta, gradp, dp_, dt, x_, **NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    for ui in u_components:
        solve(lhs(Fu[ui]) == rhs(Fu[ui]), q_[ui], bcs[ui])
    beta.assign(2.0)


def scalar_solve(ci, F, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    solve(lhs(F[ci]) == rhs(F[ci]), q_[ci], bcs[ci])
