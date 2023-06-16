__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-25"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from ..NSCoupled import *
from ..NSCoupled import __all__


def setup(u_, p_, up_, up, u, p, v, q, nu, mesh, c, ct, q_,
          scalar_components, Schmidt, fs, CG, **NS_namespace):
    """Set up all equations to be solved. Cylindrical coordinates."""
    assert mesh.geometry().dim() == 2
    r = Expression("x[1]", domain=mesh, element=CG.ufl_element())
    n = FacetNormal(mesh)
    F_nonlinear = inner(dot(grad(u_), u_), v) * r * dx()
    F_linear = (nu * inner(grad(u_), grad(v)) * r * dx() + nu * u_[1] * v[1] / r * dx()
                - inner(p_, (r * v[1]).dx(1) + r * v[0].dx(0)) * dx()
                - inner(q, (r * u_[1]).dx(1) + r * u_[0].dx(0)) * dx()
                + nu * inner(grad(u_).T, grad(v)) * r * dx())
    #   - nu*inner(dot(grad(u_).T, n), v)*ds()

    F = F_linear + F_nonlinear
    J_linear = derivative(F_linear, up_, up)
    J_nonlinear = derivative(F_nonlinear, up_, up)

    A_pre = assemble(J_linear)
    A = Matrix(A_pre)

    # Scalar with SUPG
    Fs = {"up": F}
    Ac = {}
    Js = {}
    h = CellSize(mesh)
    vw = ct + h * inner(grad(ct), u_)
    n = FacetNormal(mesh)
    for ci in scalar_components:
        Fs[ci] = (inner(dot(grad(q_[ci]), u_), vw) * dx
                + nu / Schmidt[ci] * inner(grad(q_[ci]), grad(vw)) * r * dx
                - inner(fs[ci], vw) * r * dx
                - nu / Schmidt[ci] * inner(dot(grad(q_[ci]), n), vw) * r * ds)
        Js[ci] = derivative(Fs[ci], q_[ci], c)
        Ac[ci] = Matrix()

    return dict(F_linear=F_linear, F_nonlinear=F_nonlinear,
                J_linear=J_linear, J_nonlinear=J_nonlinear,
                A_pre=A_pre, A=A, F=F, Fs=Fs, Js=Js, Ac=Ac, r=r)


def scalar_assemble(ci, Ac, Js, bcs, **NS_namespace):
    """Assemble scalar equations."""
    Ac[ci] = assemble(Js[ci], tensor=Ac[ci])
    for bc in bcs[ci]:
        bc.apply(Ac[ci])


def scalar_solve(ci, x_, x_1, Ac, c_sol, b, omega, Fs,
                 bcs, **NS_namespace):
    """Solve scalar equations."""
    x_1[ci].zero()
    c_sol.solve(Ac[ci], x_1[ci], b[ci])
    x_[ci].axpy(-omega, x_1[ci])
    b[ci] = assemble(Fs[ci], tensor=b[ci])
    for bc in bcs[ci]:
        bc.apply(b[ci], x_[ci])


def NS_assemble(A, J_nonlinear, A_pre, bcs, **NS_namespace):
    A = assemble(J_nonlinear, tensor=A)
    A.axpy(1.0, A_pre, True)
    for bc in bcs["up"]:
        bc.apply(A)


def NS_solve(A, up_1, b, omega, up_, F, bcs, up_sol,
             **NS_namespace):
    up_1.vector().zero()
    up_sol.solve(A, up_1.vector(), b["up"])
    up_.vector().axpy(-omega, up_1.vector())
    b["up"] = assemble(F, tensor=b["up"])
    for bc in bcs["up"]:
        bc.apply(b["up"], up_.vector())
