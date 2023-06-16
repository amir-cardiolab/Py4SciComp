__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from ..NSCoupled import *
from ..NSCoupled import __all__


def setup(u_, p_, up_, up, u, p, v, q, nu, f, mesh, c, ct, q_,
          scalar_components, Schmidt, fs, **NS_namespace):
    """Set up all equations to be solved."""
    F = (inner(dot(grad(u_), u_), v) * dx + nu * inner(grad(u_), grad(v)) * dx
        - inner(p_, div(v)) * dx - inner(q, div(u_)) * dx + inner(f, v) * dx)

    J = derivative(F, up_, up)
    A = Matrix()

    # Scalar with SUPG
    Fs = {"up": F}
    Ac = {}
    Js = {}
    h = CellSize(mesh)
    vw = ct + h * inner(grad(ct), u_)
    n = FacetNormal(mesh)
    for ci in scalar_components:
        Fs[ci] = (inner(dot(grad(q_[ci]), u_), vw) * dx
                + nu / Schmidt[ci] * inner(grad(q_[ci]), grad(vw)) * dx
                - inner(fs[ci], vw) * dx
                - nu / Schmidt[ci] * inner(dot(grad(q_[ci]), n), vw) * ds)
        Js[ci] = derivative(Fs[ci], q_[ci], c)
        Ac[ci] = Matrix()

    return dict(F=F, J=J, A=A, Fs=Fs, Js=Js, Ac=Ac)


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


def NS_assemble(A, J, bcs, **NS_namespace):
    A = assemble(J, tensor=A)
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
