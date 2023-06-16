__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import (Function, FunctionSpace, assemble, TestFunction, sym, grad, tr,
    Identity, dx, inner, Max, FacetFunction, DirichletBC, Constant)

from .common import derived_bcs

__all__ = ['les_setup', 'les_update']


def les_setup(u_, mesh, Wale, bcs, CG1Function, nut_krylov_solver, **NS_namespace):
    """Set up for solving Wale LES model"""
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)

    # Compute cell size and put in delta
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().axpy(1.0, assemble(TestFunction(DG) * dx))

    # Set up Wale form
    Gij = grad(u_)
    Sij = sym(Gij)
    Skk = tr(Sij)
    dim = mesh.geometry().dim()
    Sd = sym(Gij * Gij) - 1. / 3. * Identity(mesh.geometry().dim()) * Skk * Skk
    nut_form = (Wale['Cw']**2 * pow(delta, 2. / dim) * pow(inner(Sd, Sd), 1.5)
                / (Max(pow(inner(Sij, Sij), 2.5) + pow(inner(Sd, Sd), 1.25), 1e-6)))
    ff = FacetFunction("size_t", mesh, 0)
    bcs_nut = derived_bcs(CG1, bcs['u0'], u_)
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver,
                       bcs=bcs_nut, name='nut', bounded=True)
    return dict(Sij=Sij, Sd=Sd, Skk=Skk, nut_=nut_, delta=delta, bcs_nut=bcs_nut)

def les_update(nut_, **NS_namespace):
    """Compute nut_"""
    nut_()
