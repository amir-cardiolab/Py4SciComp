__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import (Function, FunctionSpace, assemble, TestFunction, sym, grad, dx, inner, sqrt,
    FacetFunction, DirichletBC, Constant)

from .common import derived_bcs

__all__ = ['les_setup', 'les_update']


def les_setup(u_, mesh, Smagorinsky, CG1Function, nut_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving Smagorinsky-Lilly LES model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)

    # Compute cell size and put in delta
    dim = mesh.geometry().dim()
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().set_local(assemble(TestFunction(DG) * dx).array()**(1. / dim))
    delta.vector().apply('insert')

    # Set up Smagorinsky form
    Sij = sym(grad(u_))
    magS = sqrt(2 * inner(Sij, Sij))
    nut_form = Smagorinsky['Cs']**2 * delta**2 * magS
    bcs_nut = derived_bcs(CG1, bcs['u0'], u_)
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver,
                       bcs=bcs_nut, bounded=True, name="nut")
    return dict(Sij=Sij, nut_=nut_, delta=delta, bcs_nut=bcs_nut)

def les_update(nut_, **NS_namespace):
    """Compute nut_"""
    nut_()
