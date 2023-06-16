__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-24'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import (Function, FunctionSpace, assemble, TestFunction, sym, grad,
    Vector, dx, inner, TrialFunction, sqrt, Matrix, dot, interpolate, Constant,
    DirichletBC, KrylovSolver)

from .common import derived_bcs

__all__ = ['les_setup', 'les_update']


def les_setup(u_, mesh, KineticEnergySGS, assemble_matrix, CG1Function, nut_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving the Kinetic Energy SGS-model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.geometry().dim()
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().axpy(1.0, assemble(TestFunction(DG) * dx))
    delta.vector().set_local(delta.vector().array()**(1. / dim))
    delta.vector().apply('insert')

    Ck = KineticEnergySGS["Ck"]
    ksgs = interpolate(Constant(1E-7), CG1)
    bc_ksgs = DirichletBC(CG1, 0, "on_boundary")
    A_mass = assemble_matrix(TrialFunction(CG1) * TestFunction(CG1) * dx)
    nut_form = Ck * delta * sqrt(ksgs)
    bcs_nut = derived_bcs(CG1, bcs['u0'], u_)
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver,
                       bcs=bcs_nut, bounded=True, name="nut")
    At = Matrix()
    bt = Vector(nut_.vector())
    ksgs_sol = KrylovSolver("bicgstab", "additive_schwarz")
    #ksgs_sol.parameters["preconditioner"]["structure"] = "same_nonzero_pattern"
    ksgs_sol.parameters["error_on_nonconvergence"] = False
    ksgs_sol.parameters["monitor_convergence"] = False
    ksgs_sol.parameters["report"] = False
    del NS_namespace
    return locals()


def les_update(nut_, nut_form, A_mass, At, u_, dt, bc_ksgs, bt, ksgs_sol,
               KineticEnergySGS, CG1, ksgs, delta, **NS_namespace):

    p, q = TrialFunction(CG1), TestFunction(CG1)

    Ck = KineticEnergySGS["Ck"]
    Ce = KineticEnergySGS["Ce"]

    Sij = sym(grad(u_))
    assemble((dt * inner(dot(u_, 0.5 * grad(p)), q) * dx
             + inner((dt * Ce * sqrt(ksgs) / delta) * 0.5 * p, q) * dx
             + inner(dt * Ck * sqrt(ksgs) * delta * grad(0.5 * p), grad(q)) * dx),
             tensor=At)

    assemble((dt * 2 * Ck * delta * sqrt(ksgs) *
             inner(Sij, grad(u_)) * q * dx), tensor=bt)
    bt.axpy(1.0, A_mass * ksgs.vector())
    bt.axpy(-1.0, At * ksgs.vector())
    At.axpy(1.0, A_mass, True)

    # Solve for ksgs
    bc_ksgs.apply(At, bt)
    ksgs_sol.solve(At, ksgs.vector(), bt)
    ksgs.vector().set_local(ksgs.vector().array().clip(min=1e-7))
    ksgs.vector().apply("insert")

    # Update nut_
    nut_()
