__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import (Function, FunctionSpace, TestFunction, sym, grad, dx, inner,
    sqrt, TrialFunction, project, CellVolume, as_vector, solve, Constant,
    LagrangeInterpolator, assemble, FacetFunction, DirichletBC)
from .DynamicModules import (tophatfilter, lagrange_average, compute_Lij,
                            compute_Mij)
import numpy as np

__all__ = ['les_setup', 'les_update']


def les_setup(u_, mesh, assemble_matrix, CG1Function, nut_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving the Germano Dynamic LES model applying
    Lagrangian Averaging.
    """

    # Create function spaces
    CG1 = FunctionSpace(mesh, "CG", 1)
    p, q = TrialFunction(CG1), TestFunction(CG1)
    dim = mesh.geometry().dim()

    # Define delta and project delta**2 to CG1
    delta = pow(CellVolume(mesh), 1. / dim)
    delta_CG1_sq = project(delta, CG1)
    delta_CG1_sq.vector().set_local(delta_CG1_sq.vector().array()**2)
    delta_CG1_sq.vector().apply("insert")

    # Define nut_
    Sij = sym(grad(u_))
    magS = sqrt(2 * inner(Sij, Sij))
    Cs = Function(CG1)
    nut_form = Cs**2 * delta**2 * magS
    # Create nut_ BCs
    ff = FacetFunction("size_t", mesh, 0)
    bcs_nut = []
    for i, bc in enumerate(bcs['u0']):
        bc.apply(u_[0].vector())  # Need to initialize bc
        m = bc.markers()  # Get facet indices of boundary
        ff.array()[m] = i + 1
        bcs_nut.append(DirichletBC(CG1, Constant(0), ff, i + 1))
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver,
                       bcs=bcs_nut, bounded=True, name="nut")

    # Create functions for holding the different velocities
    u_CG1 = as_vector([Function(CG1) for i in range(dim)])
    u_filtered = as_vector([Function(CG1) for i in range(dim)])
    dummy = Function(CG1)
    ll = LagrangeInterpolator()

    # Assemble required filter matrices and functions
    G_under = Function(CG1, assemble(TestFunction(CG1) * dx))
    G_under.vector().set_local(1. / G_under.vector().array())
    G_under.vector().apply("insert")
    G_matr = assemble(inner(p, q) * dx)

    # Set up functions for Lij and Mij
    Lij = [Function(CG1) for i in range(dim * dim)]
    Mij = [Function(CG1) for i in range(dim * dim)]
    # Check if case is 2D or 3D and set up uiuj product pairs and
    # Sij forms, assemble required matrices
    Sijcomps = [Function(CG1) for i in range(dim * dim)]
    Sijfcomps = [Function(CG1) for i in range(dim * dim)]
    # Assemble some required matrices for solving for rate of strain terms
    Sijmats = [assemble_matrix(p.dx(i) * q * dx) for i in range(dim)]
    if dim == 3:
        tensdim = 6
        uiuj_pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    else:
        tensdim = 3
        uiuj_pairs = ((0, 0), (0, 1), (1, 1))

    # Set up Lagrange functions
    JLM = Function(CG1)
    JLM.vector()[:] += 1E-32
    JMM = Function(CG1)
    JMM.vector()[:] += 1

    return dict(Sij=Sij, nut_form=nut_form, nut_=nut_, delta=delta, bcs_nut=bcs_nut,
                delta_CG1_sq=delta_CG1_sq, CG1=CG1, Cs=Cs, u_CG1=u_CG1,
                u_filtered=u_filtered, ll=ll, Lij=Lij, Mij=Mij, Sijcomps=Sijcomps,
                Sijfcomps=Sijfcomps, Sijmats=Sijmats, JLM=JLM, JMM=JMM, dim=dim,
                tensdim=tensdim, G_matr=G_matr, G_under=G_under, dummy=dummy,
                uiuj_pairs=uiuj_pairs)

def les_update(u_ab, nut_, nut_form, dt, CG1, delta, tstep,
               DynamicSmagorinsky, Cs, u_CG1, u_filtered, Lij, Mij,
               JLM, JMM, dim, tensdim, G_matr, G_under, ll,
               dummy, uiuj_pairs, Sijmats, Sijcomps, Sijfcomps, delta_CG1_sq,
               **NS_namespace):

    # Check if Cs is to be computed, if not update nut_ and break
    if tstep % DynamicSmagorinsky["Cs_comp_step"] != 0:
        # Update nut_
        nut_()
        # Break function
        return

    # All velocity components must be interpolated to CG1 then filtered
    for i in range(dim):
        # Interpolate to CG1
        ll.interpolate(u_CG1[i], u_ab[i])
        # Filter
        tophatfilter(unfiltered=u_CG1[i], filtered=u_filtered[i], **vars())

    # Compute Lij applying dynamic modules function
    compute_Lij(u=u_CG1, uf=u_filtered, **vars())

    # Compute Mij applying dynamic modules function
    alpha = 2.0
    magS = compute_Mij(alphaval=alpha, u_nf=u_CG1, u_f=u_filtered, **vars())

    # Lagrange average Lij and Mij
    lagrange_average(J1=JLM, J2=JMM, Aij=Lij, Bij=Mij, **vars())

    # Update Cs = sqrt(JLM/JMM) and filter/smooth Cs, then clip at 0.3.
    """
    Important that the term in nut_form is Cs**2 and not Cs
    since Cs here is stored as sqrt(JLM/JMM).
    """
    Cs.vector().set_local(np.sqrt(JLM.vector().array() / JMM.vector().array()))
    Cs.vector().apply("insert")
    tophatfilter(unfiltered=Cs, filtered=Cs, N=2, weight=1., **vars())
    Cs.vector().set_local(Cs.vector().array().clip(max=0.3))
    Cs.vector().apply("insert")

    # Update nut_
    nut_.vector().set_local(Cs.vector().array()**2 *
                            delta_CG1_sq.vector().array() * magS)
    nut_.vector().apply("insert")
