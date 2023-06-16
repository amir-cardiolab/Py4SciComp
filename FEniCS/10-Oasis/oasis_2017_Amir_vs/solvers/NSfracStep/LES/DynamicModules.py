__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import solve
import numpy as np


def lagrange_average(u_CG1, dt, CG1, tensdim, delta_CG1_sq, dim,
                     Sijmats, G_matr, J1=None, J2=None, Aij=None, Bij=None, **NS_namespace):
    """
    Function for Lagrange Averaging two tensors
    AijBij and BijBij, PDE's are solved implicitly.

    d/dt(J1) + u*grad(J2) = 1/T(AijBij - J1)
    d/dt(J2) + u*grad(J1) = 1/T(BijBij - J2)
    Cs**2 = J1/J2

    - eps = (dt/T)/(1+dt/T) is computed.
    - The backward terms are assembled (UNSTABLE)
    - Tensor contractions of AijBij and BijBij are computed manually.
    - Two equations are solved implicitly and easy, no linear system.
    - J1 is clipped at 1E-32 (not zero, will lead to problems).
    """

    # Update eps
    eps = (dt * (J1.vector().array() * J2.vector().array())**(1. / 8.)
           / (1.5 * np.sqrt(delta_CG1_sq.vector().array())))
    eps = eps / (1.0 + eps)

    # Compute tensor contractions
    AijBij = tensor_inner(A=Aij, B=Bij, **vars())
    BijBij = tensor_inner(A=Bij, B=Bij, **vars())

    # Compute backward convective terms J(x-dt*u) (!! NOT STABLE !!)
    """
    gradJ1 = [Function(CG1) for i in range(dim)]
    gradJ2 = [Function(CG1) for i in range(dim)]
    # Solve for the gradients of J1 and J2
    for i in range(dim):
        solve(G_matr, gradJ1[i].vector(), dt*Sijmats[i]*J1.vector(), "cg", "hypre_amg")
        solve(G_matr, gradJ2[i].vector(), dt*Sijmats[i]*J2.vector(), "cg", "hypre_amg")
    # Compute J1 - dt*dot(u_ab,grad(J1))
    J1_back = J1.vector().array()-([u_CG1[i].vector().array()*gradJ1[i].vector().array() for i in range(dim)])[0]
    # Compute J2 - dt*dot(u_ab,grad(J2))
    J2_back = J2.vector().array()-([u_CG1[i].vector().array()*gradJ2[i].vector().array() for i in range(dim)])[0]
    J2_back[J2_back < 0] = 1E3
    """

    J1_back = J1.vector().array()
    J2_back = J2.vector().array()

    # Update J1
    J1.vector().set_local(eps * AijBij + (1 - eps) * J1_back)
    J1.vector().apply("insert")

    # Update J2
    J2.vector().set_local(eps * BijBij + (1 - eps) * J2_back)
    J2.vector().apply("insert")

    # Apply ramp function on J1 to remove negative values, but not set to 0.
    J1.vector().set_local(J1.vector().array().clip(min=1E-32))
    J1.vector().apply("insert")


def tophatfilter(G_matr, G_under, unfiltered=None, filtered=None, N=1,
                 weight=0.5, **NS_namespace):
    """
    Filtering a CG1 function for applying a generalized top hat filter.
    uf = int(G*u)/int(G).

    G = CG1-basis functions.
    """

    vec_ = unfiltered.vector()
    # Apply filter N times
    for i in range(N):
        # Compute filtered quantity
        vec_ = (G_matr * vec_) * G_under.vector()
        vec_ = weight * vec_ + (1 - weight) * unfiltered.vector()

    # Zero filtered vector
    filtered.vector().zero()
    # Axpy vec_ to filtered
    filtered.vector().axpy(1.0, vec_)


def compute_Lij(Lij, uiuj_pairs, tensdim, G_matr, G_under,
                u=None, uf=None, Qij=None, **NS_namespace):
    """
    Manually compute the tensor Lij = F(uiuj)-F(ui)F(uj)
    """

    # Loop over each tensor component
    for i in range(tensdim):
        Lij[i].vector().zero()
        # Extract velocity pair
        j, k = uiuj_pairs[i]
        # Add ujuk to Lij[i]
        Lij[i].vector().axpy(1.0, u[j].vector() * u[k].vector())
        # Filter Lij[i] -> F(ujuk)
        tophatfilter(unfiltered=Lij[i], filtered=Lij[i], **vars())
        # Add to Qij if ScaleDep model
        if Qij != None:
            Qij[i].vector().zero()
            Qij[i].vector().axpy(1.0, Lij[i].vector())
        # Axpy - F(uj)F(uk)
        Lij[i].vector().axpy(-1.0, uf[j].vector() * uf[k].vector())


def compute_Mij(Mij, G_matr, G_under, Sijmats, Sijcomps, Sijfcomps, delta_CG1_sq,
                tensdim, alphaval=None, u_nf=None, u_f=None, Nij=None, **NS_namespace):
    """
    Manually compute the tensor Mij = 2*delta**2*(F(|S|Sij)-alpha**2*F(|S|)F(Sij)
    """

    Sij = Sijcomps
    Sijf = Sijfcomps
    alpha = alphaval
    deltasq = 2 * delta_CG1_sq.vector().array()

    # Apply pre-assembled matrices and compute right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        # Unfiltered rhs
        bu = [Ax * u, 0.5 * (Ay * u + Ax * v), Ay * v]
        # Filtered rhs
        buf = [Ax * uf, 0.5 * (Ay * uf + Ax * vf), Ay * vf]
    else:
        Ax, Ay, Az = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        w = u_nf[2].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        wf = u_f[2].vector()
        bu = [Ax * u, 0.5 * (Ay * u + Ax * v), 0.5 * (Az * u + Ax * w),
              Ay * v, 0.5 * (Az * v + Ay * w), Az * w]
        buf = [Ax * uf, 0.5 * (Ay * uf + Ax * vf), 0.5 * (Az * uf + Ax * wf),
               Ay * vf, 0.5 * (Az * vf + Ay * wf), Az * wf]

    for i in range(tensdim):
        # Solve for the different components of Sij
        solve(G_matr, Sij[i].vector(), bu[i], "cg", "default")
        # Solve for the different components of F(Sij)
        solve(G_matr, Sijf[i].vector(), buf[i], "cg", "default")

    # Compute magnitudes of Sij and Sijf
    magS = mag(Sij, tensdim)
    magSf = mag(Sijf, tensdim)

    # Loop over components and add to Mij
    for i in range(tensdim):
        # Compute |S|*Sij
        Mij[i].vector().set_local(magS * Sij[i].vector().array())
        Mij[i].vector().apply("insert")
        # Compute F(|S|*Sij)
        tophatfilter(unfiltered=Mij[i], filtered=Mij[i], **vars())

        # Check if Nij, assign F(|S|Sij) if not None
        if Nij != None:
            Nij[i].vector().zero()
            Nij[i].vector().axpy(1.0, Mij[i].vector())

        # Compute 2*delta**2*(F(|S|Sij) - alpha**2*F(|S|)F(Sij)) and add to Mij[i]
        Mij[i].vector().set_local(deltasq * (Mij[i].vector().array() -
                                             (alpha**2) * magSf * Sijf[i].vector().array()))
        Mij[i].vector().apply("insert")

    # Return magS for use when updating nut_
    return magS


def compute_Qij(Qij, uiuj_pairs, tensdim, G_matr, G_under, uf=None, **NS_namespace):
    """
    Function for computing Qij in ScaleDepLagrangian
    """
    # Compute Qij
    for i in range(tensdim):
        j, k = uiuj_pairs[i]
        # Filter component of Qij
        tophatfilter(unfiltered=Qij[i], filtered=Qij[i], weight=1, **vars())
        # Axpy outer(uf,uf) to Qij
        Qij[i].vector().axpy(-1.0, uf[j].vector() * uf[k].vector())


def compute_Nij(Nij, G_matr, G_under, tensdim, Sijmats, Sijfcomps, delta_CG1_sq,
                alphaval=None, u_f=None, **NS_namespace):
    """
    Function for computing Nij in ScaleDepLagrangian
    """

    Sijf = Sijfcomps
    alpha = alphaval
    deltasq = 2 * delta_CG1_sq.vector().array()

    # Need to compute F(F(Sij)), set up right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        # Filtered rhs
        buf = [Ax * uf, 0.5 * (Ay * uf + Ax * vf), Ay * vf]
    else:
        Ax, Ay, Az = Sijmats
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        wf = u_f[2].vector()
        buf = [Ax * uf, 0.5 * (Ay * uf + Ax * vf), 0.5 * (Az * uf + Ax * wf),
               Ay * vf, 0.5 * (Az * vf + Ay * wf), Az * wf]

    for i in range(tensdim):
        # Solve for the diff. components of F(F(Sij)))
        solve(G_matr, Sijf[i].vector(), buf[i], "cg", "default")

    # Compute magSf
    magSf = mag(Sijf, tensdim)

    for i in range(tensdim):
        # Filter Nij = F(|S|Sij) --> F(F(|S|Sij))
        tophatfilter(unfiltered=Nij[i], filtered=Nij[i], weight=1, **vars())
        # Compute 2*delta**2*(F(F(|S|Sij)) - alpha**2*F(F(|S))F(F(Sij)))
        Nij[i].vector().set_local(deltasq * (Nij[i].vector().array() -
                                             (alpha**2) * magSf * Sijf[i].vector().array()))
        Nij[i].vector().apply("insert")


def tensor_inner(tensdim, A=None, B=None, **NS_namespace):
    """
    Compute tensor contraction Aij:Bij of two symmetric tensors Aij and Bij.
    A numpy array is returned.
    """
    if tensdim == 3:
        contraction = (A[0].vector().array() * B[0].vector().array()
                       + 2 * A[1].vector().array() * B[1].vector().array()
                       + A[2].vector().array() * B[2].vector().array())
    else:
        contraction = (A[0].vector().array() * B[0].vector().array()
                       + 2 * A[1].vector().array() * B[1].vector().array()
                       + 2 * A[2].vector().array() * B[2].vector().array()
                       + A[3].vector().array() * B[3].vector().array()
                       + 2 * A[4].vector().array() * B[4].vector().array()
                       + A[5].vector().array() * B[5].vector().array())
    return contraction


def mag(Sij, tensdim, **NS_namespace):
    """
    Compute |S| = magS = 2*sqrt(inner(Sij,Sij))
    """
    if tensdim == 3:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S11 = Sij[2].vector().array()
        # Compute |S|
        magS = np.sqrt(2 * (S00 * S00 + 2 * S01 * S01 + S11 * S11))
    elif tensdim == 6:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S02 = Sij[2].vector().array()
        S11 = Sij[3].vector().array()
        S12 = Sij[4].vector().array()
        S22 = Sij[5].vector().array()
        # Compute |S|
        magS = np.sqrt(2 * (S00 * S00 + 2 * S01 * S01 + 2 * S02 * S02 + S11 * S11 +
                            2 * S12 * S12 + S22 * S22))

    return magS
