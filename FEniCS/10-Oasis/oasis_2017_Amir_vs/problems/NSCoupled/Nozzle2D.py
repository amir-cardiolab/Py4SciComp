from ..NSCoupled import *
from ..Nozzle2D import *

from math import sqrt, pi
from fenicstools import StructuredGrid, StatisticsProbes
import sys
from numpy import array, linspace

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    re_high = False
    NS_parameters.update(
        omega=0.4,
        nu=0.0035 / 1056.,
        folder="nozzle_results",
        max_error=1e-13,
        max_iter=25,
        re_high=re_high,
        solver='cylindrical')


def create_bcs(VQ, mesh, sys_comp, re_high, **NS_namespce):
    # Q = 5.21E-6 if not re_high else 6.77E-5  # From FDA
    Q = 5.21E-6 if not re_high else 3E-5  # From FDA
    r_0 = 0.006
    # Analytical, could be more exact numerical, different r_0
    u_maks = Q / (4. * r_0 * r_0 * (1. - 2. / pi))
    #inn = Expression(("u_maks * cos(sqrt(pow(x[1],2))/r_0/2.*pi)", "0"), u_maks=u_maks, r_0=r_0)
    inn = Expression(("u_maks * (1-x[1]*x[1]/r_0/r_0)", "0"), u_maks=u_maks, r_0=r_0)

    bc0 = DirichletBC(VQ.sub(0),    inn,  inlet)
    bc1 = DirichletBC(VQ.sub(0), (0, 0),  walls)
    bc2 = DirichletBC(VQ.sub(0).sub(1), 0, centerline)

    return dict(up=[bc0, bc1, bc2])


def pre_solve_hook(mesh, V, **NS_namespace):
    # Normals and facets to compute flux at inlet and outlet
    normal = FacetNormal(mesh)
    Inlet = AutoSubDomain(inlet)
    Outlet = AutoSubDomain(outlet)
    Walls = AutoSubDomain(walls)
    Centerline = AutoSubDomain(centerline)
    facets = FacetFunction('size_t', mesh, 0)
    Inlet.mark(facets, 1)
    Outlet.mark(facets, 2)
    Walls.mark(facets, 3)
    Centerline.mark(facets, 4)

    z_senterline = linspace(-0.18269, 0.320, 1000)
    x = array([[i, 0.0] for i in z_senterline])
    senterline = StatisticsProbes(x.flatten(), V)

    return dict(uv=Function(V), senterline=senterline, facets=facets,
                normal=normal)


def temporal_hook(**NS_namespace):
    pass
