__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

import warnings
from dolfin import FacetFunction, DirichletBC, Constant


def derived_bcs(V, original_bcs, u_):

    new_bcs = []

    # Check first if user has declared subdomains
    subdomain = original_bcs[0].user_sub_domain()
    if subdomain is None:
        mesh = V.mesh()
        ff = FacetFunction("size_t", mesh, 0)
        for i, bc in enumerate(original_bcs):
            bc.apply(u_[0].vector())  # Need to initialize bc
            m = bc.markers()  # Get facet indices of boundary
            ff.array()[m] = i + 1
            new_bcs.append(DirichletBC(V, Constant(0), ff, i + 1))

    else:
        for i, bc in enumerate(original_bcs):
            subdomain = bc.user_sub_domain()
            new_bcs.append(DirichletBC(V, Constant(0), subdomain))

    return new_bcs
