__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-10"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import Mesh, AutoSubDomain, near
import os
import platform

if not os.path.isfile("cylinder.xml"):
    if platform.system() == "Linux":
        os.system("wget -O cylinder.xml https://www.dropbox.com/s/d78g4cyjxl3ylay/cylinder.xml?dl=0")
    elif platform.system() == "Darwin":
        os.system("curl -L https://www.dropbox.com/s/d78g4cyjxl3ylay/cylinder.xml?dl=0 -o cylinder.xml")
    else:
        raise ImportError("Could not determine platform")

    # try:
        #os.system("gmsh mesh/cylinder.geo -2 -o mesh/cylinder.msh")
        #os.system("dolfin-convert mesh/cylinder.msh mesh/cylinder.xml")
        #os.system("rm mesh/cylinder.msh")
    # except RuntimeError:
        #os.system("wget -O cylinder.xml https://www.dropbox.com/s/d78g4cyjxl3ylay/cylinder.xml?dl=0")
        ##raise "Gmsh is required to run this demo"

mesh = Mesh("cylinder.xml")

H = 0.41
L = 2.2
D = 0.1
center = 0.2
cases = {
    1: {'Um': 0.3,
        'Re': 20.0},

    2: {'Um': 1.5,
        'Re': 100.0}
}

# Specify boundary conditions
Inlet = AutoSubDomain(lambda x, on_bnd: on_bnd and x[0] < 1e-8)
Wall = AutoSubDomain(lambda x, on_bnd: on_bnd and near(x[1] * (H - x[1]), 0))
Cyl = AutoSubDomain(lambda x, on_bnd: (on_bnd and x[0] > 1e-6 and x[0] < 1
                                      and x[1] < 3 * H / 4 and x[1] > H / 4))
Outlet = AutoSubDomain(lambda x, on_bnd: on_bnd and x[0] > L - 1e-8)


# Overload post_import_problem to choose between the two cases
def post_import_problem(NS_parameters, commandline_kwargs, **NS_namespace):
    """ Choose case - case could be defined through command line."""
    NS_parameters.update(commandline_kwargs)
    case = NS_parameters['case'] if 'case' in NS_parameters else 1
    Um = cases[case]["Um"]
    Re = cases[case]["Re"]
    Umean = 2. / 3. * Um
    nu = Umean * D / Re
    NS_parameters.update(nu=nu, Re=Re, Um=Um, Umean=Umean)

    return NS_parameters
