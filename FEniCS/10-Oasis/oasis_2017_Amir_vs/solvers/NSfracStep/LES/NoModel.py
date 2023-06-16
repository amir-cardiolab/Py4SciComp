__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Constant

__all__ = ['les_setup', 'les_update']

def les_setup(**NS_namespace):
    return dict(nut_=Constant(0))


def les_update(**NS_namespace):
    pass
