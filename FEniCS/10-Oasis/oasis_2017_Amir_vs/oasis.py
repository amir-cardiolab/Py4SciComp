#!/usr/bin/env python

import sys, os
sys.path.append(os.getcwd())

def main():
    assert sys.argv[1] in ('NSfracStep', 'NSCoupled')
    solver = sys.argv.pop(1)
    if solver == 'NSfracStep':
        from oasis import NSfracStep

    elif solver == 'NSCoupled':
        from oasis import NSCoupled

    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
