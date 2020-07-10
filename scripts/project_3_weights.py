#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

s3 = 3.0 ** 0.5
a = -s3
b = 0.5 * a
c = a + b
q3 = 1/3.0
e = a - 3/2.0
h = 1.0 / s3

def project(point):
    global q3
    global a
    global b
    global c
    global e
    global h
    x, y, z = point
    x -= q3
    y -= q3
    z -= q3
    return (a*x+b*y+c*z, a*x+e*y+e*z, h*x+h*y+h*z)

def test():
    for p in [
        (0,0,1),
        (0,1,0),
        (1,0,0),
        (q3, q3, q3),
        (0.3333, 0.3333, 0.3333),
        (0.333, 0.333, 0.333),
        (0.33, 0.33, 0.33),
        (0.34, 0.34, 0.34),
        (0.33, 0.33, 0.34),
    ]:
        print '(%.9f, %.9f, %.9f)' %p, '-->', '(%.6f, %.6f, %.6f)' %project(p)
    import sys
    if '--show-median' in sys.argv:
        for i in range(15):
            a = 0.05*i - 0.1
            for p in [
                (a,a,1-2*a),
                (a,1-2*a,a),
                (1-2*a,a,a),
            ]:
                print '(%.9f, %.9f, %.9f)' %p, '-->', '(%.6f, %.6f, %.6f)' %project(p)

if __name__ == '__main__':
    test()
