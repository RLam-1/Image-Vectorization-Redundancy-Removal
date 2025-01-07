#!/usr/bin/python

import sys
import os
import json
import argparse
import multiprocessing
import time
import re

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy import linalg as LA

__prog__ = os.path.basename(sys.argv[0])

helpString = """
             bezierClass - calculates bezier properties
             """
parser = argparse.ArgumentParser(
                   prog = __prog__,
                   description=helpString,
                   add_help=True)
parser.add_argument('-i', action="store", dest="tuples", required=True, \
                        default=None, help="z0, z1, z2, z3 of bezier curve")

def calculateAreaOfCubicBezier(z0, z1, z2, z3):
   # the equation to calculate the area under a cubic bezier curve is
   # 3 / 10 * y1 * x0 - 3 / 20 * y1 * x2 -
   # 3 / 20 * y1 * x3 - 3 / 10 * y0 * x1 -
   # 3 / 20 * y0 * x2 - 1 / 20 * y0 * x3 +
   # 3 / 20 * y2 * x0 + 3 / 20 * y2 * x1 -
   # 3 / 10 * y2 * x3 + 1 / 20 * y3 * x0 +
   # 3 / 20 * y3 * x1 + 3 / 10 * y3 * x2

   area = 3/10*z1[1]*z0[0] - 3/20*z1[1]*z2[0] - \
          3/20*z1[1]*z3[0] - 3/10*z0[1]*z1[0] - \
          3/20*z0[1]*z2[0] - 1/20*z0[1]*z3[0] + \
          3/20*z2[1]*z0[0] + 3/20*z2[1]*z1[0] - \
          3/10*z2[1]*z3[0] + 1/20*z3[1]*z0[0] + \
          3/20*z3[1]*z1[0] + 3/10*z3[1]*z2[0]

   return abs(area)

def numericalAreaOfCubicBezier(z0, z1, z2, z3):
   t0 = 0
   Area = 0
   while t0 < 1:
      t1 = t0 + 0.01
      x0 = ((1-t0)**3)*z0[0] + (3*t0*((1-t0)**2))*z1[0] + \
           (3*(t0**2)*(1-t0))*z2[0] + (t0**3)*z3[0]
      y0 = ((1-t0)**3)*z0[1] + (3*t0*((1-t0)**2))*z1[1] + \
           (3*(t0**2)*(1-t0))*z2[1] + (t0**3)*z3[1]
      x1 = ((1-t1)**3)*z0[0] + (3*t1*((1-t1)**2))*z1[0] + \
           (3*(t1**2)*(1-t1))*z2[0] + (t1**3)*z3[0]
      y1 = ((1-t1)**3)*z0[1] + (3*t1*((1-t1)**2))*z1[1] + \
           (3*(t1**2)*(1-t1))*z2[1] + (t1**3)*z3[1]

      segArea = (abs(x1-x0)*(y0+y1))/2
      Area += segArea
      t0 = t1

   return Area


if __name__ == '__main__':
   results = parser.parse_args()
   bracketRe = re.compile("\((.+?)\)")

   a = []
   for xy in bracketRe.findall(results.tuples):
      coords = xy.split(',')
      a.append((float(coords[0]), float(coords[1])))

   z0, z1, z2, z3 = a
   print("z0 is %s z1 is %s z2 is %s z3 is %s" % (z0, z1, z2, z3))
   print("analytical area is %s" % (calculateAreaOfCubicBezier(z0, z1, z2, z3)))
 #  print("numerical area is %s" % (numericalAreaOfCubicBezier(z0, z1, z2, z3)))
