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

import utilAPIs

fileName = os.path.abspath(__file__)
__prog__ = os.path.basename(fileName)
__dir__ = os.path.dirname(fileName)

sys.path.append(os.path.dirname(__dir__))
from bezierLib import *
import linesAPIs

helpString = """
             Test bezier curve generation from joints
             Takes 1 input - text file with joint info in format:
             -
             biMinus1 : 0,-44
             ai : 0,0
             bi : -17,35
             shift : -55,38 <--- shift to be applied
             theta : 40 <--- rotation to be applied
             -

             can store multiple joints separated by '-' -> each joint is bookended
             by '-'

             Example file : test_tools/bezierCurves.txt
             """
parser = argparse.ArgumentParser(
                   prog = __prog__,
                   description = helpString,
                   add_help = True)

parser.add_argument('-i', action="store", dest="inFile", required=True, \
                    default=None, help="file containing joint info")
parser.add_argument('-o', action="store", dest="outImgPrefix", required=False, \
                    default=None, help="file name prefix for drawing bezier curves")
parser.add_argument('-projLen', action="store", dest="projectLen", required=False, type=float, \
                    default=None, help="calculate the projections of bezier curve")
parser.add_argument('--deriv', action="store_true", dest="deriv", required=False, \
                    default=False, help="calculate the projections of bezier curve")
parser.add_argument('--drawSegs', action="store_true", dest="drawSegs", required=False, \
                    default=False, help="draw the line segs that serve as GUIDES to the curves")

if __name__ == '__main__':
   args = parser.parse_args()
   biMinus1_ai_bi_alpha = utilAPIs.getJointsPtsFromTestFile(args.inFile)
   count = 1
   nameRoot = "default_out"

   for biMinus1, ai, bi, alpha in biMinus1_ai_bi_alpha:
      segPts = []
      # the projSegPts are pts of segments derived from projected curves
      # the 1st elem is projected from behind t<=0, the 2nd elem is project from t>=1
      projSegPts = [[], []]

      segPts = utilAPIs.addOrigSegFromCurves(segPts, [biMinus1, ai, bi])
      # for bezier curves - Z0 = biMinus1, Z3 = bi
      #   Z1 = biMinus1 + (vector biMinus1_ai) * alpha
      #   Z2 = bi + (vector bi_ai) * alpha
      z1 = biMinus1 + (ai - biMinus1) * alpha
      z2 = bi + (ai - bi) * alpha
      bCurve = bezierCls([biMinus1, z1, z2, bi])
      print("Bezier curve %s info" % (count))
      bCurve.displayBezierCurveInfo()

      bCurveOrigPts = []
      bCurveOrigDeriv = []
      bCurveOrigPtsWithThisDeriv = []

      bCurveRotatedPts = []
      bCurveRotatedDeriv = []
      bCurveRotatedPtsWithThisDeriv = []

      t = 0
      while t <= 1:
         bCurveOrigPts.append(bCurve.getXYValOfCurve(t,0))
         derivGiven_t = bCurve.getDYDXgiven_t(t,0)
         bCurveOrigDeriv.append(derivGiven_t)
         bCurveOrigPtsWithThisDeriv.append(bCurve.getPtsOnCurveWithDYDXVal(derivGiven_t,0))

         bCurveRotatedPts.append(bCurve.getXYValOfCurve(t,1))
         derivGiven_t = bCurve.getDYDXgiven_t(t,1)
         bCurveRotatedDeriv.append(derivGiven_t)
         bCurveRotatedPtsWithThisDeriv.append(bCurve.getPtsOnCurveWithDYDXVal(derivGiven_t,1))

         t += 0.1

      print("Bezier curve %s points = %s" % (count, bCurveOrigPts))
      print("Bezier curve %s derivative %s" % (count, bCurveOrigDeriv))
      print("Bezier curve %s pts on curve %s with given derivative" % (count, bCurveOrigPtsWithThisDeriv))

      print("Bezier curve %s rotated points = %s" % (count, bCurveRotatedPts))
      print("Bezier curve %s rotated derivative %s" % (count, bCurveRotatedDeriv))
      print("Bezier curve %s pts on rotated curve %s with given derivative" % (count, bCurveRotatedPtsWithThisDeriv))

      if args.projectLen:
         projLen = args.projectLen
         proj, rotProj, projDeriv, rotProjDeriv = bCurve.calcBothProjsOfBCurve(projLen=args.projectLen, projDelta_t=0.01)
         for tOrig in range(2):
            if tOrig == 0:
               projDir = "projection of t < 0"
            elif tOrig == 1:
               projDir = "projection of t > 1"
            else:
               print("Unrecognized tOrig %s - break" % (tOrig))
               break

            bCurveOrigPts = proj[tOrig]
            bCurveOrigDeriv = projDeriv[tOrig]
            bCurveOrigPtsWithThisDeriv.clear()

            bCurveRotatedPts = rotProj[tOrig]
            bCurveRotatedDeriv = rotProjDeriv[tOrig]
            bCurveRotatedPtsWithThisDeriv.clear()

            for bCurveDeriv in bCurveOrigDeriv:
               bCurveOrigPtsWithThisDeriv.append(bCurve.getPtsOnCurveWithDYDXVal(bCurveDeriv,0))
            for bCurveDeriv in bCurveRotatedDeriv:
               bCurveRotatedPtsWithThisDeriv.append(bCurve.getPtsOnCurveWithDYDXVal(bCurveDeriv,1))

            # the segs that make up the PROJECTED bezier curve sections are as if created this
            # curve using Potrace and the GUIDE-LINES - since we calculate the projection and get the bCurve
            # as line segs - the 2 GUIDE-LINES are:
            #  The start pt of the first seg, The end pt of the last seg, and the intersect pt between
            #  the first seg and the last seg
            firstLine = lineCls(bCurveOrigPts[0], bCurveOrigPts[1])
            lastLine = lineCls(bCurveOrigPts[-1], bCurveOrigPts[-2])
            intersectPt = linesAPIs.getIntersectPtBtwn2Lines(firstLine, lastLine)

            if intersectPt is not None:
               projSegPts[tOrig] = utilAPIs.addOrigSegFromCurves(projSegPts[tOrig], [bCurveOrigPts[0], intersectPt, bCurveOrigPts[-1]])

            # if the projected segs are from t<0 -> need to flip the order of the points so that
            # the direction of the pts is the same as from 0 <= t <= 1 and then from t > 1
            if tOrig == 0:
               projSegPts[tOrig].reverse()

            print("Bezier curve %s - %s points = %s" % (projDir, count, bCurveOrigPts))
            print("Bezier curve %s - %s derivative %s" % (projDir, count, bCurveOrigDeriv))
            print("Bezier curve %s - %s pts on curve %s with given derivative" % (projDir, count, bCurveOrigPtsWithThisDeriv))

            print("Bezier curve %s - %s rotated points = %s" % (projDir, count, bCurveRotatedPts))
            print("Bezier curve %s - %s rotated derivative %s" % (projDir, count, bCurveRotatedDeriv))
            print("Bezier curve %s - %s pts on rotated curve %s with given derivative" % (projDir, count, bCurveRotatedPtsWithThisDeriv))

      if args.outImgPrefix:
         # first split the image name from the prefix
         root_ext = os.path.splitext(args.outImgPrefix)
         nameRoot = root_ext[0]
         imgName = "%s_%s" % (nameRoot, count)
         imgName += ".png"
         bCurve.plotBezierCurve(imgName, drawRotated=True, drawDeriv=args.deriv, unitDU=20, projectLen=args.projectLen)

      if args.drawSegs:
         print("segPts = %s" % (segPts))
         print("projPts = %s" % (projSegPts))
         origImgName = "%s_%s_%s%s" % ("origContigSeg", nameRoot, count, ".png")
         projImgName = "%s_%s_%s%s" % ("projContigSeg", nameRoot, count, ".png")
         # first draw the original contig segs
         utilAPIs.drawMatPlotLib([segPts], origImgName)
         utilAPIs.drawMatPlotLib([projSegPts[0], segPts, projSegPts[1]], projImgName, ['green', 'red', 'green'])

      count += 1
