#!/usr/bin/python

import sys
import os
import json
import argparse
import multiprocessing
import time
import re
import traceback

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
from linesAPIs import *

helpString = """
             Generate contiguous bezier curves from joints constrained so that
             the joint arms (biMinus1_ai and ai_bi have the same length)
             Takes 1 input - text file with joint info in format (example file
             contigBezierCurves.txt):
             -
             biMinus1 : 0,-44
             ai : 0,0
             alpha : 0.6621
             -

             can store multiple joints separated by '-' -> each joint is bookended
             by '-'

             NOTE - if the length biMinus1_ai and ai_bi conflict in the file
             then the length biMinus1_ai will be used as the desired length of
             the joint arms

             """
parser = argparse.ArgumentParser(
                   prog = __prog__,
                   description = helpString,
                   add_help = True)

parser.add_argument('-i', action="store", dest="inFile", required=True, \
                    default=None, help="file containing joint info")
parser.add_argument('-o', action="store", dest="outImgName", required=False, \
                    default=None, help="file name for drawing bezier curves")
parser.add_argument('-e', action="store", dest="epsilon", required=False, \
                     default=0.2, type=float, help="epsilon - tolerance for combining bezier curves")
parser.add_argument('-projLen', action="store", dest="projectLen", required=False, type=float, \
                    default=None, help="calculate the projections of bezier curve")
parser.add_argument('--drawSegs', action="store_true", dest="drawSegs", required=False, \
                    default=False, help="draw the line segs that serve as GUIDES to the curves")

if __name__ == '__main__':
   args = parser.parse_args()
   joints = utilAPIs.getJointsDataFromTestFile(args.inFile)
   bCurves = []
   for joint in joints:
      try:
         armLen = np.linalg.norm(joint.get("ai") - joint.get("biMinus1"))
         alpha = joint.get("alpha")
         unitSquareLen = joint.get("unitSquareLen", 1.0)
         biMinus1, ai, bi = givenArmLenAndAlphaRetStdJoint(armLen, alpha, unitSquareLen)
         controlPts = getControlPtsFromJoint(biMinus1, ai, bi, alpha)
         bCurve = bezierCls(controlPts)
         bCurves.append(bCurve)
      except Exception as e:
         print("error generating bezier curve from joint - %s" % (e))
         print(traceback.format_exc())

   xformedBCurves = orientBezierCurvesToFormContiguousCurve(bCurves)

   if args.outImgName:
     plotMultipleBezierCurves(args.outImgName, xformedBCurves)

   # with the transformed contiguous set of bezier curves - calculate which
   # ones can be combined into 1 bezier curve
   #
   # the contigBezierCurves dict is of the form
   # {key: value}
   # where key = (startIdx, endIdx)
   #       value = (combined Bezier curve object, penalty of combined bezier curve)
   contigBezierCurves = {}
   for startIdx in range(len(xformedBCurves)):
      contigBCurves = [xformedBCurves[startIdx]]
      combinedCurve = None
      for endIdx in range(startIdx+1, len(xformedBCurves)):
         contigBCurves.append(xformedBCurves[endIdx])
         # check if the curves can be combined into 1 bezier curve
         try:
            print("checking if curves %s to %s can be combined" % (startIdx, endIdx))
            combinedCurve, penalty = optimizeContiguousBezierCurves(contigBCurves, args.epsilon)
            print("curves %s to %s combined successfully" % (startIdx, endIdx))
         except Exception as e:
           print("error trying to optimize contiguous bezier curve - %s" % (e))
           print(traceback.format_exc())
           if combinedCurve:
              print("curves %s to %s can be combined" % (startIdx, endIdx-1))
              contigBezierCurves[(startIdx, endIdx-1)] = (combinedCurve, penalty)
              combinedCurve = None
           break
      if combinedCurve:
         print("curves %s to %s can be combined" % (startIdx, endIdx))
         combinedCurve.setLineSegsThatMakeUpCurve(xformedBCurves[startIdx:endIdx+1])
         contigBezierCurves[(startIdx, endIdx)] = (combinedCurve, penalty)

   # loop thru the contigBezierCurves dict
   bCurvesImgPrefix = os.path.splitext(args.outImgName)[0]
   for contigBCurveIdxs, contigBCurve in contigBezierCurves.items():
      print("curve %s to %s can be combined into 1 curve" % (contigBCurveIdxs[0], contigBCurveIdxs[1]))
      combinedCurveImgName = "%s_%s_%s.png" % (bCurvesImgPrefix, contigBCurveIdxs[0], contigBCurveIdxs[1])
      contigBCurve[0].plotBezierCurve(combinedCurveImgName, projectLen=args.projectLen)
      if args.drawSegs:
         # draw the curve of the original bCurve
         curveLineSegs = contigBCurve[0].getLineSegsThatMakeUpCurve()
         curveSegs = []
         for lineSeg in curveLineSegs:
            curveSegs = utilAPIs.addOrigSegFromCurves(curveSegs, [lineSeg.getStartPt(), lineSeg.getEndPt()])
         bCurveSegs = [curveSegs]
         colors = ['red']
         if args.projectLen:
            proj, rotProj, projDerivs, rotProjPtsDerivs = contigBCurve[0].calcBothProjsOfBCurve(projLen=args.projectLen)
            for idx, projCurve in enumerate(proj):
               # calculate the contig segs that the projection curve represents - the contig segs is the joint
               # that makes up the curve - the joint is calculated by taking the 1st and last representative line seg
               # of the curve and extending it to the point of intersection
               firstLine = lineCls(projCurve[0], projCurve[1])
               lastLine = lineCls(projCurve[-1], projCurve[-2])
               intersectPt = getIntersectPtBtwn2Lines(firstLine, lastLine)
               projPts = []
               if intersectPt is not None:
                  projPts = utilAPIs.addOrigSegFromCurves(projPts, [projCurve[0], intersectPt, projCurve[-1]])
               # this is the backwards projection starting from t <= 0
               if idx == 0:
                  projPts.reverse()
                  bCurveSegs.insert(0, projPts)
                  colors.insert(0, 'green')
               else:
                  bCurveSegs.append(projPts)
                  colors.append('green')
            # draw the contig segs
            contigSegImgName = "%s_%s_%s_contigSeg.png" % (bCurvesImgPrefix, contigBCurveIdxs[0], contigBCurveIdxs[1])
            utilAPIs.drawMatPlotLib(bCurveSegs, contigSegImgName, colors)
