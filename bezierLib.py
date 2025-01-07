#!/usr/bin/python
#
# This file contains the bezierCls - this class characterizes
#  a bezier curve by storing the control points (which are formatted
#  as tuples) in a list in order (z0, z1, z2, ..., zn)
#
#  This class also calculates attributes of the bezier curve such as
#   1) area under the bezier curve
#   2) the derivative of the curve (dy/dx) at any given pt (x,y)
#   3) given the derivative value (dy/dx) get the (x,y) pt on the curve
#      which has said derivative
#   4) generate all lines of the bezier curve
#  Oct 9 2022 - currently only cubic bezier curves are supported (only have
#               4 control points)
#             - also - only support the class of bezier curves where:
#           straight lines z0z1 and z3z2 intersect at point o - ie. z0, o, z3
#           form a triangle AND z1 lies between z0 and o and z2 lies between o and z3
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
from linesAPIs import *

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

# API to make sure list of bezier curves is contiguous and oriented so that the
# end pt of 1 is the start of the next one
#
# INPUT: list of bezier curves
def checkBCurvesContiguousAndOrient(bezierCurves):
   # 1)
   # check to make sure that the bezier curves are contiguous
   # use the lines Z0_Z3 of the bezier curves
   bCurvesZ0_Z3Lines = [bCurve.lineZ0_Z3[0] for bCurve in bezierCurves]
   bCurvesContiguous = checkIfLinesSinglyContiguous(bCurvesZ0_Z3Lines)
   if not bCurvesContiguous:
      print("Bezier curves not contiguous")
      return None

   # it has been observed that np.array_equal may be incorrectly set to false
   # because of slight differences in end pt / start pt values - re-set the
   # control pts of the bezier curves
   for idx, bCurve in enumerate(bezierCurves):
      bCurveCtlPts = bCurve.getControlPts()
      bCurveCtlPts[0] = bCurvesZ0_Z3Lines[idx].termPt1
      bCurveCtlPts[-1] = bCurvesZ0_Z3Lines[idx].termPt2
      bCurve.setControlPts(bCurveCtlPts)

   # given that the bezier curves have been determined to be contiguous
   # make sure that they are oriented so that the pt Z3 of bCurve i and Z0 of
   # bCurve i+1 are the same point

   for i in range(len(bezierCurves)-1):
      if i == 0 and \
         np.array_equal(bezierCurves[i].getFirstPt(), bezierCurves[i+1].getFirstPt()) or \
         np.array_equal(bezierCurves[i].getFirstPt(), bezierCurves[i+1].getLastPt()):
         print("First bezier curve of the segment not oriented properly - reverse orientation")
         bezierCurves[i].reverseOrientationOfBezierCurve()

      if not np.array_equal(bezierCurves[i].getLastPt(), bezierCurves[i+1].getFirstPt()):
         print("bCurve %s and %s are not oriented properly - reverse direction of %s" % \
               (i, i+1, i+1))
         bezierCurves[i+1].reverseOrientationOfBezierCurve()

   # after reorientation - do one final check
   for i in range(1, len(bezierCurves)):
      if not np.array_equal(bezierCurves[i].getFirstPt(), bezierCurves[i-1].getLastPt()):
         print("Error - even after reorientation bezier curve %s and %s are not contiguous" %\
               (i-1, i))
         return None

   return bezierCurves

# this API checks the convexity of the contiguous curves to make sure that they have
# the same convexity (ie.. the bCurves curve in the same direction)
def checkConvexityOfBCurves(bezierCurves):
   convexity = bezierCurves[0].getConvexity()
   for i in range(1, len(bezierCurves)):
      if (bezierCurves[i].getConvexity()*convexity) < 0:
         print("bezier curve %s has differing convexity with the rest of the contiguous bCurves" % (i))
         return False

   return True

# this API checks if the list of bezier curves passed in is contiguous
# For series of bezier curves to even be considered as candidate for combination
#  need to satisfy 3 conditions
# 1) the bezier curves must be CONTIGUOUS
# 2) the bezier curves must agree in convexity (meaning that the cross product
#    of the joints guide biMinus1_ai and ai_bi must have constant sign (+ve or -ve))
# 3) the total change in direction is less than 179 degrees - meaning that the joints
#    guide biMinus1_ai of the 1st bezier curve and the joints guide ai_bi of the LAST
#    bezier curve have a different in angle of no greater than 179 degrees
#    Since the change in direction of the angle can be both ways (since the change of
#    direction is a circle) - the direction of the change is the same as the convexity
#    of the bezier curve

# this API takes as input a list of contiguous bezier curves and see if
# these curves can be combined into 1 bezier curve by:
#  1) Taking the Z0_0 of the first curve and the Z3_0 of the last curve
#     as the slope of the combined curve's Z0_0 and Z3_0 since the combined curve
#     is tangent to the smaller curves
#  2) calculate the area occupied under the original curves and the
#     Z0_Z3 line of the combined curve
#  3) Given the area under the combined curve, can calculate ALPHA (and by extension
#     the Z1, Z2 of the combined curve)
#  4) Then we check if the combined curve matches input contiguous curves using the conditions below
#      - For each pair of contiguous input bezier curves, we have 0i, 0i+1 (which is the
#        intersect pt of Z0Z1 and Z3Z2 for curve i and i+1 respectively
#        Find the pt on the proposed optimized curve that is parallel to 0i_0i+1 (call it pt zi)
#        Let di be the Euclidean distance between zi and the line 0i_0i+1
#      - For each individual bezier curve where the lines Z0_0 and Z3_0 are actual lines in the
#        image that are connected to each other (can think of those lines as bezier guides)
#        draw unit square centered around intersect pt 0 and oriented such that
#        the line z0_0 is vertical and crosses the bottom side of the square
#        Next, get the line L that is parallel to Z0_Z3 that is closest to Z0_Z3
#        and touches the unit square
#        Also, get zi' on the proposed optimized curve, where zi' is the pt where the tangent
#         of the optimized curve is parallel to the line L.
#        Get the Euclidean distance between zi' and L (call it di')
#       We determine the optimized curve to be a good approximate if all di <= Epsilon
#       and di' >= Epsilon (constant determined by user - guideline value = 0.2)
#       Also,check if projection of pt zi on optimized curve lies between 0i_0i+1
#
#  NOTE: bi-1 -> a -> bi+1 (which is the midpt of line 1, the pt shared by line 1 and line 2,
#        and the midpt of line 2 of a contigSeg) maps to its corresponding bezier curve Z0, 0, Z3
def optimizeContiguousBezierCurves(bezierCurves, epsilon=0.2, verifyCurves=True):
   if len(bezierCurves) < 2:
      print("len of bezier curves to optimize = %s - error less than 2" % len(bezierCurves))
      return None

   if verifyCurves:
      bezierCurves = checkBCurvesContiguousAndOrient(bezierCurves)

      if not bezierCurves:
         print("Bezier curves not contiguous - cannot optimize")
         return None

      # next check that the convexity of the curves is in the same direction
      if not checkConvexityOfBCurves(bezierCurves):
         print("Bezier curves do not have same convexity")
         return None

   # now that both convexity and contiguous conditions are met - check to make sure that
   #  the last condition is met (the biMinus1_ai of the first curve and the ai_bi of the last curve
   #  change direction with angle < 179 degrees - ie. the total change in direction of the curve
   #  is < 179 degrees)
   cosMaxAngleChange = math.cos(math.radians(179))
   cosAngleChange = np.dot(bezierCurves[0].lineZ0_0[0].unitVect, bezierCurves[-1].line0_Z3[0].unitVect)

   if cosAngleChange < cosMaxAngleChange:
      print("cosAngleChange = %s, cosMaxAngleChange = %s" \
             % (totalConvexity, convexity, cosAngleChange, cosMaxAngleChange))
      return None

   # if the bezier curves are contiguous - check to see if the bezier curves can
   # be combined into 1 curve - the 1 optimized curve's control pts are made of:
   #  z0 is the z0 of the 1st curve
   #  z1 is the 0 of the 1st curve
   #  z2 is the 0 of the last curve
   #  z3 is the z3 of the last curve
   #
   #  NOTE: the z1, z2 are not the TRUE z1, z2 of this optimized curve. The points
   #  passed in as z1, z2 simply lie on the same line as the TRUE z1, z2
   #  and so can be passed in to determine the intersect pt 0 and the "joint guides"
   #  such as Z0_0, Z3_0, and Z0_Z3
   print(" === Creating combined bezier curve === ")
   combinedCurve = bezierCls([bezierCurves[0].controlPts[0][0], \
                              bezierCurves[0].intersectPt[0], \
                              bezierCurves[-1].intersectPt[0], \
                              bezierCurves[-1].controlPts[0][-1]])

   # the combined area under the input list of contiguous bezier curves
   # is the area of the individual bezier curves PLUS the area of the polygon formed
   # by the line segments Z0_Z3 for each bezier curve (the polygon is closed by the
   # imaginary line of the Z3 of the last curve to the Z0 of the 1st curve)
   area = 0.0
   for bCurve in bezierCurves:
      area += bCurve.calculateAreaOfCubicBezier()
   area += calcAreaUnderSinglyContiguousLine([bCurve.lineZ0_Z3[0] for bCurve in bezierCurves])

   # given combined area calculate new Z1, Z2
   if not combinedCurve.givenAreaCalculateZ1Z2(area):
      print("Failed to calculate Z1, Z2 given area")
      return None

   diEntries = []
   diPrimeEntries = []
   # now that the combined curve is finalized - check to see if the combined curve
   #  is a suitable curve to represent the input list of bezier curves
   for i in range(len(bezierCurves)-1):
      # 1st condition - for each pair of adjacent bezier curves
      # take the pair of their intersection pts (0) and generate a line
      #  - take the dx, dy of this line
      # given the dx, dy of this line, find the pt zi on the combinedCurve that has this
      #  dy/dx
      # In other words, find the pt on the combinedCurve whose tangent is parallel
      # to a0_a1, where ao is the intersection pt of curve 0 and a1 is the intersection
      # pt of cruve 1
      # and get the Euclidean distance between zi and a0_a1
      ai_aiPlus1 = lineCls(bezierCurves[i].intersectPt[0], bezierCurves[i+1].intersectPt[0])
      if ai_aiPlus1.unitVect[0] == 0:
         dy_dx = float('inf')
      else:
         dy_dx = ai_aiPlus1.unitVect[1] / ai_aiPlus1.unitVect[0]
      ziPts = combinedCurve.getPtsOnCurveWithDYDXVal(dy_dx)
      di = None
      if len(ziPts) < 1:
         print("Unable to find pt on bezier curve with dx = %s, dy = %s" \
               % (ai_aiPlus1.unitVect[0], ai_aiPlus1.unitVect[1]))
      else:
         print("There are %s pts that have dx = %s, dy = %s - pick closest one" \
               % (len(ziPts), ai_aiPlus1.unitVect[0], ai_aiPlus1.unitVect[1]))
         for ziPt in ziPts:
            ziEucVect, distEucVectOrigFromStartPt = calcEuclideanProjBtwnLineAndPt(ai_aiPlus1, ziPt)
            ziDist = np.linalg.norm(ziEucVect)
            if ziDist < epsilon and \
               distEucVectOrigFromStartPt < ai_aiPlus1.lineLength and \
               distEucVectOrigFromStartPt > 0:
               if not di or ziDist < di:
                  di = ziDist
            print("ziPt %s has ziDist %s, di = %s, epsilon = %s, dist of euclidean proj pt is %s, len of ai_aiPlus1 = %s" % \
                  (i, ziDist, di, epsilon, distEucVectOrigFromStartPt, ai_aiPlus1.lineLength))

      diEntries.append(di)

   # 2nd condition - loop thru each bezier curve
   #  find the pt zi' in the combined curve that is parallel
   #  to Z0_Z3 of the bezier curve
   #  THEN - get di', the Euclidean dist between:
   #   zi' on the combined curve and
   #   Li of the individual bezier curves passed as input, where
   #  Li is the line calculated in the api givenJointReturnAlphaAndLi
   #  in linesAPI
   # Li is the line as defined by the api givenJointReturnAlphaAndLi in linesAPIs.py
   #  - given a joint made up of 2 connecting lines (biMinus1_ai, ai_bi)
   #    and a unit square drawn centered at ai
   #  - Li is the line parallel to biMinus1_bi that touches the unit square and is
   #    closest to biMinus1_bi
   #  set di' to be positive if zi' is on the same side of Li as ai (which is
   #  the intersect pt of the bezier curve) and negative otherwise
   #
   #  call the ' Prime
   for i, bCurve in enumerate(bezierCurves):
      biMinus1_bi = bCurve.lineZ0_Z3[0]
      if biMinus1_bi.unitVect[0] == 0:
         dy_dx = float('inf')
      else:
         dy_dx = biMinus1_bi.unitVect[1] / biMinus1_bi.unitVect[0]
      ziPtsPrime = combinedCurve.getPtsOnCurveWithDYDXVal(dy_dx)
      diPrime = None
      if len(ziPtsPrime) < 1:
         print("Unable to find pt on bezier curve with dx = %s, dy = %s" \
               % (biMinus1_bi.unitVect[0], biMinus1_bi.unitVect[1]))
      else:
         print("There are %s pts that have dx = %s, dy = %s - pick closest one" \
               % (len(ziPtsPrime), biMinus1_bi.unitVect[0], biMinus1_bi.unitVect[1]))
         for ziPtPrime in ziPtsPrime:
            ziPrimeEucVect, distOfEucOriginFromStartPt = calcEuclideanProjBtwnLineAndPt(bCurve.Li, ziPtPrime)
            ziPrimeDist = np.linalg.norm(ziPrimeEucVect)
            # need to get the sign of the ziDist
            # if zi' is on the same side of Li as ai - it is positive
            #  otherwise it is negative
            #  NOTE: ai is the intersect pt of the guides used to create the bezier curve
            aiEucVect, distOfEucOriginFromStartPt = calcEuclideanProjBtwnLineAndPt(bCurve.Li, bCurve.intersectPt[0])
            if np.dot(aiEucVect, ziPrimeEucVect) < 0:
               ziPrimeDist *= -1
            if ziPrimeDist > -epsilon and \
               distOfEucOriginFromStartPt < biMinus1_bi.lineLength and \
               distOfEucOriginFromStartPt > 0:
               if not diPrime or (math.fabs(ziPrimeDist) < math.fabs(diPrime)):
                  diPrime = ziPrimeDist
            print("ziPtPrime %s has ziPrimeDist %s, di = %s, -epsilon = %s, dist of euclidean proj pt is %s, len of biMinus1_bi = %s" % \
                  (i, ziPrimeDist, diPrime, -epsilon, distOfEucOriginFromStartPt, biMinus1_bi.lineLength))
      diPrimeEntries.append(diPrime)

   penalty = 0
   for i in range(len(diEntries)):
      if not diEntries[i]:
         print("di at idx %s exceeds epsilon %s or its ortho projection is not within line - curves cannot be combined"\
                % (i, epsilon))
         return None
      penalty += (diEntries[i] ** 2)

   for i in range(len(diPrimeEntries)):
      if not diPrimeEntries[i]:
         print("diPrime at idx %s is less than -epsilon %s - curves cannot be combined" \
               % (i, -epsilon))
         return None
      penalty += (diPrimeEntries[i] ** 2)

   print("successfully combined curve with penalty %s" % (penalty))
   combinedCurve.setLineSegsThatMakeUpCurve(bezierCurves)
   return combinedCurve, penalty

# given multiple contiguous bezier curves, look for the combination of curves
# that can produce the minimum # of curves
#  - by combination we mean whether or not we can optimize the curves using
#    the APIs above by combining contiguous curves into 1 curve (which is determined
#    by the cumulative penalty of the combined curves)
#  use dynamic programming - from each pt onwards - store the min # of curves
#  that can be achieve thru combining curves
#
# INPUT: list of contiguous bCurves to look at
#
# OUTPUT: 2 dicts (maps) -
#   1) map of key = index of bCurves , value = min # of bCurves that can be formed
#      BEHIND the curve starting at the index (including the curve identified by index)
#   2) map of key = index of bCurves, value = list of tuples that represent the actual configuration
#      of the curves that make up the min # - the tuples are the starting and ending index of the
#      combined bCurve
#  NOTE: the index is simply the index of the list of the input contigCurves
def getFullMinNumAndConfigsOfContigCurves(contigCurves, epsilon=0.2, verifyCurves=True):
   # first make sure the list of curves are contiguous
   if verifyCurves:
      bezierCurves = checkBCurvesContiguousAndOrient(bezierCurves)

      if not bezierCurves:
         print("Bezier curves not contiguous - cannot optimize")
         return None

      # next check that the convexity of the curves is in the same direction
      if not checkConvexityOfBCurves(bezierCurves):
         print("Bezier curves do not have same convexity")
         return None

   # will use dynamic programming approach to combine list of curves into
   # min number of curves - as we loop thru each curve store the min # of curves
   # that exist before this current curve (NOTE - the count DOES INCLUDE
   #  the curve denoted by the index in the key
   # also - store the configuration of curves
   #  that give this number - that way when we iterate to the next curve can use the answers
   #  stored from the previously iterated curves (refer to as 'behind') to determine the min # of curves from
   #  the curve at the iterator (BEHIND means DOES INCLUDE THE CURVE with index as the key of the entry)
   # - as always in DP - need data structures to store explored data
   idxToMinNumCurves = {-1 : 0, 0 : 1}   # this stores the curve idx to the min # of curves behind this idx
   idxToMinNumCurvesConfig = {-1 : [], 0 : [(0,0)]}  # this stores the curve idx to the curves configurations
                                 # that yield the min number in the prev dict <- can store this
                                 # as list of tuple of the combined curves index
   for i in range(len(contigCurves)):
      listOfCurves = [contigCurves[i]]
      idxCurvesConfig = [i,i]
      minNumCurves = idxToMinNumCurves.get(i-1) + 1
      for j in range(i-1, -1, -1):
         listOfCurves.append(contigCurves[j])
         if optimizeContiguousBezierCurves(listOfCurves, epsilon, verifyCurves=False):
            numCurves = idxToMinNumCurves.get(j-1) + 1
            if numCurves < minNumCurves:
               minNumCurves = numCurves
               idxCurvesConfig[0] = j
      idxToMinNumCurves[i] = minNumCurves
      curveConfigs = copy.deepcopy(idxToMinNumCurvesConfig.get(curveConfigs[0]-1))
      curveConfigs.append(tuple(idxCurvesConfig))
      idxToMinNumCurvesConfig[i] = curveConfigs

   return idxToMinNumCurves, idxToMinNumCurvesConfig

# return the min number of curves that can be formed from this list of
# contiguous bCurves and generate the list of actual curves
# INPUT - list of bCurves
#
# OUTPUT 1) min # of curves that can be formed from this list of bCurves
#        2) the configuration of this min #
def getMinNumCurvesConfig(contigCurves, epsilon=0.2, verifyCurves=True):
   minNumCurvesDict, minNumCurvesConfig = getFullMinNumAndConfigsOfContigCurves(contigCurves, epsilon, verifyCurves)
   # the last element of the dict contains the info for the whole contig curves
   lastElem = len(contigCurves)-1
   minNumCurves = minNumCurvesDict.get(lastElem)
   minCurvesConfigIdxs = minNumCurvesConfig.get(lastElem)
   minCurvesConfig = []
   for minCurveTup in minCurvesConfigIdxs:
      # single curve - no need to combine
      if minCurveTup[0] == minCurveTup[1]:
         minCurvesConfig.append(contigCurves[minCurveTup[0]])
      else:
         combinedCurve, penalty = optimizeContiguousBezierCurves(contigCurves[minCurveTup[0]:minCurveTup[1]+1], epsilon, not verifyCurves)
         minCurvesConfig.append(combinedCurve)

   return minNumCurve, minCurvesConfig

def greedyGetLongestCurve(contigCurves, dir, epsilon=0.2, verifyCurves=True):
   # constrain the dir so that it is unit direction
   # the anchor is the 1st element in the direction specified
   #  if dir = forward (+1), anchor idx = 0 -> 1st elem
   #  if dir = backward (-1), anchor idx = len(contig list) - 1 <- last elem
   inc = dir / abs(dir)
   if inc == -1:
      anchor = len(contigCurves)-1
      endIdx = -1
   else:
      anchor = 0
      endIdx = len(contigCurves)

   startIdx = anchor + inc

   combinedCurve = contigCurves[anchor]
   penalty = None
   idxIterEnd = anchor

   for i in range(startIdx, endIdx, inc):
      # the dir to traverse the contig seg list determines how to slice the list
      # (ie. which elems that must be included in a greedy priority fashion)
      # if dir = -1 (going backwards), the elements closest to the last element are to be included first
      # if dir = 1 (going forwards), the elements closest to the first element are to be included first
      if inc == -1:
         curvesToCheck = contigCurves[startIdx:anchor+1]
      else:
         curvesToCheck = contigCurves[anchor:startIdx+1]

      try:
         combinedCurve, penalty = optimizeContiguousBezierCurves(curvesToCheck, epsilon, verifyCurves)
         idxIterEnd = i
      except Exception as e:
         print("ERROR - %s" % (e))
         break

   if idxIterEnd < anchor:
      idxTuple = (idxIterEnd, anchor)
   else:
      idxTuple = (anchor, idxIterEnd)

   return combinedCurve, penalty, idxTuple

# this class creates an object which characterizes a bezier curve
#
#  the class constructor takes as input a list of tuples as control pts
#
#  the derivCoeffs are the constant terms for dP/dt
#  they are stored in order of decreasing power of t (starting from the
#   highest power for derivative which for cubic bezier is 2)
#  t**2 terms: Q0 - 2Q1 + Q2
#  t**1 terms: -2Q0 + 2Q1
#  t**0 terms: Q0
# where again Qi = n(Pi+1-Pi)
#
#  The data that store the characteristics of the bezier curve such as
#   controlPts, intersectPt, lineZ0_0, lineZ3_0, Q0, Q1, Q2, derivCoeffs
#   are stored as a list of the data pt because we store 2 entries for each
#   piece of data - 1 entry is for the bezier curve and the other entry is for
#   the rotated bezier curve
#   The rotated bezier curve data entries are stored in idx 1
#   store the Q entries as list [Q0, Q1, Q2], where
#      Q0 = n(P1 - P0), Q1 = n(P2 - P1), Q2 = n(P3 - P2)
#       P0, P1, P2, P3 are the control points of the bezier curve
# Li is the line as defined by the api givenJointReturnAlphaAndLi in linesAPIs.py
#  - given a joint made up of 2 connecting lines (biMinus1_ai, ai_bi)
#    and a unit square drawn centered at ai
#  - Li is the line parallel to biMinus1_bi that touches the unit square and is
#    closest to biMinus1_bi
class bezierCls():
   origCoords = 0
   calcRotated = 1
   dataEntries = 2
   def __init__(self, controlPts, Li=None):
      self.controlPts = [[] for i in range(self.dataEntries)]
      self.intersectPt = [None]*self.dataEntries
      self.lineZ0_0 = [None]*self.dataEntries
      self.lineZ3_0 = [None]*self.dataEntries
      self.line0_Z3 = [None]*self.dataEntries
      self.lineZ0_Z3 = [None]*self.dataEntries

      self.hash = None

      self.alpha = None
      self.Li = Li

      self.Q = [[] for i in range(self.dataEntries)]
      self.derivCoeffs = [[] for i in range(self.dataEntries)]

      self.curveLength = None
      self.repLines = []
      self.parentObj = None

      # this is to track that line segs that make up the curve if multiple
      # curves are combined into this one curve - thus this field is only applicable
      # for combined curves
      self.lineSegsThatMakeUpCurve = []

      # this delta_t is used perhaps in the future if more sophisticated
      # calcs determining the optimal delta_t for smooth curve and equal len segs
      # FOR NOW - use arbitary default 0.01 - this splits the bezier curve into 100 line segs
      self.projDelta_t = 0.01
      # this proj len was decided empirically - at this length the bezier curve
      # is still accurately projected - however any longer and the projection
      # no longer matches the curve shape
      self.projLen = 0.2

      if len(controlPts) != 4:
         print("number of control pts is %s - only 4 is supported right now" % \
               (len(controlPts)))
      else:
         self.setControlPts(controlPts)

   # hash API - bezier curve is unique identified by its control pts
   # z0,z1,z2,z3 - treat bCurve with unique pts (z0,z1,z2,z3)1 as congruent with
   # bCurve with pts (z0,z1,z2,z3)2 where z3(2)==z0(1), z2(2)=z1(1), z1(2)=z2(1), z0(2)==z3(1)
   def __hash__(self):
      # first check if dir of curve is from z0->z3 or from z3->z0
      # convention - calc the hash such that the go from z0 -> z3 if
      # z0x < z3x -> if z0x == z3x, check if z0y < z3y otherwise take z3 to be the 1st control pt
      ctlPts = self.controlPts[self.origCoords]
      # check if z0x < z3x
      if ctlPts[0][0] > ctlPts[-1][0]:
         ctlPts.reverse()
      elif ctlPts[0][0] == ctlPts[-1][0] and \
           ctlPts[0][1] > ctlPts[-1][1]:
         ctlPts.reverse()

      return hash(tuple(map(tuple, ctlPts)))

   def getHash(self):
      self.hash = self.__hash__()
      return self.hash

   # 2 bezier curves are considered equal if they have the same control pts
   def __eq__(self, other):
      selfCPts = sorted(self.controlPts[self.origCoords], key=lambda cPt: cPt[0])
      otherCPts = sorted(other.controlPts[self.origCoords], key=lambda cPt: cPt[0])
      if len(otherCPts) != len(selfCPts):
         return False

      for idx in range(len(selfCPts)):
         if not np.array_equal(selfCPts[idx], otherCPts[idx]):
            return False

      return True

   # Common API to get the start / end pt of the curve object
   def getStartPt(self):
      return self.controlPts[self.origCoords][0]

   def getStartPtAsTuple(self):
      return tuple(self.getStartPt())

   def getEndPt(self):
      return self.controlPts[self.origCoords][-1]

   def getEndPtAsTuple(self):
      return tuple(self.getEndPt())

   # API that sets reference to parent if the curve belongs to a parent contig seg
   def setPosInParent(self, parentObj, pos):
      self.parentObj = [parentObj, pos]

   # API to get the segs of the joint that the bCurve is generated from
   #  REMEMBER that the bCurve is generated from control pts that are calculated
   #  from "guides" or "joint arms" that are taken from the contig segs of the raw image
   def getJointThatCurveEstimates(self, coords=origCoords):
      # get the contig segs that are formed by biMinus_ai and ai_bi
      # biMinus1 is the 1st control pt Z0
      # bi is the last control pt Z3
      # ai is the intersect pt between the 2 "guides"
      return (lineCls(self.controlPts[coords][0], self.intersectPt[coords]), \
              lineCls(self.intersectPt[coords], self.controlPts[coords][-1]))

   # API to get the line segs that make up this curve - this is only applicable
   # if this curve is a combined curve (ie. this curve is the combination of
   # multiple bCurves) - if this bCurve was not created from combining bCurves,
   #  return None and log
   def getLineSegsThatMakeUpCurve(self):
      if not self.lineSegsThatMakeUpCurve:
         print("This bCurve is not made up of combined curves")

      return self.lineSegsThatMakeUpCurve

   # API to set the line segs that make up this curve - pass in the list of bCurves
   # that make up this combined curve
   def setLineSegsThatMakeUpCurve(self, listOfBCurves):
      listOfSegs = []
      for bCurve in listOfBCurves:
         listOfSegs.extend(bCurve.getJointThatCurveEstimates())
      if not checkIfLinesSinglyContiguous(listOfSegs):
         print("ERROR - the joints that make up the bCurves are not singly contiguous - %s" % (listOfSegs))
      else:
         self.lineSegsThatMakeUpCurve = listOfSegs

   # Calculate points of projection of bCurve
   # INPUT:
   #   tStart (0 or 1 - this will tell whether it is backwards or forwards projection)
   #   projection length - if not passed in, use the default projection length of the curve
   #   projection delta_t - if not passed in, use the default value of the curve
   #
   #  OUTPUT:
   #   return the projection pts, rotated projection pts, derivatives of projection pts,
   #              the derivative of the rotated projection pts
   def calcProjOfBCurve(self, tStart, projLen=None, projDelta_t=None):
      if not projLen:
         projLen = self.projLen
      if not projDelta_t:
         projDelta_t = self.projDelta_t

      if tStart <= 0:
         projDelta_t *= -1

      projPts = []
      rotatedProjPts = []
      projPtsDerivs = []
      rotatedProjPtsDerivs = []

      t = tStart
      while abs(t) <= abs(tStart) + projLen:
         projPts.append(self.getXYValOfCurve(t, 0))
         rotatedProjPts.append(self.getXYValOfCurve(t, 1))
         projPtsDerivs.append(self.getDYDXgiven_t(t, 0))
         rotatedProjPtsDerivs.append(self.getDYDXgiven_t(t, 1))

         t += projDelta_t

      return projPts, rotatedProjPts, projPtsDerivs, rotatedProjPtsDerivs

   # Calculate both projections info of the bezier curve
   # INPUT:
   #   projection length - if not passed in, use the default projection length of the curve
   #   projection delta_t - if not passed in, use the default value of the curve
   #
   #  OUTPUT:
   #   return the projection pts, rotated projection pts, derivatives of projection pts,
   #              the derivative of the rotated projection pts
   def calcBothProjsOfBCurve(self, projLen=None, projDelta_t=None):
      if not projLen:
         projLen = self.projLen
      if not projDelta_t:
         projDelta_t = self.projDelta_t

      bProj, bRotProj, bProjDerivs, bRotProjDerivs = self.calcProjOfBCurve(0, projLen=projLen, projDelta_t=projDelta_t)
      fProj, fRotProj, fProjDerivs, fRotProjDerivs = self.calcProjOfBCurve(1, projLen=projLen, projDelta_t=projDelta_t)

      return [bProj, fProj], [bRotProj, fRotProj], [bProjDerivs, fProjDerivs], [bRotProjDerivs, fRotProjDerivs]

   # API to generate the representative lines of the bezier curve
   # (the representative lines are lines that estimate the bCurve)
   # INPUT - delta_t -> this value determines the length of the representative lines
   # the greater the delta_t, the coarser the lines (ie. less lines and less accurate rep)
   #
   # # TODO: FIRST PASS - generate 100 lines for now - in the future may do some extra smoothing
   # and / or uniform calculations to calculate optimal # of lines
   def genRepLines(self, delta_t=0.01):
      # check if the delta_t corresponds to length of lines already generated
      if (1 / delta_t) != len(self.repLines):
         self.repLines.clear()
         t = 0
         while t<1:
            lineObj = lineCls(self.getXYValOfCurve(t), self.getXYValOfCurve(t+delta_t))
            lineObj.setPosInParent(self, int(t/delta_t))
            self.repLines.append(lineObj)
            t += delta_t

      return self.repLines

   # API to get the length of the curve
   def getLength(self):
      if self.curveLength:
         retCurveLength = self.curveLength
      else:
         retCurveLength = self.calcCurveLength()
      return retCurveLength

   # API to get the line segment between the 2 params t1, t2
   def getLineSegOfCurve(self, t1, t2):
      if math.abs(t1) <= math.abs(t2):
         startPt = self.getXYValOfCurve(t1)
         endPt = self.getXYValOfCurve(t2)
      else:
         startPt = self.getXYValOfCurve(t2)
         endPt = self.getXYValOfCurve(t1)

      return lineCls(startPt, endPt)

   # API to calculate the length of the curve
   def calcCurveLength(self, delta=0.1):
      self.curveLength = 0.0
      t = 0.0
      while t < 1.0:
         pt1 = self.getXYValOfCurve(t)
         pt2 = self.getXYValOfCurve(t+delta)
         self.curveLength += np.linalg.norm(pt2-pt1)
         t += delta

      return self.curveLength

   # API to get the "last" pt of the Bezier curve (ie. the pt on the curve when t = 1)
   def getLastPt(self, dataIdx=origCoords):
      return self.controlPts[dataIdx][-1]

   # API to get the "first" pt of the Bezier curve (ie. the pt on the curve when t = 0)
   def getFirstPt(self, dataIdx=origCoords):
      return self.controlPts[dataIdx][0]

   # API to get control pts
   def getControlPts(self, dataIdx=origCoords):
      return self.controlPts[dataIdx]

   def setControlPts(self, controlPts, alignWithXAxis=True):
      if self.checkIfInputCtlPointsValid(controlPts):
         self.controlPts[0] = [np.array(pt) for pt in controlPts]
         self.calcBezierGuides()
         self.calculateDerivParams()
         if len(self.controlPts[self.calcRotated]) <= 0 or alignWithXAxis:
            self.rotateBezierCurveToAlignWithXAxis()
         self.calcAlphaAndLiOfBezierCurve()
         # also recalculate the length of the curve
         self.calcCurveLength()
      else:
         print("BezierCls obj error - input control points %s - intersect pt %s" \
                % (self.controlPts[self.origCoords], self.intersectPt[self.origCoords]))
   # API to get the convexity of the bezier curve
   #  the convexity of the curve is determined by the joint line guides
   #  by getting the cross product of biMinus_ai and ai_bi which in this case
   #  is Z0_0 and 0_Z3
   def getConvexity(self, dataIdx=0):
      self.convexity = np.cross(self.lineZ0_0[dataIdx].lineVect, self.line0_Z3[dataIdx].lineVect)
      return self.convexity

   # API to get the direction of the convexity
   #  Positive convexity (which is of positive angle) is counterclockwise (CCW)
   #  Negative convexity (which is of negative angle) is clockwise (CW)
   def getConvexityDir(self, dataIdx=0):
      if self.getConvexity(dataIdx) < 0:
         return -1
      return 1

   # API to get min x, y of the bezier curve
   def getMinXMinY(self):
      minX = None
      minY = None
      # check the control pts
      for i in range(self.dataEntries):
         for pt in self.controlPts[i]:
            if not minX or pt[0] < minX:
               minX = pt[0]
            if not minY or pt[1] < minY:
               minY = pt[1]

      # check the intersect pts
      for pt in self.intersectPt:
         if not minX or pt[0] < minX:
            minX = pt[0]
         if not minY or pt[0] < minY:
            minY = pt[1]

      return minX, minY

   # API to get max x, y of the bezier curve
   def getMaxXMaxY(self):
      maxX = None
      maxY = None
      # check the control pts
      for i in range(self.dataEntries):
         for pt in self.controlPts[i]:
            if not maxX or pt[0] > maxX:
               maxX = pt[0]
            if not maxY or pt[1] > maxY:
               maxY = pt[1]

      # check the intersect pts
      for pt in self.intersectPt:
         if not maxX or pt[0] > maxX:
            maxX = pt[0]
         if not maxY or pt[0] > maxY:
            maxY = pt[1]

      return maxX, maxY


   # API to display bezier curve info
   def displayBezierCurveInfo(self):
      # first display the original params (not rotated so that Z0_Z3 align with x-axis)
      print("Original (unrotated) params:")
      print(" === ControlPts = %s" % (self.controlPts[0]))
      print(" === IntersectPt (pt 0) = %s" % (self.intersectPt[0]))
      Z0_Z1 = self.controlPts[0][1] - self.controlPts[0][0]
      Z1_Z2 = self.controlPts[0][2] - self.controlPts[0][1]
      Z2_Z3 = self.controlPts[0][3] - self.controlPts[0][2]
      print(" === Z0_Z1 vect = %s, uvect = %s, slope = %s, length = %s" \
            % (Z0_Z1, Z0_Z1/np.linalg.norm(Z0_Z1), getSlopeOfVector(Z0_Z1), np.linalg.norm(Z0_Z1)))
      print(" === Z1_Z2 vect = %s, uvect = %s, slope = %s, length = %s" \
            % (Z1_Z2, Z1_Z2/np.linalg.norm(Z1_Z2), getSlopeOfVector(Z1_Z2), np.linalg.norm(Z1_Z2)))
      print(" === Z2_Z3 vect = %s, uvect = %s, slope = %s, length = %s" \
            % (Z2_Z3, Z2_Z3/np.linalg.norm(Z2_Z3), getSlopeOfVector(Z2_Z3), np.linalg.norm(Z2_Z3)))
      print("cos Angle between Z0_Z1 and Z1_Z2 = %s" \
            % (np.dot(Z0_Z1, Z1_Z2)/(np.linalg.norm(Z0_Z1)*np.linalg.norm(Z1_Z2))))
      print("cos Angle between Z1_Z2 and Z2_Z3 = %s" \
            % (np.dot(Z1_Z2, Z2_Z3)/(np.linalg.norm(Z1_Z2)*np.linalg.norm(Z2_Z3))))
      print(" === Line Z0_0:")
      self.lineZ0_0[0].displayLineInfo()
      print(" === Line Z3_0:")
      self.lineZ3_0[0].displayLineInfo()
      print(" === Line Z0_Z3:")
      self.lineZ0_Z3[0].displayLineInfo()
      print(" === Q pts = %s" % (self.Q[0]))
      print(" === derivative coefficients = %s" % (self.derivCoeffs[0]))
      print(" === Li (from potrace joint algo):")
      self.Li.displayLineInfo()
      print("Rotated params (where Z0_Z3 should be aligned with positive x-axis and Z0 = 0,0")
      print(" === ControlPts = %s" % (self.controlPts[1]))
      print(" === IntersectPt (pt 0) = %s" % (self.intersectPt[1]))
      Z0_Z1 = self.controlPts[1][1] - self.controlPts[1][0]
      Z1_Z2 = self.controlPts[1][2] - self.controlPts[1][1]
      Z2_Z3 = self.controlPts[1][3] - self.controlPts[1][2]
      print(" === Z0_Z1 vect = %s, uvect = %s, slope = %s, length = %s" \
            % (Z0_Z1, Z0_Z1/np.linalg.norm(Z0_Z1), getSlopeOfVector(Z0_Z1), np.linalg.norm(Z0_Z1)))
      print(" === Z1_Z2 vect = %s, uvect = %s, slope = %s, length = %s" \
            % (Z1_Z2, Z1_Z2/np.linalg.norm(Z1_Z2), getSlopeOfVector(Z1_Z2), np.linalg.norm(Z1_Z2)))
      print(" === Z2_Z3 vect = %s, uvect = %s, slope = %s, length = %s" \
            % (Z2_Z3, Z2_Z3/np.linalg.norm(Z2_Z3), getSlopeOfVector(Z2_Z3), np.linalg.norm(Z2_Z3)))
      print("cos Angle between Z0_Z1 and Z1_Z2 = %s" \
            % (np.dot(Z0_Z1, Z1_Z2)/(np.linalg.norm(Z0_Z1)*np.linalg.norm(Z1_Z2))))
      print("cos Angle between Z1_Z2 and Z2_Z3 = %s" \
            % (np.dot(Z1_Z2, Z2_Z3)/(np.linalg.norm(Z1_Z2)*np.linalg.norm(Z2_Z3))))
      print(" ===== Line Z0_0:")
      self.lineZ0_0[1].displayLineInfo()
      print(" ===== Line Z3_0:")
      self.lineZ3_0[1].displayLineInfo()
      print(" ===== Line Z0_Z3:")
      self.lineZ0_Z3[1].displayLineInfo()
      print(" ===== Q pts = %s" % (self.Q[1]))
      print(" ===== derivative coefficients = %s" % (self.derivCoeffs[1]))


   # api to calculate the Li of the joint used to characterize this bezier curve
   def calcAlphaAndLiOfBezierCurve(self):
      self.alpha, self.Li = givenJointReturnAlphaAndLiSimpleVer(self.lineZ0_0[0], self.line0_Z3[0])

   # api to check if the input list of points (in order) can serve as control pts
   #  in this particular class of bezier curve (where Z1 lies between Z0_0 and
   #   Z2 lies between Z3_0)

   # need to check if z1 lies between z0 and intersect pt 0 by taking
   #  dot product of Z0Z1 and Z0_0 -> if z1 lies between z0 and 0, the dot
   #  product of the unit vect between the 2 should be == 1 (but there may
   #  be numerical rounding so it may be 0.99... - thus condition is to check if
   #  dot product of the unit vects are > 0.95
   def checkIfInputCtlPointsValid(self, controlPts):
      lineZ0Z1 = lineCls(np.array(controlPts[0]), np.array(controlPts[1]))
      lineZ3Z2 = lineCls(np.array(controlPts[3]), np.array(controlPts[2]))
      try:
         intersectPt =  getIntersectPtBtwn2Lines(lineZ0Z1, lineZ3Z2)
         lineZ0_0 = lineCls(np.array(controlPts[0]), intersectPt)
         lineZ3_0 = lineCls(np.array(controlPts[3]), intersectPt)
         if np.dot(lineZ0Z1.unitVect, lineZ0_0.unitVect) > 0.95 and \
            np.dot(lineZ3Z2.unitVect, lineZ3_0.unitVect) > 0.95:
            return True

         print("Invalid bezier curve - z0=%s z1=%s z2=%s z3=%s 0=%s\
                dot(lineZ0Z1, lineZ0_0) = %s, \
                dot(lineZ3Z2, lineZ3_0) = %s" % \
                (controlPts[0], controlPts[1], controlPts[2], controlPts[3], intersectPt, \
                 np.dot(lineZ0Z1.unitVect, lineZ0_0.unitVect), \
                 np.dot(lineZ3Z2.unitVect, lineZ3_0.unitVect)))

         return False

      except Exception as e:
         print("%s - bezier guide intersect pt = %s" % (e, intersectPt))
         print("lineZ0_0: ")
         lineZ0_0.displayLineInfo()
         print("lineZ3_0: ")
         lineZ3_0.displayLineInfo()
         return False

   # api to calculate the "guides" of the bezier curve
   #   this class of bezier curves are defined by 2 actual lines that are connected
   #   from the image - thus the orientation of the control pts are constrained so that
   #   Z0_Z1 and Z3_Z2 form a triangle
   #    bi-1 (the midpt of the line 1) is Z0
   #    ai (the connection pt between line 1 and line 2) is 0 (the intersection pt
   #       between Z0_Z1 and Z3_Z2)
   #    bi (the midpt of line 2) is Z3
   #  Thus the guides that make up the bezier curve are:
   #   the intersect pt 0
   #   the lines Z0_0 and Z3_0 (NOTE that Z0 and Z3 of the bezier curve do not change)
   #    but Z1 and Z2 can me modified from its initial value as we may pass in a pt
   #    that is not Z1/Z2 but lie on the line of Z0_0/Z3_0 - with this we can create the bezier guides
   #    and adjust Z1/Z2 as needed
   def calcBezierGuides(self, dataIdx=0):
      lineZ0Z1 = lineCls(np.array(self.controlPts[dataIdx][0]), np.array(self.controlPts[dataIdx][1]))
      lineZ3Z2 = lineCls(np.array(self.controlPts[dataIdx][3]), np.array(self.controlPts[dataIdx][2]))
      self.intersectPt[dataIdx] = getIntersectPtBtwn2Lines(lineZ0Z1, lineZ3Z2)
      self.lineZ0_0[dataIdx] = lineCls(np.array(self.controlPts[dataIdx][0]), self.intersectPt[dataIdx])
      self.lineZ3_0[dataIdx] = lineCls(np.array(self.controlPts[dataIdx][3]), self.intersectPt[dataIdx])
      self.line0_Z3[dataIdx] = lineCls(np.array(self.intersectPt[dataIdx]), self.controlPts[dataIdx][3])
      self.lineZ0_Z3[dataIdx] = lineCls(np.array(self.controlPts[dataIdx][0]), np.array(self.controlPts[dataIdx][3]))

   # given t get the (x,y) coord of bezier curve
   def getXYValOfCurve(self, t, dataIdx=0):
      ctlPts = self.controlPts[dataIdx]
      x = (1-t)**3 * ctlPts[0][0] + \
          3 * (1-t)**2 * t * ctlPts[1][0] + \
          3 * (1-t) * t**2 * ctlPts[2][0] + \
          t**3 * ctlPts[3][0]
      y = (1-t)**3 * ctlPts[0][1] + \
          3 * (1-t)**2 * t * ctlPts[1][1] + \
          3 * (1-t) * t**2 * ctlPts[2][1] + \
          t**3 * ctlPts[3][1]
      return np.array([x,y])

   # API to generate the pts on the bezier curve
   #  Take 2 inputs which have default values:
   #   1) the size of increment t (default to 0.1 meaning by default
   #      this API returns 10 points)
   #   2) the dataIdx (whether to return the points of the original curve or the transformed curve)
   def generatePtsOnBezierCurve(self, t=0.1, dataIdx=0):
      retList = []
      ti = 0
      while ti <= 1:
         retList.append(self.getXYValOfCurve(ti, dataIdx))
         ti += t
      return retList

   # calculate bezier parameters such as:
   #  Q0, Q1, Q2 to calculate coefficients for derivates
   #      Q0 = n(P1 - P0), Q1 = n(P2 - P1), Q2 = n(P3 - P2)
   #       P0, P1, P2, P3 are the control points of the bezier curve
   def calculateDerivParams(self, dataIdx=0):
      for idx in range(len(self.controlPts[dataIdx])-1):
         QVal = 3*(self.controlPts[dataIdx][idx+1]-self.controlPts[dataIdx][idx])
         if len(self.Q[dataIdx]) <= idx:
            self.Q[dataIdx].append(QVal)
         else:
            self.Q[dataIdx][idx] = QVal

      self.derivCoeffs[dataIdx] = [self.Q[dataIdx][0]-2*self.Q[dataIdx][1]+self.Q[dataIdx][2], \
                                   -2*self.Q[dataIdx][0]+2*self.Q[dataIdx][1], \
                                   self.Q[dataIdx][0]]

   # this API reverses the bezier curve (ie. z3 now becomes z0,
   #  z2 now becomes z1, z1 now becomes z2, z0 now becomes z3)
   def reverseBezierCurve(self):
      try:
         self.setControlPts(self.controlPts[0].reverse(), True)
      except Exception as e:
         print("failed to reverse list - %s" % (e))

   # This API reflects the set of input points
   #   it also takes in a unit vect that indicates the direction of the line
   #   (the unit vect is assumed to have its start pt at the origin 0,0)
   #  in addition this API takes as input the shift that is applied to all of
   #  the pts before doing the reflection
   def reflectControlPtsAlongLine(self, unitVect, shift):
      # unitVect is the direction of the line that goes thru the origin that the pts
      # are reflected against
      #
      #  The angle is either theta1(-ve) or theta2(+ve) (the angle that unitVect makes with the +ve x-axis)
      #  since the unitVect that describes the line can be +/- 1 * unitVect (where the unitVect is either
      #   -y or +y
      #  and they describe the same line (since the line is infinite and goes thru the origin)
      # theta1 is negative if the unitVect is -ve y and theta2 is positive if the unitVect is +ve y
      # for debugging purposes since we can take the angle that + / -1 * unitVect makes with
      # the +ve x-axis, we will take the unitVect makes an angle whose absolute value is
      # < 90 degrees even if it's negative
      #  need to take dot product of +/- unitVect with unit vect of +ve x-axis (1,0)
      # to get the cos(theta1/theta2) -> we know that if the abs(angle) > 90 degrees, this means that
      #  cos(angle) < 0 -> thus we use the unitVect that gives us cos(angle) > 0
      cosTheta = np.dot(unitVect, np.array([1,0]))
      if cosTheta < 0:
         cosTheta = np.dot(-unitVect, np.array([1,0]))

      theta = math.acos(cosTheta)
      # to get the direction of data, since we know that the angle is between the +ve x-axis, we can tell
      # whether the angle is positive or negative by whether the y-coord of the unitVect is < 0
      if unitVect[1] < 0:
         theta *= -1

      reflectMat = np.array([[cos(2*theta), sin(2*theta)], [sin(2*theta), -cos(2*theta)]])

      newControlPts = []
      for idx, pt in enumerate(self.controlPts[self.origCoords]):
         # shift the pt first
         retPt = pt + shift
         retPt = np.matmul(reflectMat, retPt)
         # reverse the shift
         retPt -= shift
         newControlPts.append(retPt)

      self.setControlPts(newControlPts)

   # this API rotates the control pts of the bezier curve
   # (thereby xforming the bezier curve itself)
   def rotateBezierCurve(self, rotationMat):
      try:
         # first shift so that pt 0 is at (0,0)
         newControlPts = [pt-self.intersectPt[self.origCoords] for pt in self.controlPts[self.origCoords]]
         # perform rotation of each control pt using the rotation matrix
         newControlPts = [np.matmul(rotationMat, pt) for pt in newControlPts]
         # reverse shift of intersectPt
         newControlPts = [pt+self.intersectPt[self.origCoords] for pt in newControlPts]
         self.setControlPts(newControlPts)
         return True

      except Exception as e:
         print("rotateBezierCurve failed - %s" % (e))
         print(traceback.format_exc())
         return False

   # this API shifts the control pts of the bezier curve
   def shiftBezierCurve(self, shiftVect):
      try:
         newControlPts = [pt+shiftVect for pt in self.controlPts[self.origCoords]]
         self.setControlPts(newControlPts)
         return True

      except Exception as e:
         print("shiftBezierCurve failed - %s" % (e))
         print(traceback.format_exc())
         return False

   # this API linearly transforms the control pts of the bezier curve so that:
   #   z0 is (0,0) and z3 is (x3, 0)
   #   all control pts zn have yn >= 0
   def rotateBezierCurveToAlignWithXAxis(self):
      # first shift the control pts so that z0 is (0,0)
      try:
         shiftVect = -self.controlPts[0][0]
         self.controlPts[self.calcRotated] = [pt+shiftVect for pt in self.controlPts[0]]
         # next - rotate the curve so that z3 is (x3, 0)
         z0z3UnitVect = self.controlPts[self.calcRotated][-1]-self.controlPts[self.calcRotated][0]
         z0z3UnitVect /= LA.norm(z0z3UnitVect)

         # the direction to orient is (1,0) since we want z3 to be (x3,0) and want
         #  all x from z0, z1, ..., zn >= 0
         dirToOrient = np.array([1,0])
         cosTheta = np.dot(z0z3UnitVect, dirToOrient)
         angleDir = np.cross(z0z3UnitVect, dirToOrient)
         angleDir /= abs(angleDir)

         sinTheta = math.sqrt(1-cosTheta**2)*angleDir

         # now - perform rotation of each control pt using this matrix
         # cosA    -sinA
         # sinA    cosA
         rotationMat = np.array([[cosTheta, -sinTheta], [sinTheta, cosTheta]])

         self.controlPts[self.calcRotated] = [np.matmul(rotationMat, pt) for pt in self.controlPts[self.calcRotated]]

         # with this rotation matmul there are cases where the pt Z3
         # y-value is not exactly 0 - can be 6.41930466e-15
         if int(self.controlPts[self.calcRotated][-1][1]) == 0:
            self.controlPts[self.calcRotated][-1][1] = 0.0
         else:
            print("Rotation error - y-coord of last control pt is %s - should be 0" \
                  % (self.controlPts[self.calcRotated][-1][1]))

         # now - check if the y-values of the control pts z1 z2 have y < 0 - if so,
         #   flip them so that y > 0 by multiplying the y coords by -1
         yFlip = 1
         for idx, pt in enumerate(self.controlPts[self.calcRotated]):
            if pt[1] < 0:
               yFlip = -1

            if pt[1] > 0 and yFlip < 0:
               print("Error - z%s y-coord > 0 BUT there are control pts with \
                      y-coord < 0 - ctl pts = %s" % (idx, self.controlPts[self.calcRotated]))


         if yFlip < 0:
            self.controlPts[self.calcRotated] = [pt * np.array([1, yFlip]) \
                                            for pt in self.controlPts[self.calcRotated]]

         # after the transformation need to calculate the bezier guides
         self.calcBezierGuides(self.calcRotated)
         self.calculateDerivParams(self.calcRotated)
         print("rotateBezierCurveToAlignWithXAxis - new control pts are %s" % (self.controlPts[self.calcRotated]))
         return True

      except Exception as e:
         print("rotateBezierCurveToAlignWithXAxis failed - %s" % (e))
         print(traceback.format_exc())
         return False


   # calculate the area under the bezier curve analytically in closed form
   def calculateAreaOfCubicBezier(self):
      try:
         if len(self.controlPts[self.calcRotated]) <= 0:
            self.rotateBezierCurveToAlignWithXAxis()
         z0, z1, z2, z3 = self.controlPts[self.calcRotated]
      except Exception as e:
         print("cannot calculate area of bezier curve - %s" % (e))
         return None

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

      print("area under bezier curve with orig control pts %s, rotated control pts %s = %s" % \
            (self.controlPts[self.origCoords], self.controlPts[self.calcRotated], area))

      return abs(area)

   # calculate the area under bezier curve numerically using Riemann sum
   def numericalAreaOfCubicBezier(self):
      try:
         z0, z1, z2, z3 = self.controlPts[self.origCoords]
      except Exception as e:
         print("cannot calculate area of bezier curve - %s" % (e))
         return None

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

   # given the values dx, dy - find the pts x,y that have the derivative dy/dx
   #
   # the formula for calculating the derivative of a bezier curve is
   #  C(t) = SUM(i=0 to n-1)Bn-1,i(t)Qi
   #    where Bn,i(t) = n!/(i!(n-i)!) * (t**i * (1-t)**n-i)
   #    and Qi = n(Pi+1 - Pi) where P(subscript) is the ctl point
   #    and n is the degree of the bezier curve - cubic bezier curves have n = 3
   #
   #  can see that the derivative of cubic bezier curve yields quadratic equation
   #   with some algebra can see that the coefficients for the t**0, t**1, t**2 are:
   #
   #  t**2 terms: Q0 - 2Q1 + Q2 = A
   #  t**1 terms: -2Q0 + 2Q1 = B
   #  t**0 terms: Qo = C
   # REMINDER - derivCoeffs store [A, B, C] in that order

   # API that, given t gives the dy/dx at that given pt parametrized by t
   def getDYDXgiven_t(self, t, dataIdx=0):
      dx_dt = self.derivCoeffs[dataIdx][0][0] * t**2 + \
              self.derivCoeffs[dataIdx][1][0] * t + \
              self.derivCoeffs[dataIdx][2][0]

      dy_dt = self.derivCoeffs[dataIdx][0][1] * t**2 + \
              self.derivCoeffs[dataIdx][1][1] * t + \
              self.derivCoeffs[dataIdx][2][1]

      try:
         if dx_dt == 0:
            dy_dx = float("inf")
         else:
            dy_dx = dy_dt / dx_dt
      except ZeroDivisionError:
         print("dx_dt = 0 - return infinity")
         dy_dx = float("inf")
      except Exception as e:
         print("Failed to calculate dy_dx - %s" % (e))
         print(traceback.format_exc())
         dy_dx = None

      return dy_dx

   # API that returns the pt x, y
   #  given the values dx, dy - find the pts x,y that have the derivative dy/dx
   #
   def getPtsOnCurveWithDYDXVal(self, dy_dx, dataIdx=0):
      derivCoeffs = self.derivCoeffs[dataIdx]

      ptsWithDeriv = []

      if dy_dx < float("inf"):
         deriv = dy_dx
         # because dx != 0 - proceed with
         # dy = Ayt**2 + Byt**1 + Cyt**0
         # --  --------------------------
         # dx = Axt**2 + Bxt**1 + Cxt**0
         # This means that the quadratic eqn to calculate t (after applying algebra)
         #  C1t**2 + C2t**1 + C3t**0 = 0, where
         # C1 = Ay-(dy/dx)Ax, C2 = By-(dy/dx)Bx, C3 = Cy-(dy/dx)Cx
         C1 = derivCoeffs[0][1]-deriv*derivCoeffs[0][0]
         C2 = derivCoeffs[1][1]-deriv*derivCoeffs[1][0]
         C3 = derivCoeffs[2][1]-deriv*derivCoeffs[2][0]
         for root in np.roots([C1,C2,C3]):
            if not np.iscomplexobj(root) and root >= 0.0 and root <= 1.01:
               ptsWithDeriv.append(self.getXYValOfCurve(root, dataIdx))
      else:
         # dx = 0 -> this means only need to solve
         # quadratic eqn for dx/dt = 0
         # the quadratic equation for Axt**2 + Bxt**1 + Cxt**0 is
         #   (-B +/- sqrt(B**2-4*A*C)) / (2*A)
         # meaning that there are 2 solutions - only take solutions that are
         # are between 0 and 1 since that is the range of the parametric equation
         # of Bezier curves
         A = derivCoeffs[0][0]; B = derivCoeffs[1][0]; C = derivCoeffs[2][0]
         for root in np.roots([A,B,C]):
            if not np.iscomplexobj(root) and root >= 0.0 and root <= 1.01:
               # calculate the (x,y) value given t
               ptsWithDeriv.append(self.getXYValOfCurve(root, dataIdx))

      return ptsWithDeriv

   # given the area under the Bezier curve -> determine its z1 / z2 control pts
   #  NOTE: here we follow the potrace algo for determining the z1/z2 control pts
   #  by setting z1 = ALPHA * Z0_0 + Z0 and z2 = ALPHA * Z3_0 + Z3
   #  ALPHA is determined by the area under the cubic bezier curve of the form
   #  in the description above
   #   If the cubic bezier curve is of the description above - then the "unit" bezier
   #   curve (where Z0 = (-1,0), 0 = (0,1), Z3 = (1,0) and the control pts Z1 and Z2
   #   are determined by ALPHA where Z1 = (-1+ALPHA, ALPHA) and Z2 = (1-ALPHA, ALPHA)
   #  (NOTE normally the ALPHA and BETA along Z0_0 and Z3_0 can be different but we add
   #  constraint so that the ALPHA is equal on BOTH sides)
   #   Given this constraint - the area under this curve is 3/10 * (4 - (2-ALPHA)*(2-ALPHA))
   #  The area under the bezier curve scales by the length of Z0Z3 and the height of 0 linearly
   #  much like a triangle (NOTE - like a triangle, a shift in 0 such that the length of Z0_0
   #  != length of Z3_0 does NOT change the area as long as the y coord of 0 remains unchanged
   #  THUS - if we orient the bezier curve such that Z0_Z3 lie on the x-axis - given the area
   #  under the desired bezier curve, we can calculate the ALPHA and by extension Z1,Z2 by doing the following:
   #   Given the desired area under the Bezier curve - scale down this area by factor of:
   #    - (Z0_Z3) / 2 <- this is the scaling of the Z0_Z3 of our curve to the unit curve where Z0 = -1,0 and Z1 = 1,0
   #    - y-coord (height) of 0 <- this is the scaling of the height of the 0 of our curve to the unit curve where 0 is (0,1)
   #  After scaling down the area - then can apply 3/10 * (4 - (2-ALPHA)*(2-ALPHA)) to calculate ALPHA
   #  and as long as 0 < ALPHA < 1 -> can calculate ALPHA - which is then used to calculate Z1, Z2
   #  Z1 = (len(Z0_0) * ALPHA) * unitVectOfZ0_0 + Z0
   #  Z2 = (len(Z3_0) * ALPHA) * unitVectOfZ3_0 + Z3
   def givenAreaCalculateZ1Z2(self, area):
      try:
         # first need to rotate the bezier curve to align with xaxis
         self.rotateBezierCurveToAlignWithXAxis()
         # get the length from Z0 to Z3 - since the rotated control pts Z0 and Z3
         #  should be (0,0) and (X3,0), respectively, simply need the x-coord of Z3 for length
         baseLength = self.controlPts[self.calcRotated][-1][0]
         # get the height of the intersectPt 0 - reminder the intersect pt
         # is the pt of intersection of the line segment Z0_Z1 and Z3_Z2
         baseHeight = self.intersectPt[self.calcRotated][1]
         # To scale the area such that the equation below can be used - need to apply the ratio
         #  A / unitCurveA = (L * W of joint) / 2 (since the unit curve area has width 2)
         unitCurveArea = 2 * area / (baseLength * baseHeight)
         print("rotated curve - baseLength = %s, baseHeight = %s, area desired = %s" % (baseLength, baseHeight, area))

         # given the equation of the area under the unit bezier curve is
         #  A = 3/10 * (4-(2-ALPHA)*(2-ALPHA))
         # with some algebra - the quadratic equation for ALPHA is
         #  ALPHA**2 - 4*ALPHA**1 + (10/3)*A = 0
         alphaSolns = ((4 + math.sqrt(4**2 - 4*1*(10/3)*unitCurveArea)) / (2*1),
                       (4 - math.sqrt(4**2 - 4*1*(10/3)*unitCurveArea)) / (2*1))

         alpha = None
         for alphaVal in alphaSolns:
            if alphaVal > 0 and alphaVal < 1:
               if alpha:
                  print("2 unique solutions of alpha that are between 0 and 1 \
                         %s, %s - error" % (alpha, alphaVal))
                  return False
               else:
                  alpha = alphaVal

         # given alpha is now calculated - calculate the correct Z1,Z2 in both
         # the original and the rotated frame
         pt0 = self.getFirstPt()
         pt1 = self.lineZ0_0[0].termPt1 + alpha*self.lineZ0_0[0].lineLength*self.lineZ0_0[0].unitVect
         pt2 = self.lineZ3_0[0].termPt1 + alpha*self.lineZ3_0[0].lineLength*self.lineZ3_0[0].unitVect
         pt3 = self.getLastPt()
         self.setControlPts([pt0,pt1, pt2, pt3], True)
        # self.controlPts[self.calcRotated][1] = self.lineZ0_0[self.calcRotated].termPt1 + \
        #                                   alpha*self.lineZ0_0[self.calcRotated].lineLength*self.lineZ0_0[self.calcRotated].unitVect
        # self.controlPts[self.calcRotated][2] = self.lineZ3_0[self.calcRotated].termPt1 + \
        #                                   alpha*self.lineZ3_0[self.calcRotated].lineLength*self.lineZ3_0[self.calcRotated].unitVect

         return True
      except Exception as e:
         print("Failed to calculate actual Z1, Z2 from curve area - %s" % (e))
         print(traceback.format_exc())
         return False

   # API to draw the bezier curve
   #   the dataIdx specifies whether or not to draw the bezier curve:
   #     dataIdx = 0 -> draw bCurve in original orientation
   #     dataIdx = 1 -> draw bCurve in orientation where Z0_Z3 aligns with x-axis
   def drawBezierCurve(self, imgHeight, imgWidth, imgName, drawDeriv=False, delta_t=0.1, color=(0,255,0)):
      line_thickness = 2
      imgOutBCurve = np.ones((int(imgHeight), int(imgWidth), 3), dtype=np.uint8)
      t = 0
      pts = [[] for i in range(self.dataEntries)]
      while t < 1:
         pts[0].append(self.getXYValOfCurve(t, 0))
         pts[1].append(self.getXYValOfCurve(t, 1))
         t += delta_t
      origPts = np.array([[pt[0], pt[1]] for pt in pts[0]]).reshape((-1,1,2)).astype(np.int32)
      derivPts = np.array([[pt[0], pt[1]] for pt in pts[1]]).reshape((-1,1,2)).astype(np.int32)
      print(origPts)
      cv.drawContours(imgOutBCurve, [origPts], 0, color, line_thickness)

      if drawDeriv:
         cv.drawContours(imgOutBCurve, [derivPts], 0, color, line_thickness)

      cv.imwrite(imgName, imgOutBCurve)


   # use matplotlib to plot the bezier curve
   def plotBezierCurve(self, imgName, delta_t=0.1, drawRotated=False, drawDeriv=False, unitDU=1, projectLen=None, projectedDelta_t=0.01):
      t = 0
      pts = [[] for i in range(self.dataEntries)]
      ptsDerivs = [[] for i in range(self.dataEntries)]
      while t < 1:
         pts[0].append(self.getXYValOfCurve(t, 0))
         pts[1].append(self.getXYValOfCurve(t, 1))
         ptsDerivs[0].append(self.getDYDXgiven_t(t, 0))
         ptsDerivs[1].append(self.getDYDXgiven_t(t, 1))
         t += delta_t
      origPtsX = [pt[0] for pt in pts[0]]
      origPtsY = [pt[1] for pt in pts[0]]

      rotatedPtsX = [pt[0] for pt in pts[1]]
      rotatedPtsY = [pt[1] for pt in pts[1]]

      if projectLen:
         # the projection x and y points are list of 2 lists as there are 2 projections
         #  1) t < 0
         #  2) t > 1
         projPts, rotatedProjPts, projPtsDerivs, rotatedProjPtsDerivs = self.calcBothProjsOfBCurve()

         # extract the ptsX / ptsY
         projPtsX = [[pt[0] for pt in projPts[0]], [pt[0] for pt in projPts[1]]]
         projPtsY = [[pt[1] for pt in projPts[0]], [pt[1] for pt in projPts[1]]]
         rotatedProjPtsX = [[pt[0] for pt in rotatedProjPts[0]], [pt[0] for pt in rotatedProjPts[1]]]
         rotatedProjPtsY = [[pt[1] for pt in rotatedProjPts[0]],[pt[1] for pt in rotatedProjPts[1]]]
         projTangentX = [[], []]
         projTangentY = [[], []]
         rotatedProjTangentX = [[], []]
         rotatedProjTangentY = [[], []]

      tangentX = [[] for i in range(self.dataEntries)]
      tangentY = [[] for i in range(self.dataEntries)]
      # calculate the tangent lines if drawDeriv
      if drawDeriv:
         # for each pts deriv get its tangent by taking each pt as calculated in the
         #  pts list and adding 2 new pts to form a tangent line
         #  The 2 new pts are pt +/- du, where du is a vector where its
         #  x-component is 1 and y-component is therefore dy/dx
         #   however - if dy/dx is float(inf) because dx = 0, then du = (0,1)
         if drawRotated:
            rangeEnd = self.dataEntries
         else:
            rangeEnd = self.calcRotated
         for i in range(rangeEnd):
            for idx in range(len(ptsDerivs[i])):
               if ptsDerivs[i][idx] == float("inf"):
                  DU = np.array([0,unitDU])
               else:
                  DU = np.array([unitDU, unitDU*ptsDerivs[i][idx]])
               ptMinusDU = pts[i][idx] - DU
               ptPlusDU = pts[i][idx] + DU
               tangentX[i].append([ptMinusDU[0], pts[i][idx][0], ptPlusDU[0]])
               tangentY[i].append([ptMinusDU[1], pts[i][idx][1], ptPlusDU[1]])

         if projectLen:
            for projIdx in range(2):
               for idx in range(len(projPtsDerivs[projIdx])):
                  if projPtsDerivs[projIdx][idx] == float("inf"):
                     DU = np.array([0,unitDU])
                  else:
                     DU = np.array([unitDU, unitDU*projPtsDerivs[projIdx][idx]])
                  ptMinusDU = projPts[projIdx][idx] - DU
                  ptPlusDU = projPts[projIdx][idx] + DU
                  projTangentX[projIdx].append([ptMinusDU[0], projPts[projIdx][idx][0], ptPlusDU[0]])
                  projTangentY[projIdx].append([ptMinusDU[1], projPts[projIdx][idx][1], ptPlusDU[1]])

            if drawRotated:
               for projIdx in range(2):
                  for idx in range(len(rotatedProjPtsDerivs[projIdx])):
                     if rotatedProjPtsDerivs[projIdx][idx] == float("inf"):
                        DU = np.array([0,unitDU])
                     else:
                        DU = np.array([unitDU, unitDU*rotatedProjPtsDerivs[projIdx][idx]])
                     ptMinusDU = rotatedProjPts[projIdx][idx] - DU
                     ptPlusDU = rotatedProjPts[projIdx][idx] + DU
                     rotatedProjTangentX[projIdx].append([ptMinusDU[0], rotatedProjPts[projIdx][idx][0], ptPlusDU[0]])
                     rotatedProjTangentY[projIdx].append([ptMinusDU[1], rotatedProjPts[projIdx][idx][1], ptPlusDU[1]])

      fig = plt.figure()
      ax = plt.subplot(221)
      ax.plot(origPtsX, origPtsY)
      # now plot the guides
      ax.plot([self.lineZ0_0[0].termPt1[0], self.lineZ0_0[0].termPt2[0],\
               self.lineZ3_0[0].termPt1[0]], \
              [self.lineZ0_0[0].termPt1[1], self.lineZ0_0[0].termPt2[1],\
               self.lineZ3_0[0].termPt1[1]], color='brown')

      # draw the projections - for now draw them explicitly separately
      #  (so that different colors can be used)
      if projectLen:
         # plot the 1st projection (t < 0)
         ax.plot(projPtsX[0], projPtsY[0], color='darkviolet')
         # plot the 2nd projection (t > 1)
         ax.plot(projPtsX[1], projPtsY[1], color='magenta')

      # if drawDeriv is TRUE need to draw tangents
      if drawDeriv:
         for i in range(len(tangentX[0])):
            ax.plot(tangentX[0][i], tangentY[0][i], color='green')
         if project:
            # draw the derivs of the 1st projection
            for i in range(len(projTangentX[0])):
               ax.plot(projTangentX[0][i], projTangentY[0][i], color='lawngreen')
            # draw the derives of the 2nd projection
            for i in range(len(projTangentX[1])):
               ax.plot(projTangentX[1][i], projTangentY[1][i], color='lawngreen')

      if drawRotated:
         ax2 = plt.subplot(222)
         ax2.plot(rotatedPtsX, rotatedPtsY)
         # now plot the guides
         ax2.plot([self.lineZ0_0[self.calcRotated].termPt1[0], self.lineZ0_0[self.calcRotated].termPt2[0],\
                   self.lineZ3_0[self.calcRotated].termPt1[0]], \
                  [self.lineZ0_0[self.calcRotated].termPt1[1], self.lineZ0_0[self.calcRotated].termPt2[1],\
                   self.lineZ3_0[self.calcRotated].termPt1[1]], color='brown')
         if projectLen:
            # plot the 1st projection (t < 0)
            ax2.plot(rotatedProjPtsX[0], rotatedProjPtsY[0], color='darkviolet')
            # plot the 2nd projection (t > 1)
            ax2.plot(rotatedProjPtsX[1], rotatedProjPtsY[1], color='magenta')

         if drawDeriv:
            for i in range(len(tangentX[1])):
               ax2.plot(tangentX[1][i], tangentY[1][i], color='green')
            if project:
               # draw the derivs of the 1st projection
               for i in range(len(rotatedProjTangentX[0])):
                  ax2.plot(rotatedProjTangentX[0][i], rotatedProjTangentY[0][i], color='lawngreen')
               # draw the derives of the 2nd projection
               for i in range(len(rotatedProjTangentX[1])):
                  ax2.plot(rotatedProjTangentX[1][i], rotatedProjTangentY[1][i], color='lawngreen')

      plt.savefig(imgName)


if __name__ == '__main__':
   results = parser.parse_args()
   bracketRe = re.compile("\((.+?)\)")

   a = []
   for xy in bracketRe.findall(results.tuples):
      coords = xy.split(',')
      a.append((float(coords[0]), float(coords[1])))

   z0, z1, z2, z3 = a
   print("z0 is %s z1 is %s z2 is %s z3 is %s" % (z0, z1, z2, z3))
   cubicBCurve = bezierCls(a)
