#!/usr/bin/python

# this file serves as a library for general utilities
# such as performing calculation across lines, contig segs, bezier curves etc.

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

__prog__ = os.path.basename(__file__)

# get the center of mass of the input list of pts
#
#  INPUT: list of np array pts
#
#  OUTPUT: center of mass as calculated by eqn
#    sum[i to N](m * (ri - R)) = 0
#   where the mass of each pt = 1 since they are weighted equally
def getCenterOfMassFromListOfPts(listOfPts):
   return (sum(listOfPts) / len(listOfPts))

# API to check if the pt is between the span of the line
def checkIfPtInLineSpan(pt, line):
   startPt = line.getStartPt()
   endPt = line.getEndPt()

   return ((refPt[0] >= startPt[0] and refPt[0] <= endPt[0]) or \
           (refPt[0] <= startPt[0] and refPt[0] >= endPt[0])) or \
          ((refPt[1] >= startPt[1] and refPt[1] <= endPt[1]) or \
           (refPt[1] <= startPt[1] and refPt[1] >= endPt[1]))

# this API takes as input a pt and a line
# if the pt is between the line then get the min dist between the pt and its
# complement on the line
#
#  its complement is defined as having the same x- or y- coord on the line as
#  the pt - this means that there are 2 possible candidates - if there are 2
#  candidates, return the one that is min dist to the refPt
def getComplPtOnLine(refPt, line):
   # if the x or y-value of the refPt falls between these 2 particular sec pts
   # we know that this refPt should fall
   startPt = line.getStartPt()
   endPt = line.getEndPt()

   if checkIfPtInLineSpan(pt, line):
      # interpolate the gap between 2 pts as a line
      U = endPt - startPt
      # try using the: 1) x- 2) y-coord of the refPt as reference on the sec line
      # and use the secPt candidate that is min dist from refPt
      try:
         B = (refPt[1] - startPt[1]) / U[1]
         secPtRefY = np.array([startPt[0] + B * U[0], refPt[1]])
      except:
         print("{} - FAILED to get secPt from y-cord of refPt {}".format(__name__, refPt[1]))
         secPtRefY = None

      try:
         # use the x-coord of refPt to calculate the y pt on secPtsList
         # target pt = start pt + B * U where,
         #  U = end pt - start pt
         B = (refPt[0] - startPt[0]) / U[0]
         secPtRefX = np.array([refPt[0], startPt[1] + B * U[1]])
      except:
         print("{} - FAILED to get secPt from x-coord of refPt {}".format(__name__, refPt[0]))
         secPtRefX = None

      if not secPtRefX:
         secPt = secPtRefY
      elif not secPtRefY:
         secPt = secPtRefX
      elif np.linalg.norm(secPtRefX-refPt) < np.linalg.norm(secPtRefY-refPt):
         secPt = secPtRefX
      else:
         secPt = secPtRefY

   return secPt

# compare the set of pts from the sec contig seg of arbitrary form
# to the list of pts from the ref contig seg, whose list of
# pts will be used as base of comparison
# given the x-val of the pt from the ref list find the y-val with this corresponding
# x-val from the sec list and get the diff vect from the 2 pts
#
# INPUT - refPtsList, secPtsList
# OUTPUT - dict - key = pt on the refPtsList
#                 val = diff vect between the refPt and the secPtsList with that same
#                       x-val (interpolate if x-val falls between the pts on the secPtsList)
def checkSecContigSegsPtsOverlapWithRef(refPtsList, secPtsList):
   ptToDiffVect = {}
   for refPt in refPtsList:
      # given the x-val of the refPt - find the x,y pt on the secPtsList that
      # matches the x-val of the refPt
      # NOTE: for any x-pts that do not EXACTLY match any of the pts on the
      # secPtsList (which is very likely) - interpolate the pts in the secPtsList
      # so that assume the gap between 2 pts in secPtsList is a straight line
      # and get the pts value this way
      minDiffVect = None
      for i in range(len(secPtsList)-1):
         # if the x or y-value of the refPt falls between these 2 particular sec pts
         # we know that this refPt should fall
         line = lineCls(secPtsList[i], secPtsList[i+1])
         secPt = getComplPtOnLine(refPt, line)

         if secPt:
            # this is to handle curve where there are 2 pts with the same x-val
            # but different y-val
            if not minDiffVect or \
               (np.linalg.norm(secPt - refPt) < np.linalg.norm(minDiffVect)):
               minDiffVect = secPt - refPt

      # this handles the case where the ref contig seg is longer than the sec contig seg
      # in which case the minDiffVect will not exist
      if minDiffVect:
         ptToDiffVect[refPt] = minDiffVect

   return minDiffVect

# this API performs the comparison between 2 contig segs list of pts
# and checks how much of the REF CONTIG SEG LIST and SEC CONTIG SEG LIST
# overlap - return the overlap portion of the REF
# The overlap is defined if portion of the REF SEG is covered by the SEC SEG
#
# Takes as input:
#  1) reference line pts list (line pts that return remain seg)
#  2) sec line pts list (line pts to compare with the ref line seg)
def getOverlapPortion(refLinePtsList, secLinePtsList):
   allOverlapSegs = []
   overlapSegPts = []
   for refIdx, refPt in enumerate(refLinePtsList):
      minDist = None
      minDistComplPt = None
      for i in range(len(secLinePtsList)-1):
         # this is the 1st pt that lies in between a section of the sec line pts list
         if checkIfPtInLineSpan(refPt, lineCls(secLinePtsList[i], secLinePtsList[i+1])):
            # if the refPt is the 1st pt this means that the entire portion of the ref line
            # is within that secLinePtsList
            # OR if this pt is a part of the contiguous
            if refIdx == 0 or overlapSegPts:
               overLapSegPts.append(refPt)
               break
            # only check the complement pt if we are NOT in the middle of an overlapping seg
            # of the REF seg - this is determined by if the overlapSegStartAndEnd has one element
            # in it (the start pt of the overlap found)
            if not overlapSegPts:
               # if the ref pt is in between the line composed of secLinePtsList[i] and
               # secLinePtsList[i+1] - get the projected pt on the refLine
               complPt = getComplPtOnLine(secLinePtsList[i], lineCls(refLinePtsList[refIdx-1], refPt))
               # given - the complement point of the start pt of the sec line that the ref pt lies between
               # on the ref line - get the dist between the pt before the current refPt
               complPtDist = np.linalg.norm(complPt-refLinePtsList[refIdx-1])
               if not minDist or complPtDist < minDist:
                  minDist = complPtDist
                  minDistComplPt = complPt
      # since we are moving along the refLine in order - once the refPt along the
      # refLine is found to overlap - break
      if minRefFound:
         break

   # now return the section that is outstanding
   if startDir:
      retList = refLinePtsList[:endIdx+1]
      if endPt:
         retList.append(endPt)
   else:
      refLinePtsList.reverse()
      secLinePtsList.reverse()
      # since the endIdx is counted from the end - need to offset the idx by
      # the max idx of the list (which is len - 1) to get the equivalent idx
      # from the beginning
      retList = refLinePtsList[len-1-endIdx:]
      if endPt:
         retList[:0] = endPt

   return retList

# this API compares the 2 contig segs list of pts
#  will do the following comparisons
#  1) ref contig seg -> sec contig seg
#  2) sec contig seg -> ref contig (seg)
# The penalty criteria shall be:
#  1) check to make sure the magnitude of each diff vect does not exceed a threshold value
#  2) The sum of the diff vects - resultant vect -> has magnitude less than threshold value
#    (may not be the same threshold value as 1)
# NOTE - condition is that the secPtsList must completely span the pts on the refPtsList (ie.
#  the minX(secPtsList) < minX(refPtsList) and maxX(secPtsList) > maxX(refPtsList)
#
#  INPUT: - list of pts from: 1) REF CONTIG SEG 2) SEC CONTIG SEG
#         - min diff vect threshold (threshold which returns false if ANY min diff vect magnitude
#           exceeds value)
#         - min total diff vect threshold - threshold which returns false if the sum of the min diff
#           vects exceeds the value
# OUTPUT: - boolean (TRUE if 2 contig segs points overlap)
#           the dict of start pt of REF CONTIG SEG to diff vect of corresponding pt (same x- or same y-)
#
def checkIf2ContigSegsPtsOverlap(refListOfPts, secListOfPts, minDiffVectThresh, minDiffTotalThresh):
   refToSecDiff = checkSecContigSegsPtsOverlapWithRef(refListOfPts, secListOfPts)
   secToRefDiff = checkSecContigSegsPtsOverlapWithRef(secListOfPts, refListOfPts)
   # first convert the secToRefDiff so that the key is the ref pt
   #  to get the equivalent pt on the REF CONTIG SEG - take the pt on the SEC CONTIG SEG
   #  and subtract the diffVect. Then to get the diffVect pointing from REF SEC to SEC CONTIG SEG
   #  multiply it by -1
   for pt, diffVect in secToRefDiff.items():
      refToSecDiff.update({pt-diffVect : -diffVect})

   ret = True
   diffVectsSum = np.array([0,0])
   for pt, diff in refToSecDiff.items():
      diffVectsSum += diff
      if np.linalg.norm(diff) > minDiffVectThresh:
         print("diff vect %s of value %s has magnitude %s which exceeds minDiffVectThresh %s" % \
              (idx, diff, np.linalg.norm(diff), minDiffVectThresh))
         ret = False

   if np.linalg.norm(diffVectsSum) > minDiffTotalThresh:
      print("summed diff vector %s with magnitude %s exceeds minDiffTotalThresh %s" % \
            (diffVectsSum, np.linalg.norm(diffVectsSum), minDiffTotalThresh))
      ret = False

   return ret, diffVects

# API to check the relative overlap of 2 contig segs, meaning that the SEC CONTIG SEG
# is shifted so that its center of mass is aligned with the center of mass of the
# REF CONTIG SEG and then check the degree of overlap
def checkRelativeOverlapOf2ContigSegs(refListOfPts, secListOfPts, minDiffVectThresh, minDiffTotalThresh):
   refSegCOM = getCenterOfMassFromListOfPts(refListOfPts)
   secSegCOM = getCenterOfMassFromListOfPts(secListOfPts)
   COMDiff = refSegCOM - secSegCOM
   secContigShiftedPts = [secPt + COMDiff for secPt in secListOfPts]

   return(checkIf2ContigSegsPtsOverlap(refListOfPts, secContigShiftedPts, minDiffVectThresh, minDiffTotalThresh))
