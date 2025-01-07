import numpy as np
import math
import sys
import os
import traceback
from matplotlib import pyplot as plt
from linesAPIs import *
import re

indexRe = re.compile('\(([0-9]+)\)')

###############################################
### This section contains APIs that read text
### files and constructs whatever data
### structure that file stores
###############################################

# This API processes text file with joint info in format:
# -
# biMinus1 : 0,-44
# ai : 0,0
# bi : -17,35
# shift : -55,38 <--- shift to be applied
# theta : 40 <--- rotation to be applied
# alpha : 0 <= a <= 1 <--- alpha parameter used to generate the control pts
#        for the bezier curve that uses the joint as guide
# phi : 30 <--- angle between the line ai_bi and the positive y-axis (0,1)
#         if the joint is oriented such that biMinus_ai is along the negative y-axis
#         and ai is located at (0,0)
# -
# can store multiple joints separated by '-' -> each joint is bookended
# by '-'
# NOTE: some parameters may not be used at all such as alpha, depending on what
#       we are testing. This is simply to maximize the ablity of testing bezier curve
#       generation

def getJointsDataFromTestFile(inFile):
   joints = []
   jointParams = {}
   with open(inFile) as jointsFile:
      for lineIdx, line in enumerate(jointsFile):
         try:
            if line.rstrip() == '-':
               if jointParams:
                  joints.append(jointParams)
               jointParams = {}
            else:
               lineSplit = [elem.lstrip().rstrip() for elem in line.split(':')]
               print("%s" % (lineSplit))
               # the angle of rotation is stored as a float
               #  otherwise they are stored as pair of coords x,y both floats
               if lineSplit[0].lower() == 'theta' or \
                  lineSplit[0].lower() == 'phi':
                  # need to convert to radians
                  jointParams[lineSplit[0].lower()] = math.radians(float(lineSplit[1]))
               elif lineSplit[0].lower() == 'alpha':
                  jointParams['alpha'] = float(lineSplit[1])
               else:
                  xycoord = lineSplit[1].split(',')
                  jointParams[lineSplit[0]] = np.array([float(xycoord[0]), float(xycoord[1])])
         except Exception as e:
            print("Error parsing input joint file line %s - %s" % (lineIdx+1, e))

   return joints

def getJointsPtsFromTestFile(inFile):
   biMinus1_ai_bi_alpha = []
   # after reading the file now calculate whether the joint can be a bezier curve or not
   joints = getJointsDataFromTestFile(inFile)

   for idx, joint in enumerate(joints):
      print("processing joint %s - %s" % (idx+1, joint))
      biMinus1 = joint.get("biMinus1")
      ai = joint.get("ai")
      bi = joint.get("bi")
      alpha = joint.get("alpha")

      try:
         # need to transpose to apply rotation / translation to all points in one operation
         pointsMat = np.array([biMinus1, ai, bi])
         # apply the rotation if applicable
         theta = joint.get("theta")
         shift = joint.get("shift", np.array([0,0]))
         if theta:
            # need to transpose to apply rotation to all points in one operation
            pointsMat = np.transpose(pointsMat)
            rotationMat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            pointsMat = np.matmul(rotationMat, pointsMat)
            pointsMat = np.transpose(pointsMat)
         else:
            rotationMat = np.array([[math.cos(0), -math.sin(0)], [math.sin(0), math.cos(0)]])

         pointsMat += shift

         biMinus1 = pointsMat[0]
         ai = pointsMat[1]
         bi = pointsMat[2]

         biMinus1_ai_bi_alpha.append((biMinus1, ai, bi, alpha))

         print("testPotrace - biMinus1 = %s : ai = %s : bi = %s : theta = %s : alpha = %s : rotationMatrix = %s : shift = %s" \
               % (biMinus1, ai, bi, theta, alpha, rotationMat, shift))
      except Exception as e:
         biMinus1_ai_bi_alpha.append((None, None, None, None))
         print(traceback.format_exc())

   return biMinus1_ai_bi_alpha

# This API takes as input a text file with lineCls objects specified in the
# following format
#  x1,y1
#  x2,y2
#  -
# where the termPts section is temrinated by a single dash '-'
# the 1st entry is termPt1 and the 2nd entry is termPt2
def readLineObjFromTxtFile(fileName, ptToIdx={}):
   lines = []
   with open(fileName) as linesInfo:
      termPts = [None, None]
      currIdx = 0
      for entry in linesInfo:
         try:
            if entry.rstrip() == '-':
               if currIdx == 2:
                  lines.append(lineCls(termPts[0], termPts[1]))
                  currIdx = 0
               else:
                  raise Exception("LineCls input file has incorrect format - currIdx=%s" % (currIdx))
            else:
               # remove the index entry from the line
               ptCoords = re.sub('\([0-9]+\)', '', entry)
               ptCoords = ptCoords.lstrip().rstrip().split(',')
               pt = (float(ptCoords[0]), float(ptCoords[1]))
               # if there is an index value assigned to the value (denoted as (1))
               # then extract it and assign that idx to that point
               idxSearch = indexRe.search(entry)
               if idxSearch:
                  idx = int(idxSearch.group(1))
                  ptToIdx[pt] = idx
               termPts[currIdx] = np.array(pt)
               currIdx += 1
         except Exception as e:
            print("ERROR reading lines file %s - %s" % (fileName, e))
            print(traceback.format_exc())
   return lines

############# READ TEXT FILE SECTION END

# API to add the orig segs that the curves are derived from
#
# the segPts biMinus1, ai, bi are the pts on the joint that form the bCurve
# the joint is also the orig line segs that the curve approximates
# NOTE: check if biMinus1 of the current joint is eqaul to the bi of the previous joint
# if it is - this means that the joint is contiguous - dont have to re-add the biMinus1
def addOrigSegFromCurves(segPtsList, listOfPtsToAdd):
   segPts = segPtsList
   if len(segPts) > 0 and np.array_equal(listOfPtsToAdd[0], segPts[-1]):
      segPts.extend(listOfPtsToAdd[1:])
   else:
      segPts.extend(listOfPtsToAdd)

   return segPts

# API to draw any arbitrary shape using matplotlib
#  INPUT: array of pts
#         image name
#         color of drawing (optional)
#
#  OUTPUT: image with given name
def drawMatPlotLib(ptsLists, imgName, curveColors=['red']):
   fig = plt.figure()
   ax = plt.subplot(111)

   for idx, ptsList in enumerate(ptsLists):
      ptsX = [pt[0] for pt in ptsList]
      ptsY = [pt[1] for pt in ptsList]
      ax.plot(ptsX, ptsY, curveColors[idx])

   plt.savefig(imgName)


############################################################
###### test APIs for vertices graph tree structure that
###### maintains adjacency relationship
############################################################
