#!/usr/bin/python

import numpy as np
import cv2 as cv
from numpy import linalg as LA
from enum import Enum
from operator import itemgetter, attrgetter
import math
import copy
import json
from boundBoxCls import *

class lineType(int, Enum):
   UNKNOWN = 0
   SLOPED = 1
   HORIZONTAL = 2
   VERTICAL = 3

class valType(int, Enum):
   NONE = 1
   UNIQUEVAL = 2
   ALLREALVAL = 3

class orientation(int, Enum):
   UNKNOWN = 0
   CW = 1
   CCW = 2
   STRAIGHT = 3

class lineCls:

   def __init__(self):
      self.termPt1 = np.array([False, False])
      self.termPt2 = np.array([False, False])

      self.midPts = []

      self.lineType = None

      self.yMin = None
      self.yMax = None

      self.xMin = None
      self.xMax = None

      self.lineSlope = None
      self.yIntercept = None
 #     self.xIntercept = None

      self.A = None
      self.B = None
      self.C = None

      self.lineLength = None
      self.lineMidPt = None

      self.unitVect = None

      self.maxArea = 0

   def __eq__(self, other):
      if (np.array_equal(self.termPt1, other.termPt1) and \
          np.array_equal(self.termPt2, other.termPt2)) or \
         (np.array_equal(self.termPt1, other.termPt2) and \
          np.array_equal(self.termPt2, other.termPt1)):
         return True

      return False

   def __ne__(self, other):
      return not self.__eq__(other)


   # this is the hash API needed to hash a line so that the lineCls object can be used
   # as a key in a dict - hash by using start and end (x,y) pts in 4 member tuple (startX, startY, endX, endY)
   # where
   #  start pt is defined as pt with min value of X
   #    - if X of start and end pt is the same, take the min value of Y
   def __hash__(self):
      if self.termPt1[0] < self.termPt2[0]:
         hashTuple = (self.termPt1[0], self.termPt1[1], self.termPt2[0], self.termPt2[1])
      elif self.termPt1[0] == self.termPt2[0]:
         if self.termPt1[1] <= self.termPt2[1]:
            hashTuple = (self.termPt1[0], self.termPt1[1], self.termPt2[0], self.termPt2[1])
         else:
            hashTuple = (self.termPt2[0], self.termPt2[1], self.termPt1[0], self.termPt1[1])
      else:
         hashTuple = (self.termPt2[0], self.termPt2[1], self.termPt1[0], self.termPt1[1])

      return hash(hashTuple)

   def dumpAsJSON(self):

      if self.lineSlope:
         lineSlope = float(self.lineSlope)
      else:
         lineSlope = None

      if self.yIntercept:
         yIntercept = float(self.yIntercept)
      else:
         yIntercept = None

      if self.A:
         A = float(self.A)
      else:
         A = None

      if self.B:
         B = float(self.B)
      else:
         B = None

      if self.C:
         C = float(self.C)
      else:
         C = None

      return {
                "termPt1" : [float(self.termPt1[0]), float(self.termPt1[1])],
                "termPt2" : [float(self.termPt2[0]), float(self.termPt2[1])],
                "lineType" : self.lineType,
                "yMin" : float(self.yMin),
                "yMax" : float(self.yMax),
                "xMin" : float(self.xMin),
                "xMax" : float(self.xMax),
                "lineSlope" : lineSlope,
                "yIntercept" : yIntercept,
                "A" : A,
                "B" : B,
                "C" : C,
                "lineLength" : float(self.lineLength),
                "lineMidPt" : [float(self.lineMidPt[0]), float(self.lineMidPt[1])],
                "unitVect" : [float(self.unitVect[0]), float(self.unitVect[1])]
             }

   def populateLineInfoFromJSON(self, jsonEntry):
      self.termPt1 = np.array(jsonEntry.get("termPt1", [False, False]))
      self.termPt2 = np.array(jsonEntry.get("termPt2", [False, False]))
      self.lineType = lineType(jsonEntry.get("lineType", None))
      self.yMin = jsonEntry.get("yMin", None)
      self.yMax = jsonEntry.get("yMax", None)
      self.xMin = jsonEntry.get("xMin", None)
      self.xMax = jsonEntry.get("xMax", None)
      self.lineSlope = jsonEntry.get("lineSlope", None)
      self.yIntercept = jsonEntry.get("yIntercept")
      self.A = jsonEntry.get("A", None)
      self.B = jsonEntry.get("B", None)
      self.C = jsonEntry.get("C", None)
      self.lineLength = jsonEntry.get("lineLength", None)
      self.lineMidPt = np.array(jsonEntry.get("lineMidPt", [False, False]))
      self.unitVect = np.array(jsonEntry.get("unitVect", [False, False]))

   def setStartPt(self, pt):
      print("changing start pt of line from " + str(self.termPt1) + " to " + str(pt) + " -- end pt is " + str(self.termPt2))
      self.termPt1 = pt
      self.calcLineMetadata()

   def setEndPt(self, pt):
      print("changing end pt of line from " + str(self.termPt2) + " to " + str(pt) + " -- start pt is " + str(self.termPt1))
      self.termPt2 = pt
      self.calcLineMetadata()

   def checkIfLineExt(self, pt):
      #self.midPts.append(pt)
      xPtsList = [self.termPt1[0]]
      yPtsList = [self.termPt1[1]]

      for p in self.midPts:
         xPtsList.append(p[0])
         yPtsList.append(p[1])

      xPtsList.append(pt[0])
      yPtsList.append(pt[1])

      xPts = np.array(xPtsList)
      yPts = np.array(yPtsList)

      area = 0.5*np.abs(np.dot(xPts,np.roll(yPts,1)) - np.dot(yPts,np.roll(xPts,1)))

      if area <= self.maxArea:
         status = True
         self.midPts.append(pt)
      else:
         print("midPts are " + str(self.midPts))
         print("startpt is " + str(self.termPt1))
    #     del self.midPts[-1]
         print("midPts after del is " + str(self.midPts))
         status = False

      return status

   def finalizeLine(self):
      if len(self.midPts) > 0:
         self.termPt2 = self.midPts[-1]
      else:
         self.termPt2 = self.termPt1
  #    print("midPts before inserting to termPt2 is " + str(self.midPts))
  #    print("last element of midPts before inserting to termPt2 is " + str(self.midPts[-1]))
  #    self.termPt2 = self.midPts[-1]
      self.calcLineMetadata()

   # calculate line type, line slope, yIntercept, (A, B, C in Ax+By+c = 0)
   # xMin, xMax, yMin, yMax
   def calcLineMetadata(self):
      if self.termPt1[1] == self.termPt2[1]:
         self.lineType = lineType.HORIZONTAL
         self.lineSlope = 0
      elif self.termPt1[0] == self.termPt2[0]:
         self.lineType = lineType.VERTICAL
      else:
         self.lineType = lineType.SLOPED
         self.lineSlope = (self.termPt2[1] - self.termPt1[1])/(self.termPt2[0] - self.termPt1[0])
         self.yIntercept = self.termPt2[1] - self.lineSlope * self.termPt2[0]
  #       self.xIntercept = -1 * self.yIntercept / self.lineSlope
         self.A = self.lineSlope
         self.B = -1.0
         self.C = self.yIntercept

      if self.termPt1[0] > self.termPt2[0]:
         self.xMax = self.termPt1[0]
         self.xMin = self.termPt2[0]
      else:
         self.xMax = self.termPt2[0]
         self.xMin = self.termPt1[0]

      if self.termPt1[1] > self.termPt2[1]:
         self.yMax = self.termPt1[1]
         self.yMin = self.termPt2[1]
      else:
         self.yMax = self.termPt2[1]
         self.yMin = self.termPt1[1]

      self.lineLength = LA.norm(self.termPt2 - self.termPt1)
      xMid = ((self.termPt2[0] - self.termPt1[0]) / 2) + self.termPt1[0]
      yMid = ((self.termPt2[1] - self.termPt1[1]) / 2) + self.termPt1[1]
      self.lineMidPt = np.array([xMid, yMid])
      self.midPts = []

      vect = self.termPt2 - self.termPt1
      if LA.norm(vect) > 0:
         self.unitVect = vect / LA.norm(vect)
      else:
         self.unitVect = vect

   def getYGivenX(self, xcoord):
      retValue = (valType.NONE, 0)
      if self.lineType == lineType.HORIZONTAL:
         retValue = (valType.UNIQUEVAL, self.yMax) # horizontal line - yMin should be equal to yMax
      elif self.lineType == lineType.VERTICAL:
         if xcoord == xMin: # vertical line - xMin should be equal to xMax
            retValue = (valType.ALLREALVAL, 0)
      else:
         ycoord = self.lineSlope * xcoord + self.yIntercept
         retValue = (valType.UNIQUEVAL, ycoord)

      return retValue

   def getXGivenY(self, ycoord):
      retValue = (valType.NONE, 0)
      if self.lineType == lineType.HORIZONTAL:
         if ycoord == self.yMax: # horizontal line - yMin should be equal to yMax
            retValue = (valType.ALLREALVAL, 0)
      elif self.lineType == lineType.VERTICAL:
         retValue = (valType.UNIQUEVAL, self.xMax) # vertical line = xMin should be equal to xMax
      else:
         xcoord = (ycoord - self.yIntercept) / self.lineSlope
         retValue = (valType.UNIQUEVAL, xcoord)

      return retValue

   def determineIfPtIsWithinLineSpan(self, pt):
      if (pt[0] < self.xMin) or \
         (pt[0] > self.xMax) or \
         (pt[1] < self.yMin) or \
         (pt[1] > self.yMax):
         return False

      return True

   def displayLineInfo(self):
      print("line start pt: " + str(self.termPt1))
      print("line end pt: " + str(self.termPt2))
      print("line type is " + str(self.lineType))
      print("line length is " + str(self.lineLength))
      print("line midpoint is: " + str(self.lineMidPt))
      if self.lineSlope:
         print("line slope is: " + str(self.lineSlope))
      print("line unit vector is: " + str(self.unitVect))

   def modifyTermPt(self, ptCoord, ptType):
      if ptType == 1:
         self.termPt1 = ptCoord
      elif ptType == 2:
         self.termPt2 = ptCoord
      else:
         print("Unrecognized ptType " + str(ptType) + " - must be 1 (start) or 2 (end)")
         return

      self.calcLineMetadata()

   def checkIfPointIsOnLine(self, pt):
      diffVect = pt - self.termPt1

      if LA.norm(diffVect) == 0:
         print("pt is actually the start pt of the line " + str(self.termPt1))
         return True

      uDiffVect = diffVect / LA.norm(diffVect)

      if (np.dot(uDiffVect, self.unitVect) == 1) and \
         (LA.norm(diffVect) <= self.lineLength):
         print("pt " + str(pt) + " LIES on line with start pt " + str(self.termPt1) + " and end pt " + str(self.termPt2))
         return True

      print("uDiffVect is " + str(uDiffVect) + " unitVect is " + str(self.unitVect) + " ptDist is " + str(LA.norm(diffVect)) + " lineLength is " + str(self.lineLength))
      print("pt " + str(pt) + " DOES NOT LIE on line with start pt " + str(self.termPt1) + " and end pt " + str(self.termPt2))
      return False

   def flipLine(self):
      # rotate the line by 180 degrees
      tmpPoint = self.termPt1
      self.termPt1 = self.termPt2
      self.termPt2 = tmpPoint
      self.calcLineMetadata()

   def closestDistBtwnPtAndLine(self, pt):
      # return the closest distance between the pt and
      #  either of the end points of the line
      #  also return 1/2 (start pt vs end pt of line)
      dist1 = LA.norm(pt - self.termPt1)
      dist2 = LA.norm(pt2 - self.termPt2)

      retTuple = (1, dist1)

      if dist2 > dist:
         retTuple = (2, dist2)

      return retTuple

   def prepDistBtwnPtAndLine(self, pt):
      # return the perpendicular distance between point and line
      # there are two perpendicular vects given norm vect (x,y) ->
      #   (-y,x) and (y,-x)
      # doesn't matter which one chosen because we only care about dist
      perpVect = np.array([-self.unitVect[1], self.unitVect[0]])
      diff = pt - self.termPt1
      normDiff = diff / LA.norm(diff)
      normDot = np.dot(normDiff, perpVect)
      perpDist = LA.norm(diff) * normDot

      return perpDist

   def calcLenOfOverlapBtwnPtAndLine(self, pt):
      # calculate the length of the line that the pt
      #  is within the span of the line (ie. if pt were
      #  perpendicularly projected onto the line
      #  get the dist between start pt and projected pt)
      diff = pt - self.termPt1
      normDiff = diff / LA.norm(diff)
      parallelAngle = np.dot(self.unitVect, self.normDiff)
      parallelProj = LA.norm(diff) * parallelAngle

      return parallelProj

   # note: return -1 if pt is outside of line
   # return the following tuple - (dist btwn pt and line, closest to point on line (x), closest to point on line (y))
   def getDistBetweenPtAndLineV1(self, pt):
      retTuple = (-1, 0, 0)
      if self.lineType == lineType.HORIZONTAL:
         if pt[0] >= self.xMin and pt[0] <= self.xMax:
            retDist = math.fabs(pt[1] - self.yMin)  # since line is horizontal, yMin = yMax
            retTuple = (retDist, pt[0], self.yMin)
      elif self.lineType == lineType.VERTICAL:
         if pt[1] >= self.yMin and pt[1] <= self.yMax:
            retDist = math.fabs(pt[0] - self.xMin) # since line is vertical, xMin = xMax
            retTuple = (retDist, self.xMin, pt[1])
      else:
         retDist = math.fabs(self.A * pt[0] + self.B * pt[1] + self.C) / math.sqrt(math.pow(self.A, 2) + math.pow(self.B, 2))
         retX = (self.B * (self.B * pt[0] - self.A * pt[1]) - self.A * self.C) / (math.pow(self.A, 2) + math.pow(self.B, 2))
         retY = (self.A * (-1 * self.B * pt[0] + self.A * pt[1]) - self.B * self.C) / (math.pow(self.A, 2) + math.pow(self.B, 2))
         retTuple = (retDist, retX, retY)

      return retTuple

   # since these are not infinite lines but line segments, can use
   # system of linear equation to solve
   #
   # perpVect is the perpendicular vector between pt and the line
   #
   #   A * unitVectX + pt1X = ptX + B * perpVectX
   #   A * unitVectY + pt1Y = ptY + B * perpVectY
   #   Variables to solve are A, B
   #   Rearrange into C1A + C2B = C3
   #
   #  return value - (A<>, abs(B)<>)
   #  INPUT: pt of interest to get dist from
   #         startOrEnd - 1 is startPt (termPt1), 2 is endPt (termPt2)
   def getDistBtwnPtAndStartOrEndOfLine(self, pt, startOrEnd=1):

      refPt = self.termPt1
      unitVect = self.unitVect
      # set the refPt to be the end point of the line
      # and take the negative of the normal vector since we are using endpt as ref
      # this way, if pt projected to ref line lies within line, A > 0
      if startOrEnd == 2:
         refPt = self.termPt2
         unitVect = -self.unitVect

      perpVect = np.array([-self.unitVect[1], self.unitVect[0]])
      print("chosen perpVect is " + str(perpVect))
      A1Const = unitVect[0]
      B1Const = -perpVect[0]
      C1Const = pt[0] - refPt[0]

      A2Const = unitVect[1]
      B2Const = -perpVect[1]
      C2Const = pt[1] - refPt[1]

      print("A1Const: " + str(A1Const) + " B1Const: " + str(B1Const) + " C1Const: " + str(C1Const))
      print("A2Const: " + str(A2Const) + " B2Const: " + str(B2Const) + " C2Const: " + str(C2Const))

      a = np.array([[A1Const, B1Const],[A2Const, B2Const]])
      b = np.array([C1Const, C2Const])

      x = LA.solve(a,b)
      print("result is " + str(x))

      return (x[0], math.fabs(x[1]))

   # this API projects:
   #  - the start pt of the input line to the start of the self line
   #    using the unit vect of the self line and the perp vect of the input line
   #  - OR the end pt of the input line to the end of the self line
   #    using the unit vect of the self line and the perp vect of the input line
   # from which this member function is called
   #
   # by using this eqn: termPt1Self + A * UnitSelf = termPt1In + B * UnitInPerp
   # OR                 termPt2Self + A * (-UnitSelf) = termPt2In + B * UnitInPerp
   #
   #
   #   A is the parameter of interest because it is used to calculate the projected start / end pt
   #   from the in line to the self line
   #
   # this API assumes that the in line and the self line are oriented in the same direction
   #   (ie. UnitVectSelf dot UnitVecIn >= 0)
   #
   # return A/B in tuple form - where A is the magnitude and direction of the vector
   # between the start pt of the self line and the projected start pt of the inLine
   #  OR btwn the end pt of the self line and the projected end pt of the inLine
   def lineProjInputToSelfWithInPerpSelfUnit(self, inLine, projEndPt=False):

      # check that the secLine and the refLine are oriented in the same direction - if not , spit out warning
      if np.dot(self.unitVect, inLine.unitVect) < 0:
         print("ERROR - lineProjInputToSelfWithInPerpSelfUnit expects self line and inLine to be in same direction")

      # set the refPt to be the end point of the line
      # and take the negative of the normal vector since we are using endpt as ref
      # this way, if pt projected to ref line lies within line, A > 0
      if projEndPt:
         selfPt = self.termPt2
         unitVect = -self.unitVect
         inPt = inLine.termPt2
      else:
         selfPt = self.termPt1
         unitVect = self.unitVect
         inPt = inLine.termPt1

      perpVect = np.array([-inLine.unitVect[1], inLine.unitVect[0]])
     # perpVect = np.array([-self.unitVect[1], self.unitVect[0]])

      # check if the perp of the in line unit vect is parallel / antiparallel to unit vect of self line
      # if so - means that the in line is already perpendicular to self line
      if LA.norm(np.dot(perpVect, self.unitVect)) > 0.99:
         perpVect = inLine.unitVect

      print("chosen perpVect is " + str(perpVect))
      A1Const = unitVect[0]
      B1Const = -perpVect[0]
      C1Const = inPt[0] - selfPt[0]

      A2Const = unitVect[1]
      B2Const = -perpVect[1]
      C2Const = inPt[1] - selfPt[1]

      print("A1Const: " + str(A1Const) + " B1Const: " + str(B1Const) + " C1Const: " + str(C1Const))
      print("A2Const: " + str(A2Const) + " B2Const: " + str(B2Const) + " C2Const: " + str(C2Const))

      a = np.array([[A1Const, B1Const],[A2Const, B2Const]])
      b = np.array([C1Const, C2Const])

      try:
         x = LA.solve(a,b)
         print("result is " + str(x))
      except:
         print("ERROR TRYING TO SOLVE 2x2 matrix")
         print("a is " + str(a) + " and b is " + str(b))

      return (x[0], math.fabs(x[1]))

   # this API projects:
   #  - the start pt of the input line to the start of the self line
   #    using the unit vect AND the perp vect of the self line
   #  - OR the end pt of the input line to the end of the self line
   #    using the unit vect AND the perp vect of the self line
   # from which this member function is called
   #
   # by using this eqn: termPt1Self + A * UnitSelf + B * UnitSelfPerp = termPt1In
   # OR                 termPt2Self + A * (-UnitSelf) + B * UnitSelfPerp = termPt2In
   #
   #
   #   A is the parameter of interest because it is used to calculate the projected start / end pt
   #   from the in line to the self line
   #
   # this API assumes that the in line and the self line are oriented in the same direction
   #   (ie. UnitVectSelf dot UnitVecIn >= 0)
   #
   # return A/B in tuple form - where A is the magnitude and direction of the vector
   # between the start pt of the self line and the projected start pt of the inLine
   #  OR btwn the end pt of the self line and the projected end pt of the inLine
   def lineProjInputToSelfWithSelfPerpSelfUnit(self, inLine, projEndPt=False):

      # check that the secLine and the refLine are oriented in the same direction - if not , spit out warning
      if np.dot(self.unitVect, inLine.unitVect) < 0:
         print("ERROR - lineProjSecPerpToRefUnit expects refLine and secLine to be in same direction")

      # set the refPt to be the end point of the line
      # and take the negative of the normal vector since we are using endpt as ref
      # this way, if pt projected to ref line lies within line, A > 0
      if projEndPt:
         selfPt = self.termPt2
         unitVect = -self.unitVect
         inPt = inLine.termPt2
      else:
         selfPt = self.termPt1
         unitVect = self.unitVect
         inPt = inLine.termPt1

      perpVect = np.array([-self.unitVect[1], self.unitVect[0]])

      print("chosen perpVect is " + str(perpVect))
      A1Const = unitVect[0]
      B1Const = perpVect[0]
      C1Const = inPt[0] - selfPt[0]

      A2Const = unitVect[1]
      B2Const = perpVect[1]
      C2Const = inPt[1] - selfPt[1]

      print("A1Const: " + str(A1Const) + " B1Const: " + str(B1Const) + " C1Const: " + str(C1Const))
      print("A2Const: " + str(A2Const) + " B2Const: " + str(B2Const) + " C2Const: " + str(C2Const))

      a = np.array([[A1Const, B1Const],[A2Const, B2Const]])
      b = np.array([C1Const, C2Const])

      try:
         x = LA.solve(a,b)
         print("result is " + str(x))
      except:
         print("ERROR TRYING TO SOLVE 2x2 matrix")
         print("a is " + str(a) + " and b is " + str(b))

      return (x[0], math.fabs(x[1]))

   # check if line is a point (if start pt and end pt is the same)
   def checkIfLineIsPoint(self):
      if np.array_equal(self.termPt1, self.termPt2):
         print("line with start pt " + str(self.termPt1) + " and end pt " + str(self.termPt2) + " has become a point")
         return True

      return False

def checkIf2LinesAreContiguous(line1, line2):
   if np.array_equal(line1.termPt1, line2.termPt1) or \
      np.array_equal(line1.termPt1, line2.termPt2) or \
      np.array_equal(line1.termPt2, line2.termPt1) or \
      np.array_equal(line1.termPt2, line2.termPt2):
      return True

   return False

def checkIfShortLineCompletelyLiesOnLongLine(line1, line2):
   if line1.lineLength > line2.lineLength:
      longLine = line1
      shortLine = line2
   else:
      longLine = line2
      shortLine = line1

   if longLine.checkIfPointIsOnLine(shortLine.termPt1) and \
      longLine.checkIfPointIsOnLine(shortLine.termPt2):
      print("short line with start pt " + str(shortLine.termPt1) + " and end pt " + str(shortLine.termPt2) + " is completely overlapped by long line with start pt " + str(longLine.termPt1) + " and end pt " + str(longLine.termPt2))
      return True

   print("short line with start pt " + str(shortLine.termPt1) + " and end pt " + str(shortLine.termPt2) + " is NOT overlapped by long line with start pt " + str(longLine.termPt1) + " and end pt " + str(longLine.termPt2))
   return False

class contigsSegsCls:

   def __init__(self, lineIdx=None, line=None):
      self.lines = []
      self.lineIdxs = []
      self.startPt = None
      self.endPt = None
      self.length = 0
      self.dotProdBtwnLines = []
      self.orientations = []
      self.intermedPts = []
      self.ptsCount = {}
      self.contigSegments = []
      self.lineIdxOnly = None
      self.startPtNeighborIdx = None
      self.endPtNeighborIdx = None
      # the start and end pt is varied depending
      # on which direction the contig seg is traversed - thus store the term pts
      # as LHSMostTermPt, RHSMostTermPt - where the LHSMostTermPt is the termPt
      #  with the lowest x value - if the termPts have the same x pt, then take the LHSMostTermPt
      #  as the lowest y value
      self.LHSMostTermPt = None
      self.RHSMostTermPt = None

      if lineIdx != None:
         self.lineIdxs.append(lineIdx)
      if line:
         self.lines.append(line)
         self.length += line.lineLength

   def displayContigsSegsData(self):
      for i in range(len(self.lines)):
         print("line " + str(i) + " is: ")
         self.lines[i].displayLineInfo()

      print("Contig seg start pt is " + str(self.startPt))
      print("Contig seg end pt is " + str(self.endPt))
      print("Contig seg length is " + str(self.length))
      print("Contig seg lineIdxs is " + str(self.lineIdxs))

   def insertLineToContigSeg(self, lineIdx, line, checkDotProd=True, insertIdxOnly=False):
      insertedLineToContigSeg = True

      if self.lineIdxOnly and self.lineIdxOnly != insertIdxOnly:
         print("contig seg DOES NOT support inserting line idx ONLY AND inserting lines for certain seg - line not inserted")
         print("self.lineIdxOnly is " + str(self.lineIdxOnly) + " and input insertIdxOnly is " + str(insertIdxOnly))
         return False

      self.lineIdxOnly = insertIdxOnly

      if not self.lines:
         if not insertIdxOnly:
            self.lines.append(line)
         self.lineIdxs.append(lineIdx)
         self.length += line.lineLength
      else:
         insertedLineToContigSeg = checkIf2LinesAreContiguous(self.lines[-1], line)
         if insertedLineToContigSeg:
            # given that the 2 lines are contiguous check - if oriented
            # such that line1.termPt2 == line2.termPt1 if the dot product is < 0
            # if dot product > 0, means they belong to same contig seg
            if checkDotProd:
               line1 = copy.deepcopy(self.lines[-1])
               line2 = copy.deepcopy(line)
               if np.array_equal(line2.termPt2, line1.termPt2):
                  line2.flipLine()
               dotProd = np.dot(line1.unitVect, line2.unitVect)
               if dotProd <= 0:
                  print("insertLineToContigSeg - dot product between last line and candidate line is " + str(dotProd) + " do not insert into contigSeg")
                  print("prev line info ")
                  line1.displayLineInfo()
                  print("this line info ")
                  line2.displayLineInfo()
                  insertedLineToContigSeg = False
               else:
                  if not insertIdxOnly:
                     self.lines.append(line)
                  self.lineIdxs.append(lineIdx)
                  self.length += line.lineLength
            else:
               if not insertIdxOnly:
                  self.lines.append(line)
               self.lineIdxs.append(lineIdx)
               self.length += line.lineLength
         else:
            print("insertLineToContigSeg - failed to insert line with start pt " + str(line.termPt1) + " end pt " + str(line.termPt2) + " after last line in contig with start pt " + str(self.lines[-1].termPt1) + " and end pt " + str(self.lines[-1].termPt2))

      # insert the start / end points of line into contig seg
      if insertedLineToContigSeg:
         if not self.ptsCount.get((line.termPt1[0], line.termPt1[1]), None):
            self.ptsCount[(line.termPt1[0], line.termPt1[1])] = 1
         else:
            self.ptsCount[(line.termPt1[0], line.termPt1[1])] += 1

         if not self.ptsCount.get((line.termPt2[0], line.termPt2[1]), None):
            self.ptsCount[(line.termPt2[0], line.termPt2[1])] = 1
         else:
            self.ptsCount[(line.termPt2[0], line.termPt2[1])] += 1

         if len(self.lines) > 1:
            print("insertLineToContigSeg - successfully inserted line with start pt " + str(line.termPt1) + " end pt " + str(line.termPt2) + " after last line in contig with start pt " + str(self.lines[-2].termPt1) + " and end pt " + str(self.lines[-2].termPt2))
         else:
            print("insertLineToContigSeg - successfully inserted first line with start pt " + str(line.termPt1) + " end pt " + str(line.termPt2) + " into contig seg")

      return insertedLineToContigSeg

   # This contig seg API deletes line segs from the contig seg
   #  since lines and lineIdxs are stored as lists (self.lines) and (self.lineIdxs)
   #
   #  The input of this API is list of lineIdx contiguous spans to delete (of the form)
   #   [(lineIdxStart1, lineIdxEnd1), (lineIdxStart2, lineIdxEnd2)]
   #
   #  THIS API assumes that the lineIdxs that comprise the contigseg are in ascending contiguous
   #   order
   def deleteLinesFromContigSegByLineIdxs(self, listOfLineSegIdxSpansToDel):
      iterIdxOfSpans = 0
      spanToDel = listOfLineSegIdxSpansToDel[iterIdxOfSpans]
      contigSegment = None
      nextStartMod = False
      newLines = []
      newLineIdxs = []

      for idx in range(len(self.lineIdxs)):
         # check if the idx is in the span to delete
         if (self.lineIdxs[idx] >= spanToDel[0]) and \
            (self.lineIdxs[idx] <= spanToDel[1]):
            # if it is in span - check if it is border pt
            if self.lineIdxs[idx] == spanToDel[0]:
               # if there is running span of contiguous lineIdxs for contig segs
               # need to terminate
               if contigSegment:
                  self.contigSegments.append(((contigSegment[0][0], contigSegment[0][1]), (contigSegment[1][0], contigSegment[1][1])))
                  contigSegment = None
            if self.lineIdxs[idx] == spanToDel[1]:
               # shift the next entry in listOfLineSegIdxSpansToDel
               iterIdxOfSpans += 1
               if iterIdxOfSpans < len(listOfLineSegIdxSpansToDel):
                  spanToDel = listOfLineSegIdxSpansToDel[iterIdxOfSpans]
               nextStartMod = True
         else:
            if not contigSegment:
               contigSegment = [[self.lineIdxs[idx], nextStartMod], [self.lineIdxs[idx], True]]
            else:
               contigSegment[1][0] = self.lineIdxs[idx]

            # if the line idx is NOT in the span to delete - add it to newLines and newLineIdxs
            if not self.lineIdxOnly:
               newLines.append(self.lines[idx])
            newLineIdxs.append(self.lineIdxs[idx])


      # need to close out contigSegment if reached end of the self.lineIdxs
      if contigSegment:
         contigSegment[1][1] = False
         self.contigSegments.append(((contigSegment[0][0], contigSegment[0][1]), (contigSegment[1][0], contigSegment[1][1])))

      self.lines = newLines
      self.lineIdxs = newLineIdxs

   def returnAnyNonContigLines(self):
      retList = []

      for i in range(len(self.lines)-1):
         if not orient2ContigLines(self.lines[i], self.lines[i+1]):
            retList.append((i, i+1))

      return retList

   def orient2ContigLines(self, line1, line2):
      rc = True

      if np.array_equal(line1.termPt1, line2.termPt1):
         print("line1 start pt equal to line2 start pt - flip line1")
         line1.flipLine()
      elif np.array_equal(line1.termPt2, line2.termPt2):
         print("line1 end pt equal to line2 end pt - flip line2")
         line2.flipLine()
      elif np.array_equal(line1.termPt1, line2.termPt2):
         print("line1 start pt equal to line2 end pt - flip both line1 and line2")
         line1.flipLine()
         line2.flipLine()
      elif np.array_equal(line1.termPt2, line2.termPt1):
         print("line1 end pt equal to line2 start pt - do nothing")
      else:
         print("line1 and line2 are not contiguous - they share no points")
         rc = False

      return rc

   def orientContigSegLines(self):
      ret = True
      for i in range(0, len(self.lines)-1):
         if not self.orient2ContigLines(self.lines[i], self.lines[i+1]):
            print("error - identified contiguous segment is NOT contiguous - lines " + str(self.lineIdxs[i]) + " and " + str(self.lineIdxs[i+1]) + " not contiguous")
            ret = False

      return ret

   def reverseOrientationOfContigSeg(self):
      self.lines.reverse()
      self.lineIdxs.reverse()

      for line in self.lines:
         line.flipLine()

      self.finalizeContigSeg()

   def getDotProdBtwnLines(self):
      for i in range(0, len(self.lines)-1):
         self.dotProdBtwnLines.append(np.dot(self.lines[i].unitVect, self.lines[i+1].unitVect))

   # orientation of a line is defined as the direction of the unit vector of the line
   #  in relation to the "center of curvature" as defined by "curve" between current line and its immediate
   #  adjacent line -> RIGHT -> CCW, LEFT -> CW
   #
   # to get this orientation - use cross product:
   #    -> a = unit vector of the line seg
   #    -> b = unit vector of the vector generated by subtracting start pt of curr line and the end point of the next line
   #  take a x b -> if curve is (CW), resulting cross-product vector z component is positive
   #             -> if curve is (CCW), resulting cross-product vector z component is negative
   #                 NOTE the above conventions are reversed because y-positive of an image is pointing downwards
   #                  on cartesian plane
   #             -> if curve is straight, resulting cross-product vector z component is 0
   #
   # to take cross product, convert a and b to 3d vector with z-component 0
   def getOrientationsOfLines(self):
      for i in range(0, len(self.lines)-1):
         line1 = self.lines[i]
         line2 = self.lines[i+1]
         diffLine = (line2.termPt2 - line1.termPt1)
         cross = int(np.cross(line1.unitVect, diffLine))
         if cross < 0:
            self.orientations.append(orientation.CCW)
         elif cross > 0:
            self.orientations.append(orientation.CW)
         else:
            self.orientations.append(orientation.STRAIGHT)
      # the last line segment will have the same orientation as the one before it
      if self.orientations:
         lastOrientation = self.orientations[-1]
         self.orientations.append(lastOrientation)

   def getIntermedPts(self):
      for i in range(len(self.lines)-1):
         self.intermedPts.append(self.lines[i].termPt2)

   def clearIntermedData(self):
      self.dotProdBtwnLines = []
      self.orientations = []
      self.intermedPts = []

   def finalizeContigSeg(self):
      if len(self.lines) > 0:
         if self.orientContigSegLines():
            self.startPt = self.lines[0].termPt1
            self.endPt = self.lines[-1].termPt2

            self.clearIntermedData()

            self.getDotProdBtwnLines()
            self.getOrientationsOfLines()

            self.getIntermedPts()

            if (self.startPt[0] < self.endPt[0]) or \
               ((self.startPt[0] == self.endPt[0]) and \
                (self.startPt[1] <= self.endPt[1])):
               self.LHSMostTermPt = self.startPt
               self.RHSMostTermPt = self.endPt
            else:
               self.LHSMostTermPt = self.endPt
               self.RHSMostTermPt = self.startPt

   # shift is an np_array([x, y]) entry
   # that shifts the entire contig seg
   def shiftContigSeg(self, shift):
      for line in self.lines:
         line.termPt1 += shift
         line.termPt2 += shift

      self.finalizeContigSeg()

   def checkIfInputContigSegAdj(self, contigSeg):
      if np.array_equal(self.startPt, contigSeg.startPt) or \
         np.array_equal(self.startPt, contigSeg.endPt) or \
         np.array_equal(self.endPt, contigSeg.startPt) or \
         np.array_equal(self.endPt, contigSeg.endt):
         return True

      return False

   def __eq__(self, other):

      equal = True

      if np.array_equal(self.startPt, other.endPt) and \
         np.array_equal(self.endPt, other.startPt):
         other.reverseOrientationOfContigSeg()

      if np.array_equal(self.startPt, other.startPt) and \
         np.array_equal(self.endPt, other.endPt) and \
         (len(self.lines) == len(other.lines)):
         for i in range(len(self.lines)):
            if self.lines[i] != other.lines[i]:
               equal = False
               break
      else:
         equal = False

      return equal

   def __ne__(self, other):
      return not self.__eq__(other)

#############################################################
#####
#### APIs for contigSeg where actual line segs are NOT
#### stored locally but are part of a map
#####
#############################################################

   def calcContigSegMetadata(self, lineSegMap=None):
      if not lineSegMap:
         print("line seg map not provided - use line segs stored in contig seg")
         self.finalizeContigSeg()
      else:
         print("line seg map provided - use the idx to get line segs")
         for idx in self.lineIdxs:
            try:
               self.length += lineSegMap.get(idx).lineLength
            except:
               print("failed to calcContigSegMetadata at idx " + str(idx))
               print("lineSegMap is " + str(lineSegMap))
               self.displayContigsSegsData()

   def checkContigSegIsContiguous(self, lineSegMap=None):
      print("checking if line segs in contig seg are continguous")
      retVal = True

      if not lineSegMap:
         print("line seg map not provided - must mean line segs stored in contigSeg - go thru stored line segs")
         for i in range(len(self.lines)-1):
            if not np.array_equal(self.lines[i].termPt2, self.lines[i+1].termPt1):
               print("line " + str(i) + " with end pt " + str(self.lines[i].termPt2) + " and line " + str(i+1) + " start pt " + str(self.lines[i+1].termPt1) + " are not contiguous - ERROR")
               retVal = False
      else:
         print("line seg map provided - use the idx to get line segs")
         for i in range(len(self.lineIdxs)-1):
            if not np.array_equal(lineSegMap.get(self.lineIdxs[i]).termPt2, lineSegMap.get(self.lineIdxs[i+1]).termPt1):
               print("line with idx " + str(self.lineIdx[i]) + " with end pt " + str((self.lineIdxs[i]).termPt2) + " and line with idx " + str(self.lineIdxs[i+1]) + " start pt " + str((self.lineIdxs[i+1]).termPt1) + " are not contiguous - ERROR")
               retVal = False

      return retVal

   def printContigSegInfo(self):
      print("number of lines in contiguous segment is " + str(len(self.lines)))
      print("line idxs in contiguous segment are " + str(self.lineIdxs))
      print("contiguous segment start pt is " + str(self.startPt))
      print("contiguous segment end pt is " + str(self.endPt))
      print("contiguous segment dot product between lines are " + str(self.dotProdBtwnLines))
      print("contiguous segment orientations are " + str(self.orientations))
      print("contiguous segment intermediate points are " + str(self.intermedPts))
      print("contiguous segment length is " + str(self.length))
      print("contiguous segment points count is ")
      for pt, count in self.ptsCount.items():
         npPt = np.array([pt[0], pt[1]])
         print("contiguous segment pt " + str(npPt) + " belong to " + str(count) + " lines")
         if count > 2:
            print("ERROR: contiguous segment pt " + str(npPt) + " belong to " + str(count) + " lines")
      for line in self.lines:
         line.displayLineInfo()

def contigSegLenSort(contig):
   return contig.length

def checkIfOneContigCompletelyOverlap(contig1, contig2):
   retContig = None
   notMatch = False

   if len(contig1.lines) < len(contig2.lines):
      shortContig = contig1
      longContig = contig2
   else:
      shortContig = contig2
      longContig = contig1

   loopIncrement = 1
   loopEndIdx = len(longContig.lines)
   loopStartIdx = 0

   foundFirstOverlap = False

   # check to see if the first line in the short contig seg
   # is found in the long contig seg
   for idx in range(len(longContig.lines)):
      if checkIfShortLineCompletelyLiesOnLongLine(shortContig.lines[0], longContig.lines[idx]):
         # check if the lines overlap are in the opposite direction
         if np.dot(shortContig.lines[0].unitVect, longContig.lines[idx].unitVect) < 0:
            loopIncrement = -1
            loopEndIdx = -1
         foundFirstOverlap = True
         loopStartIdx = idx
         break

   # found the idx of line in long contig seg that matches with first line of short contig seg
   if foundFirstOverlap:
      # check to see if number of lines in short contig seg is greater than the number of contig segs
      # starting from idx and ending at loopEndIdx - if so - not overlap
      longContigRemain = math.fabs(loopEndIdx - loopStartIdx)
      if len(shortContig.lines) <= longContigRemain:
         for shortIdx in range(len(shortContig.lines)):
            longIdx = loopStartIdx + (loopIncrement * shortIdx)
            if not checkIfShortLineCompletelyLiesOnLongLine(shortContig.lines[shortIdx], longContig.lines[longIdx]):
               notMatch = True
               break
   else:
      notMatch = True

#   if not notMatch:
#      retContig = shortContig

   return not notMatch

# since these are not infinite lines but line segments, can use
# system of linear equation to solve
#
#   A * unitVect1X + pt1X = pt2X + B * UnitVect2X
#   A * unitVect1Y + pt1Y = pt2Y + B * UnitVect2Y
#   Variables to solve are A, B
#   Rearrange into C1A + C2B = C3
#
#  return value - pt of intersection -> if there is no point of intersection
#                 return None
#  INPUT: line1 and line2
def getIntersectPtBtwn2Lines(line1, line2):
   A1Const = line1.unitVect[0]
   B1Const = -line2.unitVect[0]
   C1Const = line2.termPt1[0] - line1.termPt1[0]
   A2Const = line1.unitVect[1]
   B2Const = -line2.unitVect[1]
   C2Const = line2.termPt1[1] - line1.termPt1[1]

   a = np.array([[A1Const, B1Const],[A2Const, B2Const]])
   b = np.array([C1Const, C2Const])

   try:
      soln = LA.solve(a,b)
      intersectX = soln[0] * line1.unitVect[0] + line1.termPt1[0]
      intersectY = soln[0] * line1.unitVect[1] + line1.termPt1[1]
      retPt = np.array([intersectX, intersectY])
      print("lines intersect at pt " + str(retPt))
   except:
      print("lines do not intersect - line1 unitVect " + str(line1.unitVect) + " line2 unitVect " + str(line2.unitVect))
      retPt = None

   return retPt

def getXSectPtBtwn2LinesCheckIfInRefLine(refLine, secLine):
   retPt = np.array([False, False])
   xsectPt = getIntersectPtBtwn2Lines(refLine, secLine)
   if xsectPt.all():
      if refLine.checkIfPointIsOnLine(xsectPt):
         retPt = xsectPt

   return retPt

def filterOutPtLinesInContigSeg(contigSegIn):
   retContigSeg = contigsSegsCls()
   for idx in range(len(contigSegIn.lines)):
      if not contigSegIn.lines[idx].checkIfLineIsPoint():
         retContigSeg.insertLineToContigSeg(contigSegIn.lineIdxs[idx], contigSegIn.lines[idx], False)
      else:
         print("line with index " + str(contigSegIn.lineIdxs[idx]) + " is a point - do not insert into contigSeg")
         contigSegIn.lines[idx].displayLineInfo()

   retContigSeg.finalizeContigSeg()
   return retContigSeg

# API to check if the 2 lines passed into the API are redundant / parallel
#  meaning that if the 2 lines can be replaced by just one line - this is due to
#  tracing a pencil drawing and you get 2 edges
def checkIf2LinesRedundant(self, line1, line2, maxPerpDist=5.0):

   returnLine = None

   defaultMinDotProdParallel = 0

   dotProdBtw2Lines = np.dot(line1.unitVect, line2.unitVect)
   print("dot is " + str(dotProdBtw2Lines) + " line1.unitVect " + str(line1.unitVect) + " line2.unitVect " + str(line2.unitVect))
   if LA.norm(dotProdBtw2Lines) >= defaultMinDotProdParallel:
      if line1.lineLength > line2.lineLength:
         refLine = copy.deepcopy(line1)
         secLine = copy.deepcopy(line2)
         print("refLine is line1 - secLine is line2")
         delLineToReturn = 2

      else:
         refLine = copy.deepcopy(line2)
         secLine = copy.deepcopy(line1)
         print("refLine is line2 - secLine is line1")
         delLineToReturn = 1

      # check if the 2 lines are oriented in opposite directions
      # if so, flip the secondary line
      if dotProdBtw2Lines < 0:
         secLine.flipLine()
         print("dot product between refLine and secLine is negative - flip it")

      secLinePt1Dist = refLine.getDistBtwnPtAndStartOrEndOfLine(secLine.termPt1, 1)
      secLinePt2Dist = refLine.getDistBtwnPtAndStartOrEndOfLine(secLine.termPt2, 2)

      print("dot product between line1 and line2 is " + str(dotProdBtw2Lines))
      print("secLinePt1Dist is " + str(secLinePt1Dist) + " length of refLine is " + str(refLine.lineLength))
      print("secLinePt2Dist is " + str(secLinePt2Dist) + " length of refLine is " + str(refLine.lineLength))

      normDistFromStartPt = secLinePt1Dist[0]
      perpDistFromStartPt = secLinePt1Dist[1]
      normDistFromEndPt = secLinePt2Dist[0]
      perpDistFromEndPt = secLinePt2Dist[1]

      print("normDistFromStartPt is " + str(normDistFromStartPt) + " perpDistFromStartPt is " + str(perpDistFromStartPt) + " normDistFromEndPt is " + str(normDistFromEndPt) + " perpDistFromEndPt is " + str(perpDistFromEndPt) + " refLine length is " + str(refLine.lineLength))

      if (perpDistFromStartPt < maxPerpDist) and \
         (perpDistFromEndPt < maxPerpDist):
         if (normDistFromStartPt >= 0) and \
            (normDistFromEndPt >= 0):
            returnLine = delLineToReturn
         else:
            #  if ((normDistFromStartPt < 0) and (abs(normDistFromStartPt) < refLine.lineLength * self.dispFactor)) or \
            #     ((normDistFromEndPt < 0) and (abs(normDistFromEndPt) < refLine.lineLength * self.dispFactor)) :
            if ((normDistFromStartPt > 0) and (normDistFromStartPt < refLine.lineLength)) or \
               ((normDistFromEndPt > 0) and (normDistFromEndPt < refLine.lineLength)):
               returnLine = delLineToReturn

   return returnLine

class lineMapCls:

   def __init__(self):
      # this is a MAP of line idxs to actual lineCls lines
      # this map has format {lineIdx : lineCls object}
      self.lineIdxToLineMap = {}
      # this is a MAP of contour idx to the line idx of the lines that make up this contour
      # this map has format {contourIdx : [lineIdxs]}
      self.lineContourToLineIdxs = {}
      # this is a MAP of lines that ALREADY exist (the line may go in different direction
      # ie if startPt of line1 is A and endPt of line1 is B -> the following 2 lines are considered
      #  identical (line1: A, B) (line2: B, A)
      # map is of format {key - (startX, startY, endX, endY) : value - lineIdx}
      self.lineTermPtsCacheMap = {}

      self.maxLineIdx = 0
      self.minDotProdParallel = 0
      self.dispFactor = 2
      self.maxPerpDist = 5

      # this is a MAP of contour idx to its unqiue contours after deleting parallel segs in self
      # the map has format {contourIdx : {1: [array of line idxs], 2: [array of line idxs]}}
      self.contourIdxToUniqueConts = {}

      # this is a MAP of contour idx to the processed unique contig segs - these are processed by merging
      # the 2 unique contours for each contour idx in the above map
      self.contourIdxToUniqueContigSegs = {}

      # this is MAP of total contig segs length to the list of contig segs for that contour
      # NOTE: the value is list of lists because 2 different contours can have the same total length
      # of the contig segs
      self.contigSegsTotalLenToContigSegs = {}

      self.uniqueContigSegs = []

      self.maxContigSegIdx = 0

      self.imgHeight = None
      self.imgWidth = None

      self.minX = self.maxX = self.minY = self.maxY = 0

   def getAllLinesAsList(self):
      retLineList = []
      for idx in self.lineIdxToLineMap:
         retLineList.append(self.lineIdxToLineMap.get(idx, None))

      return retLineList

   def readInfoFromJSON(self, jsonName):
      with open(jsonName, 'r') as jsonFile:
         jsonData = json.load(jsonFile)

      lineIdxToLineMapJSON = jsonData.get("lineIdxToLineMap", {})
      for key, lineValues in lineIdxToLineMapJSON.items():
         line = lineCls()
         line.populateLineInfoFromJSON(lineValues)
         self.lineIdxToLineMap[int(key)] = line

      lineContourToLineIdxJSON = jsonData.get("lineContourToLineIdxs", {})
      for key, contourIdxs in lineContourToLineIdxJSON.items():
         self.lineContourToLineIdxs[int(key)] = contourIdxs

      contourIdxToUniqueContsJSON = jsonData.get("contourIdxToUniqueConts", {})
      for key, uniqueConts in contourIdxToUniqueContsJSON.items():
         uniqueContsEntry = {}
         for idx, cont in uniqueConts.items():
            uniqueContsEntry[int(idx)] = cont
         self.contourIdxToUniqueConts[int(key)] = uniqueContsEntry

      self.maxLineIdx = jsonData.get("maxLineIdx", 0)

      self.imgHeight = jsonData.get("imgHeight", None)
      self.imgWidth = jsonData.get("imgWidth", None)

      # regenerate the lineTermPtsCacheMap
      for idx, line in self.lineIdxToLineMap.items():
         self.lineTermPtsCacheMap[(line.termPt1[0], line.termPt1[1], line.termPt2[0], line.termPt2[1])] = idx

   def dumpInfoToJSON(self, jsonName):
      lineIdxToLineMapJSONFormat = {}

      for key, line in self.lineIdxToLineMap.items():
         lineIdxToLineMapJSONFormat[key] = line.dumpAsJSON()

      jsonToDump = {
                      "lineIdxToLineMap" : lineIdxToLineMapJSONFormat,
                      "lineContourToLineIdxs" : self.lineContourToLineIdxs,
                      "contourIdxToUniqueConts" : self.contourIdxToUniqueConts,
                      "maxLineIdx" : self.maxLineIdx,
                      "imgHeight" : self.imgHeight,
                      "imgWidth" : self.imgWidth
                   }

      with open(jsonName, 'w') as jsonFile:
         json.dump(jsonToDump, jsonFile)

   def insertLineToIdxMap(self, line):
      retIdx = -1
      if line.lineLength > 0:
         termPtKey1 = (line.termPt1[0], line.termPt1[1], line.termPt2[0], line.termPt2[1])
         termPtKey2 = (line.termPt2[0], line.termPt2[1], line.termPt1[0], line.termPt1[1])

         if not self.lineTermPtsCacheMap.get(termPtKey1, None) and \
            not self.lineTermPtsCacheMap.get(termPtKey2, None):

            self.lineIdxToLineMap[self.maxLineIdx] = line
            retIdx = self.maxLineIdx

            if self.minX > line.termPt1[0]:
                self.minX = line.termPt1[0]
            elif self.maxX < line.termPt1[0]:
                self.maxX = line.termPt1[0]
            if self.minX > line.termPt2[0]:
                self.minX = line.termPt2[0]
            elif self.maxX < line.termPt2[0]:
                self.maxX = line.termPt2[0]

            if self.minY > line.termPt1[1]:
                self.minY = line.termPt1[1]
            elif self.maxY < line.termPt1[1]:
                self.maxY = line.termPt1[1]
            if self.minY > line.termPt2[1]:
                self.minY = line.termPt2[1]
            elif self.maxY < line.termPt2[1]:
                self.maxY = line.termPt2[1]

            self.imgWidth = int((self.maxX - self.minX ) * 2)
            self.imgHeight = int((self.maxY - self.minY) * 2)

            # insert the line into the cache map
            self.lineTermPtsCacheMap[termPtKey1] = self.maxLineIdx

            self.maxLineIdx += 1

      return retIdx

   def insertContigSegToIdxMap(self, contigSeg):
      if len(contigSeg.lines) > 0:
         self.contigSegIdxToContigSeg[self.maxContigSegIdx] = contigSeg
         retIdx = self.maxContigSegIdx
         self.maxContigSegIdx += 1
      else:
         retIdx = -1

      return retIdx

   def detectLinesFromCVContours(self, cvContours):
      # opencv contour points are stored in a list
      #   however, each point is stored in a 1-elem list. Thus, need extra 0 indexing
      idx = 0

      for cvContour in cvContours:
         lineIdxs = []
         print("Contour idx is " + str(idx))
         line = lineCls()
         for cvContourPtArray in cvContour:
            cvContourPt = cvContourPtArray[0]
            print("cvContourPt is " + str(cvContourPt))
            if not line.termPt1.any():
               print("line start pt not set - set as " + str(cvContourPt))
               line.setStartPt(cvContourPt)
            else:
               if not line.checkIfLineExt(cvContourPt):
                  line.finalizeLine()
                  startOfNextLine = line.termPt2
                  lineIdx = self.insertLineToIdxMap(line)
                  if lineIdx >= 0:
                     lineIdxs.append(lineIdx)
                  print("line idx " + str(lineIdx) + " is terminated - start new line ")
                  line = lineCls()
             #     line.setStartPt(cvContourPt)
                  line.setStartPt(startOfNextLine)
                  line.midPts.append(cvContourPt)

         if len(line.midPts) > 0:
            line.finalizeLine()
            lineIdx = self.insertLineToIdxMap(line)
            if lineIdx >= 0:
               lineIdxs.append(lineIdx)

         self.lineContourToLineIdxs[idx] = lineIdxs
         idx += 1

   def detectContigSegsFromContours(self):
      contigSeg = contigsSegsCls()
      for idx, contour in self.lineContourToLineIdxs.items():
         print("contigSeg - looking through contour idx " + str(idx) + " with line idxs " + str(contour))
         for lineIdx in contour:
            print("contig seg - line idx is " + str(lineIdx))
            if not contigSeg.insertLineToContigSeg(lineIdx, self.lineIdxToLineMap[lineIdx]):
               contigSeg.finalizeContigSeg()
               retIdx = self.insertContigSegToIdxMap(contigSeg)
               if retIdx < 0:
                  print("contigSeg - failed to insert contigSeg")
               contigSeg = contigsSegsCls(lineIdx, self.lineIdxToLineMap[lineIdx])

   def alignLinesInContour(self):
      for idx, contour in self.lineContourToLineIdxs.items():
       #  for i in range(1, len(contour)):
       #     lineIdx = contour[i]
       #     linePrev = contour[i-1]
       #     self.lineIdxToLineMap[lineIdx].modifyTermPt(self.lineIdxToLineMap[linePrev].termPt2, 1)
         for i in range(0, len(contour)-1):
            lineIdx = contour[i]
            lineNext = contour[i+1]
            self.lineIdxToLineMap[lineIdx].modifyTermPt(self.lineIdxToLineMap[lineNext].termPt1, 2)

   def calcIntersectionOf2Lines(self, line1, line2):
      retDict = (False, np.array([0,0]))

      if(line1.lineType == lineType.VERTICAL):
         line2YCoord = line2.getYGivenX(line1.xMax)
         if(line2YCoord[0] == valType.UNIQUEVAL): # either sloped line or horizontal line
            retDict = (True, np.array([line1.xMax, line2YCoord[1]]))
         elif(line2YCoord[0] == valType.ALLREALVAL): # vertical line with same x value
            retDict = (True, line2.termPt1)

      elif(line1.lineType == lineType.HORIZONTAL):
         line2XCoord = line2.getXGivenY(line1.yMax)
         if(line2XCoord[0] == valType.UNIQUEVAL): # either sloped line or vertical line
            retDict = (True, np.array([line2XCoord[1], line1.yMax]))
         elif(line2XCoord[0] == valType.ALLREALVAL): # horizontal line with same y-value
            retDict = (True, line2.termPt1)

      elif(line2.lineType == lineType.VERTICAL):
         line1YCoord = line1.getYGivenX(line2.xMax)
         if(line1YCoord[0] == valType.UNIQUEVAL): # either sloped line or horizontal line
            retDict = (True, np.array([line2.xMax, line1YCoord[1]]))
         elif(line1YCoord[0] == valType.ALLREALVAL): # vertical line with same x value
            retDict = (True, line2.termPt1)

      elif(line2.lineType == lineType.HORIZONTAL):
         line1XCoord = line1.getXGivenY(line2.yMax)
         if(line1XCoord[0] == valType.UNIQUEVAL): # either sloped line or vertical line
            retDict = (True, np.array([line1XCoord[1], line2.yMax]))
         elif(line1XCoord[0] == valType.ALLREALVAL): # horizontal line with same y-value
            retDict = (True, line2.termPt1)

      else:
         if (line1.lineSlope == line2.lineSlope):
            yCoordProj = line2.getYGivenX(line1.termPt2[0])
            if yCoordProj[1] == line1.termPt2[1]:
               retDict = (True, line2.termPt1)
         else:
            xcommon = (line2.yIntercept - line1.yIntercept) / (line1.lineSlope - line2.lineSlope)
            ycommon = line1.lineSlope * xcommon + line1.yIntercept
            retDict = (True, np.array([xcommon, ycommon]))

      return retDict

   def shouldLineBeInsertedBtwn2Lines(self, line1, line2):
      retDict = (False, None)
      if (line1.lineType == lineType.SLOPED) and \
         (line2.lineType == lineType.SLOPED):
         # check if the the 2 lines have the same sign for slope
         # to do this multiply the slopes from the 2 lines
         #  if both slopes + or -, product is +
         #  else product is -
         prod = line1.lineSlope * line2.lineSlope
         if prod > 0:
            # create candidate line
            lineMid = lineCls()
            lineMid.setStartPt(line1.termPt2)
            lineMid.setEndPt(line2.termPt1)
            lineMid.calcLineMetadata()
            # this candidate line is valid if slope of line is also equal to slope of line 1
            #  and if slope(line1) > slope(line2) > slope(line3) OR
            #      if slope(line1) < slope(line2) < slope(line3)
            if lineMid.lineSlope:
               lineMidSlopeProd = lineMid.lineSlope * line1.lineSlope
               if lineMidSlopeProd > 0:
                  if ((math.fabs(line1.lineSlope) > math.fabs(lineMid.lineSlope)) and \
                      (math.fabs(lineMid.lineSlope) > math.fabs(line2.lineSlope))) or \
                     ((math.fabs(line1.lineSlope) < math.fabs(lineMid.lineSlope)) and \
                      (math.fabs(lineMid.lineSlope) < math.fabs(line2.lineSlope))):
                     retDict = (True, lineMid)

      return retDict

   def checkIfIntersectPtBtwn2Lines(self, pt, line1, line2):
      vect1 = pt - line1.termPt2
      unitVect1 = vect1 / LA.norm(vect1)
      vect2 = line2.termPt1 - pt
      unitVect2 = vect2 / LA.norm(vect2)

      dot1 = np.dot(vect1, line1.unitVect)
      dot2 = np.dot(vect2, line2.unitVect)

      return ((dot1 > 0) and (dot2 > 0))


   def join2LinesInContour(self, line1, line2):
      retTuple = (False, None)
      intersectTuple = self.calcIntersectionOf2Lines(line1, line2)
      if intersectTuple[0]:
         print("line1 and line2 have intersectPt " + str(intersectTuple[1]))
         intersectPt = intersectTuple[1]
         # if intersection pt is greater than line 1 but less than line 2
         if line1.determineIfPtIsWithinLineSpan(intersectPt) or \
            line2.determineIfPtIsWithinLineSpan(intersectPt) or \
            self.checkIfIntersectPtBtwn2Lines(intersectTuple[1], line1, line2):

            line1.setEndPt(intersectPt)
            line1.calcLineMetadata()
            line2.setStartPt(intersectPt)
            line2.calcLineMetadata()
            print("Adjusting end point of line 1 and start point of line 2 to " + str(intersectPt))
      else:
         lineInsertionTuple = self.shouldLineBeInsertedBtwn2Lines(line1, line2)
         if lineInsertionTuple[0]:
            retTuple = lineInsertionTuple
         else:
            print("Adjust end point of line 1 to start point of line 2 " + str(line2.termPt1))
            line1.setEndPt(line2.termPt1)
            line1.calcLineMetadata()

      return retTuple

   def addLineBtwn2Lines(self, line1, line2):
      addLine = lineCls()
      addLine.setStartPt(line1.termPt2)
      addLine.setEndPt(line2.termPt1)
      addLine.calcLineMetadata()
      return addLine

   # check if 2 lines that are contiguous should belong to the same
   # curve -> the criteria for that is that its dot product between
   #  the 2 normal vectors of the 2 lines is greater than a min threshold
   def checkIf2ContiguousLinesBelongToSameCurve(self, line1, line2):
      minDotOf2LinesInCurve = 0.7
      dotProd = np.dot(line1.unitVect, line2.unitVect)
      if dotProd >= minDotOf2LinesInCurve:
         return True

      return False

   # given an array of idxs of the segs of a specific contour,
   #  return the longest contiguous contour
   def extractLongestContigSegFromContour(contour, contourIdxs):
      contourIdxs.sort()
      contigStartIdx = 0
      maxContigLen = 1

      runningStartIdx = 0
      contigLen = 1
      for i in range(0, len(contourIdxs)-1):
         # check if contour line index is sequential
         currContourLineIdx = contour[contourIdxs[i]]
         nextContourLineIdx = contour[contourIdxs[i+1]]
         if currContourLineIdx + 1 == nextContourLineIdx:
            contigLen += 1
         else:
            if contigLen > maxContigLen:
               maxContigLen = contigLen
               contigStartIdx = runningStartIdx

            runningStartIdx = i+1

      retList = contourIdxs[contigStartIdx:contigStartIdx+maxContigLen]
      return retList

   # API to crawl all contours and resolve all parallel lines segs in the same contour
   def crawlAllContoursForSelfContigParallelSegs(self):
      for contourIdx in self.lineContourToLineIdxs:
         self.crawlContoursForContiguousParallelSegs(contourIdx, contourIdx)

   # API to crawl all contours against each other contour NOT itself and resolve all
   # parallel line segs between diff contours
   def crawlAllDiffContoursForParallelSegs(self):
      contourIdxList = [idx for idx in self.lineContourToLineIdxs.keys()]
      for i in range(0, len(contourIdxList)):
         for j in range(i+1, len(contourIdxList)):
            self.analysis_displayParallelLineSegsBtwn2Contours(i, j)

   # crawl 2 contours for contiguous parallel segs
   #  contour1 is the reference contour and contour2 one to crawl in reference to the first one
   #  for each line seg in contour1, add all parallel segs from contour2 to candidate contiguous seg
   #   once contour1 breaks, create longest contiguous seg from contour2 candidates
   #     then, compare the 2 contiguous segments and choose the one with more line segs (more details)
   #  contour1 breaks on the following conditions:
   #    1) if contour1 reaches line already crawled (either by contour1 or contour2 since this API can handle contour1 == contour2
   #            in order to crawl own contour for parallel segs)
   #    2) if closest seg is NOT contiguous
   #    3) if NO parallel corresponding line seg from contour2 THAT HAS NOT ALREADY BEEN PROCESSED is found for the current line seg on contour1
   #
   def crawlContoursForContiguousParallelSegs(self, contour1Idx, contour2Idx, delContour1):

      contour1 = self.lineContourToLineIdxs.get(contour1Idx, None)
      contour2 = self.lineContourToLineIdxs.get(contour2Idx, None)

      if not contour1 or not contour2:
         print("Failed to get either contour1 with idx " + str(contour1Idx) + " or contour2 with idx " + str(contour2Idx))
         return

      print("process parallel segs of contour idx " + str(contour1Idx) + " and contour idx " + str(contour2Idx))
      contour1LinesToDelete = []
      contour2LinesToDelete = []

      refContourProcessing = []
      secCorrespondingContourProcessing = []

      linesAlreadyProcessed = []
      secLinesAlreadyProcessed = []
      secLinesAlreadyMarkedForDeletion = []
      refLinesAlreadyMarkedForDeletion = []

      for i in range(0, len(contour1)):
        # if (contour1[i] in linesAlreadyProcessed) or contour1[i] in secLinesAlreadyMarkedForDeletion or contour1[i] in secLinesAlreadyProcessed:
         if (contour1[i] in linesAlreadyProcessed) or contour1[i] in secLinesAlreadyMarkedForDeletion:
            print("contour1 line i " + str(i) + " with line index " + str(contour1[i]) + " is already processed")
            # handle the 2 processing parallel contours to see if which has more contours
            secCorrespondingContourProcessing.sort()
            refContourProcessing.sort()
            print("refContourProcessing to be deleted is " + str([contour1[k] for k in refContourProcessing]))
            print("secCorrespondingContourProcessing to be deleted is " + str([contour2[k] for k in secCorrespondingContourProcessing]))
           # if len(secCorrespondingContourProcessing) < len(refContourProcessing):
            if not delContour1:
               contour2LinesToDelete.extend(secCorrespondingContourProcessing)
               secLinesAlreadyMarkedForDeletion.extend([contour2[k] for k in secCorrespondingContourProcessing])
            else:
               contour1LinesToDelete.extend(refContourProcessing)

            refContourProcessing = []
            secCorrespondingContourProcessing = []
            continue

         line1 = self.lineIdxToLineMap.get(contour1[i], None)
         if not line1:
            continue
         if refContourProcessing:
            prevLine1 = self.lineIdxToLineMap.get(contour1[refContourProcessing[-1]],None)
            if not checkIf2LinesAreContiguous(prevLine1, line1) or \
               not self.checkIf2ContiguousLinesBelongToSameCurve(prevLine1, line1):
               # handle the 2 processing parallel contours to see if which has more contours
               refContourProcessing.sort()
             #  print("refContourProcessing to be deleted is " + str(refContourProcessing))
               print("refContourProcessing to be deleted is " + str([contour1[k] for k in refContourProcessing]))
               secCorrespondingContourProcessing.sort()
               print("secCorrespondingContourProcessing to be deleted is " + str([contour2[k] for k in secCorrespondingContourProcessing]))
            #   if len(secCorrespondingContourProcessing) < len(refContourProcessing):
               if not delContour1:
                  contour2LinesToDelete.extend(secCorrespondingContourProcessing)
                  secLinesAlreadyMarkedForDeletion.extend([contour2[k] for k in secCorrespondingContourProcessing])
               else:
                  contour1LinesToDelete.extend(refContourProcessing)

               refContourProcessing = []
               secCorrespondingContourProcessing = []

         foundParallelLine = False
         for j in range(0, len(contour2)):
            line2 = self.lineIdxToLineMap.get(contour2[j], None)
            if not line2:
               continue
            if (line1 == line2) or contour2[j] in linesAlreadyProcessed or contour2[j] in contour2LinesToDelete:
              # checkIf2LinesAreContiguous(line1, line2) or \
              # contour2[j] in linesAlreadyProcessed:
               continue

            lineToRemove = self.checkIf2LinesRedundant(line1, line2)
            if lineToRemove:
               if j not in secCorrespondingContourProcessing:
                  secCorrespondingContourProcessing.append(j)
                  foundParallelLine = True
                  secLinesAlreadyProcessed.append(contour2[j])
               if i not in refContourProcessing:
                  refContourProcessing.append(i)

         if not foundParallelLine:
            print("not found parallel line to process for i " + str(i) + " and j " + str(j))
            print("contour1 idx " + str(contour1Idx) + " to delete are " + str(contour1LinesToDelete))
            print("refContourProcessing is " + str(refContourProcessing))
            # handle the 2 processing parallel contours to see if which has more contours
            refContourProcessing.sort()
            print("refContourProcessing to be deleted is " + str([contour1[k] for k in refContourProcessing]))
            secCorrespondingContourProcessing.sort()
            print("secCorrespondingContourProcessing to be deleted is " + str([contour2[k] for k in secCorrespondingContourProcessing]))
           # if len(secCorrespondingContourProcessing) < len(refContourProcessing):
            if not delContour1:
               contour2LinesToDelete.extend(secCorrespondingContourProcessing)
               print("contour2 idx " + str(contour2Idx) + " to delete are " + str(contour2LinesToDelete))
               secLinesAlreadyMarkedForDeletion.extend([contour2[k] for k in secCorrespondingContourProcessing])
            else:
               print("contour1 idx " + str(contour1Idx) + " to delete are " + str(contour1LinesToDelete))
               contour1LinesToDelete.extend(refContourProcessing)

            refContourProcessing = []
            secCorrespondingContourProcessing = []
         else:
            print("found parallel lines to process at i " + str(i))
            print("contour1 idx " + str(contour1Idx) + " to delete are " + str(contour1LinesToDelete))
            print("contour2 idx " + str(contour2Idx) + " to delete are " + str(contour2LinesToDelete))
            print("refContourProcessing is " + str(refContourProcessing))
            print("secCorrespondingContourProcessing is " + str(secCorrespondingContourProcessing))

         linesAlreadyProcessed.append(contour1[i])
        # refContourProcessing.append(i)

      print("contour1 idx " + str(contour1Idx) + " to delete are " + str(contour1LinesToDelete))
      print("contour2 idx " + str(contour2Idx) + " to delete are " + str(contour2LinesToDelete))

      # delete any contour1 lines
      linesInContour1AfterPruning = []

   #   if contour1Idx == contour2Idx:
   #      contour1LinesToDelete.extend(contour2LinesToDelete)
   #      contour1LinesToDelete = list(set(contour1LinesToDelete))
   #      contour2LinesToDelete = []

      if contour1LinesToDelete:
         contour1LinesToDelete = list(set(contour1LinesToDelete))
         contour1LinesToDelete.sort()
         contour1lineIdxToDelete = contour1LinesToDelete.pop(0)

         for line1Index in range(0, len(contour1)):
            if line1Index == contour1lineIdxToDelete:
               if contour1LinesToDelete:
                  contour1lineIdxToDelete = contour1LinesToDelete.pop(0)
               else:
                  linesInContour1AfterPruning.extend(contour1[line1Index+1:])
                  break
            else:
               linesInContour1AfterPruning.append(contour1[line1Index])

   #      self.lineContourToLineIdxs[contour1Idx] = linesInContour1AfterPruning

      # delete any contour2 lines
      linesInContour2AfterPruning = []
      if contour2LinesToDelete:
         contour2LinesToDelete = list(set(contour2LinesToDelete))
         contour2LinesToDelete.sort()
         contour2lineIdxToDelete = contour2LinesToDelete.pop(0)

         for line2Index in range(0, len(contour2)):
            if line2Index == contour2lineIdxToDelete:
               if contour2LinesToDelete:
                  contour2lineIdxToDelete = contour2LinesToDelete.pop(0)
               else:
                  linesInContour2AfterPruning.extend(contour2[line2Index+1:])
                  break
            else:
               linesInContour2AfterPruning.append(contour2[line2Index])

  #       self.lineContourToLineIdxs[contour2Idx] = linesInContour2AfterPruning

      if delContour1:
         if len(linesInContour1AfterPruning) > 0:
            contourToReturn = linesInContour1AfterPruning
         else:
            contourToReturn = contour1
      else:
         if len(linesInContour2AfterPruning) > 0:
            contourToReturn = linesInContour2AfterPruning
         else:
            contourToReturn = contour2

      return contourToReturn

   # given a contour of line idxs generate the contig segs for said contour
   def generateContigSegsFromContour(self, contour):
      alreadyInContigSeg = []
      retContigSegs = []
      for i in range(len(contour)):
         if i not in alreadyInContigSeg:
            alreadyInContigSeg.append(i)
            contigSegCandidate = contigsSegsCls(contour[i], self.lineIdxToLineMap[contour[i]])
            for j in range(i+1, len(contour)):
               if j not in alreadyInContigSeg:
                  if contigSegCandidate.insertLineToContigSeg(contour[j], self.lineIdxToLineMap[contour[j]]):
                     alreadyInContigSeg.append(j)
            contigSegCandidate.finalizeContigSeg()
            retContigSegs.append(contigSegCandidate)

      return retContigSegs

   # This API takes as input 2 contours -> these contours must be of the same
   #  contour as crawled by cv.findContours
   #  The reason why there are 2 contours from the same contour is that
   #    the same contour may have parallel contours because of the way contours are crawled and generated

   #  The API crawlContoursForContiguousParallelSegs above iterates through the contour with 2 contour iterators
   #    and thus a set of 2 parallel segs are detected -> contour1 and contour2
   #  Depending on the input flag passed into the API, contour1 may be deleted or contour2
   #   Thus, the input contour1 is if contour2 is deleted, and contour2 is if contour1 is deleted
   def mergeContour1AndContour2OfSameContour(self, contourIdx, contour1, contour2):
   #   print("processing contourIdx " + str(contourIdx))
      # crawl contour1 for contig segs
      print("contourIdx " + str(contourIdx) + ": contour1 is " + str(contour1) + " contour2 is " + str(contour2))
      contour1ContigSegs = self.generateContigSegsFromContour(contour1)
      print("contourIdx " + str(contourIdx) + " contour1ContigSegs are: ")
      for idx in range(len(contour1ContigSegs)):
         print("contig1 idx " + str(idx))
         contour1ContigSegs[idx].printContigSegInfo()

      c1contigSegsLen = 0
      for contigSeg in contour1ContigSegs:
         c1contigSegsLen += contigSeg.length

      c1AvgContigSegLen = c1contigSegsLen / len(contour1ContigSegs)

      # crawl contour2 for contig segs
      contour2ContigSegs = self.generateContigSegsFromContour(contour2)
      print("contourIdx " + str(contourIdx) + " contour2ContigSegs are: ")
      for idx in range(len(contour2ContigSegs)):
         print("contig2 idx " + str(idx))
         contour2ContigSegs[idx].printContigSegInfo()

      c2contigSegsLen = 0
      for contigSeg in contour2ContigSegs:
         c2contigSegsLen += contigSeg.length

      c2AvgContigSegLen = c2contigSegsLen / len(contour2ContigSegs)

      # sort the contour 1 and contour 2 contig segs list from longest to shortest contig segs
      contour1ContigSegs.sort(key=contigSegLenSort, reverse=True)
      contour2ContigSegs.sort(key=contigSegLenSort, reverse=True)

      if c1AvgContigSegLen > c2AvgContigSegLen:
         refContourContigSegs = contour1ContigSegs
         otherContourContigSegs = contour2ContigSegs
      else:
         refContourContigSegs = contour2ContigSegs
         otherContourContigSegs = contour1ContigSegs

      # merge the other contour (the one with lower avg contigSeg length)
      # into each contig seg in the ref contour (the one with the higher avg contigSeg length)
      retContigSegs = self.mergeSecContourToRef(refContourContigSegs, otherContourContigSegs)

      return retContigSegs

   def mergeSecContourToRef(self, refContourContigSegs, otherContourContigSegs):
      # this merge contigs map contains the following
      # key is list of integers [p1, s1, s2, s3] - where p1 is the idx of the contig
      #                                            seg in regContourContigSegs
      #                                          - s1 is the sec contour that was joined with
      #                                            with the ref contour
      #  when we loop through the sec contour contig segs - check if the ref contour is already merged with another
      #   sec contour - if yes, it may have shifted its position as well as end points - use the merged ref contig seg
      #
      #  when we loop through the ref contour contig segs - check if the secondary contour is already merged with a previous
      #   ref contour - if yes use the one that was merged with the ref contour
      #
      # value is list of contig segs in order that correspond to s1, s2, s3 that are merged with p1
      mergedContigsMap = {}

      # map of ref / sec contig segs contour idx to the contigseg itself
      # only push into ref processed contigs if:
      # for both -> store into map if ref or sec completely overlaps or completely merges -> store result in both
      #  for ref -> either processed ALL sec contig segs and found NONE that should be merged / overlapped
      #          -> or found ref with sec that overlaps - stored the new shifted ref
      #  for sec -> either merge with ref count -> shift depends on
      refProcessedContigs = {}
      secProcessedContigs = {}

      # need to have 2 maps to keep track of how much the refContour contig segs moved and the secContour contig segs moved
      #  key = idx of contig seg for ref or sec contour
      #  value = np_array of shift that occurred
      #  when comparing the contig segs - if contig seg already found in either the ref or sec contigs map as noted above
      #  need to shift back to orig position to see if they are parallel and should be merged
      #    if they should be merged - shift the shorter contigseg
      refContigSegIdxDispMap = {}
      secContigsegidxToDispMap = {}

      # try using just one map
      contigSegIdxToContigSeg = {}
      newContigSegIdxToContigSeg = {}

      # initial setup - put refContourContigSegs and otherContourContigSegs into contigSegIdxToContigSeg
      for refIdx in range(len(refContourContigSegs)):
         contigSegIdxToContigSeg[refIdx] = refContourContigSegs[refIdx]
      for secIdx in range(len(otherContourContigSegs)):
         contigSegIdxToContigSeg[len(refContourContigSegs) + secIdx] = otherContourContigSegs[secIdx]

      # first version - limit the number of iterations so we don't run into infinite loop
      firstIterMax = 1
      iterCount = 0

      madeChanges = True

      while iterCount < firstIterMax and madeChanges:
         madeChanges = False
         print("iteration number " + str(iterCount + 1))

         secAlreadyDetermineIgnore = []

         addNewIdxToContigSegMap = len(contigSegIdxToContigSeg)

         for refIdx in range(len(contigSegIdxToContigSeg)):

            if refIdx in secAlreadyDetermineIgnore:
               continue

            breakRefAndDoNotStore = False
            for secIdx in range(refIdx+1, len(contigSegIdxToContigSeg)):

               print("refIdx is " + str(refIdx) + " secIdx is " + str(secIdx))

               # try to get from newContigSeg since modified lines will be inserted into newContigSegIdxMap
               refContig = copy.deepcopy(newContigSegIdxToContigSeg.get(refIdx, None))
               secContig = copy.deepcopy(newContigSegIdxToContigSeg.get(secIdx, None))

               if not refContig:
                  refContig = copy.deepcopy(contigSegIdxToContigSeg[refIdx])
               if not secContig:
                  secContig = copy.deepcopy(contigSegIdxToContigSeg[secIdx])

               print("refContig idx " + str(refIdx) + " info ")
               refContig.printContigSegInfo()
               print("secContig idx " + str(secIdx) + " info ")
               secContig.printContigSegInfo()

               refContigOverlapIdxs = []
               secContigOverlapIdxs = []

               refAndSecOverlapSegs = []
               isSecContigReverse = False

               secAllContigOverlapIdxs = []

               for i in range(len(refContig.lines)):
                  foundParallel = False
                  for j in range(len(secContig.lines)):

                     print("refContig line index " + str(i) + " has line idx " + str(refContig.lineIdxs[i]))
                     print("refContig line info ")
                     refContig.lines[i].displayLineInfo()

                     print("secContig line index " + str(j) + " has line idx " + str(secContig.lineIdxs[j]))
                     print("secContig line info ")
                     secContig.lines[j].displayLineInfo()

                     lineToRemove = self.checkIf2LinesRedundant(refContig.lines[i], secContig.lines[j], 5)
                     if lineToRemove:
                        if not foundParallel:
                           refContigOverlapIdxs.append(i)
                           foundParallel = True
                        if j not in secContigOverlapIdxs:
                           secContigOverlapIdxs.append(j)

                  if not foundParallel or \
                     ((i == len(refContig.lines)-1) and (j == len(secContig.lines)-1)):

                     if len(refContigOverlapIdxs) > 0 and \
                        len(secContigOverlapIdxs) > 0:

                        refAndSecContigOverlapCompletely = False

                        # check if the refConfig and secContig are that are detected as overlap
                        #  are actually the COMPLETELY OVERLAPPED lines
                        if len(refContigOverlapIdxs) == len(secContigOverlapIdxs):
                           secCheckOverlap = secContigOverlapIdxs[:]
                           secCheckOverlap.sort()
                           if secCheckOverlap != secContigOverlapIdxs:
                              secCheckOverlap.sort(reverse=True)
                           print("processing refContigOverlapIdxs " + str(refContigOverlapIdxs) + " and secContigOverlapIdxs " + str(secContigOverlapIdxs))
                           # create a contigSeg with refContigOverlapIdxs
                           refCheckOverlapContigSeg = contigsSegsCls()
                           print("generating refCheckOverlapContigSeg ")
                           for refContigOverlapIdx in refContigOverlapIdxs:
                              refCheckOverlapContigSeg.insertLineToContigSeg(refContig.lineIdxs[refContigOverlapIdx], refContig.lines[refContigOverlapIdx], False)
                           secCheckOverlapContigSeg = contigsSegsCls()
                           print("generating secCheckOverlapContigSeg ")
                           for secContigOverlapIdx in secCheckOverlap:
                              secCheckOverlapContigSeg.insertLineToContigSeg(secContig.lineIdxs[secContigOverlapIdx], secContig.lines[secContigOverlapIdx], False)

                           if checkIfOneContigCompletelyOverlap(refCheckOverlapContigSeg, secCheckOverlapContigSeg):
                              print("refCheckOverlapContigSeg and secCheckOverlapContigSeg completely overlaps")
                              refAndSecContigOverlapCompletely = True
                           else:
                              print("refCheckOverlapContigSeg and secCheckOverlapContigSeg NOT completely overlaps")

                        if not refAndSecContigOverlapCompletely:
                           # check if sec contig overlap idxs is sorted ->
                           # if not this means that the orientations of the lines are
                           # running opposite to each other and that the sec contour must be reverse sorted
                           # since ref contour is sorted in ascending order
                           secContigCopy = secContigOverlapIdxs[:]
                           secContigCopy.sort()
                           if secContigCopy != secContigOverlapIdxs:
                              secContigOverlapIdxs.sort(reverse=True)
                              isSecContigReverse = True
                           # insert the start and end idx of both the ref and the sec parallel lines
                           #  the tuple is (ref start line idx, ref end line idx, sec start line idx, sec end line idx)
                           #  NOTE: the idx here is in reference to the idx of the line in contigSeg.lines list
                           #  the sec start line idx may be greater than sec end line idx -> this is if the
                           #  sec contour contig seg is oriented in opposite direction to ref contour contig seg
                           refAndSecOverlapSegs.append([refContigOverlapIdxs[0], refContigOverlapIdxs[-1], secContigOverlapIdxs[0], secContigOverlapIdxs[-1]])
                           print("refAndSecOverlapSegs is " + str(refAndSecOverlapSegs))
                           print("refContigOverlapIdxs is " + str(refContigOverlapIdxs) + " secContigOverlapIdxs is " + str(secContigOverlapIdxs))

                           secAllContigOverlapIdxs.extend(secContigOverlapIdxs)

                        refContigOverlapIdxs = []
                        secContigOverlapIdxs = []

               # check if secContig is reverse from the ref - this can happen in 2 cases
               # 1) the secStartIdx, secEndIdx is flipped (out of order) in the refAndSecOverlapSegs tuple
               # 2) the secStartIdx, secEndIdx is in order BUT the (secStartIdx, secEndIdx) of entry x+1 is less than entry x
               if len(secAllContigOverlapIdxs) > 0:
                  print("secAllContigOverlapIdxs is " + str(secAllContigOverlapIdxs))
                  # check if sec contig overlap idxs is sorted ->
                  # if not this means that the orientations of the lines are
                  # running opposite to each other and that the sec contour must be reverse sorted
                  # since ref contour is sorted in ascending order
                #  secAllContigOverlapIdxs = list(set(secAllContigOverlapIdxs))
                  secAllContigCopy = secAllContigOverlapIdxs[:]
                  secAllContigCopy.sort()
                  if secAllContigCopy != secAllContigOverlapIdxs:
                     isSecContigReverse = True
                     # if the entries secStartIdx < secEndIdx for each tuple of refAndSecOverlapSegs termPts, reverse them
                     for refAndSecEntry in refAndSecOverlapSegs:
                        if refAndSecEntry[2] < refAndSecEntry[3]:
                           tmp = refAndSecEntry[2]
                           refAndSecEntry[2] = refAndSecEntry[3]
                           refAndSecEntry[3] = tmp

               # need to check if the parallel segs are actually the same segs (ie. this is the case only if
               #  the segments have been processed
               if refAndSecOverlapSegs:
                  diffFound = False
                  print("refAndSecOverlapSegs - check if the overlap segs are actually the same line segments")

                  for overlapSeg in refAndSecOverlapSegs:

                     refStartIdx = overlapSeg[0]
                     refEndIdx = overlapSeg[1]
                     secStartIdx = overlapSeg[2]
                     secEndIdx = overlapSeg[3]
                     refNumLines = refEndIdx - refStartIdx + 1
                     secNumLines = math.fabs(secEndIdx - secStartIdx) + 1
                     if refNumLines == secNumLines:
                        if refNumLines == 1:
                           refIdxIncrement = secIdxIncrement = 0
                        else:
                           refIdxIncrement = (refEndIdx - refStartIdx) / (math.fabs(refEndIdx - refStartIdx))
                           secIdxIncrement = (secEndIdx - secStartIdx) / (math.fabs(secEndIdx - secStartIdx))

                        for increment in range(refNumLines):
                           refLineIdx = int(refStartIdx + (increment * refIdxIncrement))
                           secLineIdx = int(secStartIdx + (increment * secIdxIncrement))
                           print("refLine with idx " + str(refLineIdx) + " has info ")
                           print(refContig.lines[refLineIdx].displayLineInfo())
                           print("secLine with idx " + str(secLineIdx) + " has info ")
                           print(secContig.lines[secLineIdx].displayLineInfo())
                           if refContig.lines[refLineIdx] != secContig.lines[secLineIdx]:
                              diffFound = True
                              break
                     else:
                        diffFound = True
                        break

                  if not diffFound:
                     refAndSecOverlapSegs = []

               if refAndSecOverlapSegs:
                  madeChanges = True
                  print("handling refAndSecOverlapSegs " + str(refAndSecOverlapSegs))
                  print("refContig idx " + str(refIdx) + " has line idxs " + str(refContig.lineIdxs))
                  print("secContig idx " + str(secIdx) + " has line idxs " + str(secContig.lineIdxs))
                  # handle the 2 parallel contig segs
                  # use the longer contig seg as the new anchor
                  if refContig.length > secContig.length:
                     refIsAnchor = True
                  else:
                     refIsAnchor = False

                  # need to make sure secContig idxs are in order and that the secContig idxs are consecutive
                  # and contiguous
                  # if they are NOT:
                  #   1) if the one before COMPLETELY overlaps the one after, remove the one after
                  #   2) if the one before partially overlaps the one after, change the bound of the one after so that
                  #      its start idx is 1+end idx of the one before -> need to recheck to see which of the existing refContigs
                  #      still match the one with the new bounds
                  # in the secContig start / end idx -> it has been observed that the secContig startIdx, endIdx
                  # of the preceeding entry somewhat eclipses the succeeding entry
                  # eg. [[3, 4, 2, 1], [9, 11, 3, 0]]
                  if refIsAnchor:
                     if isSecContigReverse:
                        secStartIdx = 3
                        secEndIdx = 2
                     else:
                        secStartIdx = 2
                        secEndIdx = 3

                     print("refIsAnchor - refAndOverlapSegs is " + str(refAndSecOverlapSegs))

                     sortedRefAndSecOverlapSegs = sorted(refAndSecOverlapSegs, key=itemgetter(secStartIdx, secEndIdx))
                     newRefAndSecOverlapSegs = [sortedRefAndSecOverlapSegs[0]]
                     for i in range(1, len(sortedRefAndSecOverlapSegs)):
                        # check if end idx of the prev is greater than the start idx of the current entry
                        if sortedRefAndSecOverlapSegs[i][secStartIdx] <= newRefAndSecOverlapSegs[-1][secEndIdx]:
                           # check if the end idx current entry is less than the end idx of the prev entry - if it is - exclude this entry
                           # otherwise - adjust the secStartIdx of the current so that it is 1+secEndIdx of the prev
                           print("refIsAnchor - refAndOverlapSegs with idx " + str(i) + " is " + str(sortedRefAndSecOverlapSegs[i]) + " - its secStartIdx " + str(sortedRefAndSecOverlapSegs[i][secStartIdx]) + " is smaller than secEndIdx of prev entry " + str(newRefAndSecOverlapSegs[-1][secEndIdx]))

                           if sortedRefAndSecOverlapSegs[i][secEndIdx] > newRefAndSecOverlapSegs[-1][secEndIdx]:
                              print("refIsAnchor - refAndOverlapSegs with idx " + str(i) + " is " + str(sortedRefAndSecOverlapSegs[i]) + " - its secEndIdx " + str(sortedRefAndSecOverlapSegs[i][secEndIdx]) + " is greater than secEndIdx of prev entry " + str(newRefAndSecOverlapSegs[-1][secEndIdx]) + " - modify the secStartIdx so that it is 1 + prev secEndIdx")
                              sortedRefAndSecOverlapSegs[i][secStartIdx] = newRefAndSecOverlapSegs[-1][secEndIdx] + 1
                              # now that the sec range has been adjusted - recheck the ref range to see if lines for ref still overlap with sec
                              refs = []
                              for j in range(sortedRefAndSecOverlapSegs[i][0], sortedRefAndSecOverlapSegs[i][1]+1):
                                 for k in range(sortedRefAndSecOverlapSegs[i][secStartIdx], sortedRefAndSecOverlapSegs[i][secEndIdx]+1):
                                    lineToRemove = self.checkIf2LinesRedundant(refContig.lines[j], secContig.lines[k])
                                    if lineToRemove and j not in refs:
                                       refs.append(j)
                                       break
                              # sort the refs and get the ref min / max point to get the new range
                              refs = refs.sort()
                              sortedRefAndSecOverlapSegs[i][0] = refs[0]
                              sortedRefAndSecOverlapSegs[i][1] = refs[-1]
                              newRefAndSecOverlapSegs.append(sortedRefAndSecOverlapSegs[i])
                        else:
                           newRefAndSecOverlapSegs.append(sortedRefAndSecOverlapSegs[i])

                     refAndSecOverlapSegs = newRefAndSecOverlapSegs
                     print("refIsAnchor - final refAndOverlapSegs is " + str(refAndSecOverlapSegs))

                  # generate contig seg from ref contour first
                  # rolling index refI
                  newRefContigSegCorrespIdxInserted = False
                  newRefContigSeg = None
                  refI = 0
                  if not refIsAnchor:
                     newRefContigSeg = contigsSegsCls()
                     for overlapSeg in refAndSecOverlapSegs:
                        print("refContig - handle overlapSeg " + str(overlapSeg))
                        # first set the lines preceding the start of the overlap
                        for rindex in range(refI, overlapSeg[0]):
                           # if this is the final line before segment where the 2 contours overlap
                           # need to adjust the end point IF sec contour contig seg is longer
                           print("refContig - insert line idx " + str(refContig.lineIdxs[rindex]))
                           #if rindex == overlapSeg[0] - 1:
                              # need to check if sec contour is in opposite direction to ref contour
                           #   if isSecContigReverse:
                               #  refContig.lines[rindex].setEndPt(secContig.lines[overlapSeg[2]].termPt2)
                               #  print("refContig - adjusting line idx " + str(refContig.lineIdxs[rindex]) + " with start pt " + str(refContig.lines[rindex].termPt1) + " -> end pt to " + str(secContig.lines[overlapSeg[2]].termPt2))
                               # adjust future overlapSeg term pt to refContig.lines[rindex] end point
                           #   else:
                               #  refContig.lines[rindex].setEndPt(secContig.lines[overlapSeg[2]].termPt1)
                               #  print("refContig - adjusting line idx " + str(refContig.lineIdxs[rindex]) + " with start pt " + str(refContig.lines[rindex].termPt1) + " -> end pt to " + str(secContig.lines[overlapSeg[2]].termPt1))

                           newRefContigSeg.insertLineToContigSeg(refContig.lineIdxs[rindex], refContig.lines[rindex], False)
                          # if not newRefContigSeg.insertLineToContigSeg(refContig.lineIdxs[rindex], refContig.lines[rindex], False):
                          #    newRefContigSeg.finalizeContigSeg()
                          #    if not newRefContigSegCorrespIdxInserted:
                          #       newContigSegIdxToContigSeg[refIdx] = newRefContigSeg
                          #       newRefContigSegCorrespIdxInserted = True
                          #       newRefContigSeg = contigsSegsCls(refContig.lineIdxs[rindex], refContig.lines[rindex])
                          #    else:
                          #       newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newRefContigSeg
                          #       addNewIdxToContigSegMap += 1
                          #       newRefContigSeg = contigsSegsCls(refContig.lineIdxs[rindex], refContig.lines[rindex])

                        # need to make sure that the end point of the last line in the newSecContigSeg is equal to the start pt of the refOverlapstartIdx
                        # Aug 2020 - enhanced -> try to get the intersecPt between the last line of newRefContigSeg and the beginning of secContig overlapSeg
                        #                        and modify both the newRefContigSeg and the connecting point (end pt or start pt depending on whether the secContig is reversed)
                        #                        to the intersect pt: if the intersect pt does NOT lie on the overlap line in the secContig seg -> move the connecting point
                        #                        to the end point of the newRefContigSeg
                        if len(newRefContigSeg.lines) > 0:
                           xsectPt = getXSectPtBtwn2LinesCheckIfInRefLine(secContig.lines[overlapSeg[2]], newRefContigSeg.lines[-1])
                           if isSecContigReverse:
                              if xsectPt.all():
                                 print("refContig -> intersect pt " + str(xsectPt) + " exists between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and secContig overlap line with start pt " + str(secContig.lines[overlapSeg[2]].termPt1) + " end pt " + str(secContig.lines[overlapSeg[2]].termPt2))
                                 newRefContigSeg.lines[-1].setEndPt(xsectPt)
                                 secContig.lines[overlapSeg[2]].setEndPt(xsectPt)
                              else:
                                 print("refContig -> intersect pt does not exist between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and secContig overlap line with start pt " + str(secContig.lines[overlapSeg[2]].termPt1) + " end pt " + str(secContig.lines[overlapSeg[2]].termPt2))
                                 #newRefContigSeg.lines[-1].setEndPt(secContig.lines[overlapSeg[2]].termPt2)
                                 secContig.lines[overlapSeg[2]].setEndPt(newRefContigSeg.lines[-1].termPt2)
                           else:
                              if xsectPt.all():
                                 print("refContig - reverse -> intersect pt " + str(xsectPt) + " exists between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and secContig overlap line with start pt " + str(secContig.lines[overlapSeg[2]].termPt1) + " end pt " + str(secContig.lines[overlapSeg[2]].termPt2))
                                 newRefContigSeg.lines[-1].setEndPt(xsectPt)
                                 secContig.lines[overlapSeg[2]].setStartPt(xsectPt)
                              else:
                                 print("refContig - reverse -> intersect pt does not exist between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and secContig overlap line with start pt " + str(secContig.lines[overlapSeg[2]].termPt1) + " end pt " + str(secContig.lines[overlapSeg[2]].termPt2))
                                 #newRefContigSeg.lines[-1].setEndPt(secContig.lines[overlapSeg[2]].termPt1)
                                 secContig.lines[overlapSeg[2]].setStartPt(newRefContigSeg.lines[-1].termPt2)

                        # now set the overlap portion
                        print("refContig - handling overlap portion")
                        if isSecContigReverse:
                           # need to go in reverse order and flip line
                           for oindex in range(overlapSeg[2], overlapSeg[3]-1, -1):
                              print("refContig - reverse adding line with idx " + str(secContig.lineIdxs[oindex]))
                              overlapLine = copy.deepcopy(secContig.lines[oindex])
                              overlapLine.flipLine()

                              newRefContigSeg.insertLineToContigSeg(secContig.lineIdxs[oindex], overlapLine, False)
                          #    if not newRefContigSeg.insertLineToContigSeg(secContig.lineIdxs[oindex], overlapLine, False):
                          #       newRefContigSeg.finalizeContigSeg()
                          #       if not newRefContigSegCorrespIdxInserted:
                          #          newContigSegIdxToContigSeg[refIdx] = newRefContigSeg
                          #          newRefContigSegCorrespIdxInserted = True
                          #          newRefContigSeg = contigsSegsCls(secContig.lineIdxs[oindex], overlapLine)
                          #       else:
                          #          newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newRefContigSeg
                          #          addNewIdxToContigSegMap += 1
                          #          newRefContigSeg = contigsSegsCls(secContig.lineIdxs[oindex], overlapLine)

                           # adjust the future NON overlap portion
                           # Aug 2020 -> again use the intersect pt between the last line of newRefContigSeg and what the potential net line can be
                           if overlapSeg[1] < len(refContig.lines)-1:
                              xsectPt = getXSectPtBtwn2LinesCheckIfInRefLine(newRefContigSeg.lines[-1], refContig.lines[overlapSeg[1]+1])
                              if xsectPt.all():
                                 print("refContig - reverse -> intersect pt " + str(xsectPt) + " exists between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and next refContig line with start pt " + str(refContig.lines[overlapSeg[1]+1].termPt1) + " end pt " + str(refContig.lines[overlapSeg[1]+1].termPt2))
                                 newRefContigSeg.lines[-1].setEndPt(xsectPt)
                                 refContig.lines[overlapSeg[1]+1].setStartPt(xsectPt)
                              else:
                                 print("refContig - reverse -> intersect pt does not exist between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and next refContig line with start pt " + str(refContig.lines[overlapSeg[1]+1].termPt1) + " end pt " + str(refContig.lines[overlapSeg[1]+1].termPt2))
                                 #newRefContigSeg.lines[-1].setEndPt(secContig.lines[overlapSeg[2]].termPt1)
                                 newRefContigSeg.lines[-1].setEndPt(refContig.lines[overlapSeg[1]+1].termPt1)
                        #      refContig.lines[overlapSeg[1]+1].setStartPt(secContig.lines[overlapSeg[3]].termPt1)
                        #      print("refContig - reverse future adjusting line idx " + str(refContig.lineIdxs[overlapSeg[1]+1]) + " start pt to " + str(refContig.lines[overlapSeg[1]+1].termPt1) + " -> end pt is " + str(refContig.lines[overlapSeg[1]+1].termPt2))
                        else:
                           for oindex in range(overlapSeg[2], overlapSeg[3]+1):
                              print("refContig - adding line with idx " + str(secContig.lineIdxs[oindex]))
                              overlapLine = copy.deepcopy(secContig.lines[oindex])

                              newRefContigSeg.insertLineToContigSeg(secContig.lineIdxs[oindex], overlapLine, False)
                              #if not newRefContigSeg.insertLineToContigSeg(secContig.lineIdxs[oindex], overlapLine, False):
                              #   newRefContigSeg.finalizeContigSeg()
                              #   if not newRefContigSegCorrespIdxInserted:
                              #      newContigSegIdxToContigSeg[refIdx] = newRefContigSeg
                              #      newRefContigSegCorrespIdxInserted = True
                              #      newRefContigSeg = contigsSegsCls(secContig.lineIdxs[oindex], overlapLine)
                              #   else:
                              #      newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newRefContigSeg
                              #      addNewIdxToContigSegMap += 1
                              #      newRefContigSeg = contigsSegsCls(secContig.lineIdxs[oindex], overlapLine)

                           # adjust the future NON overlap portion
                           if overlapSeg[1] < len(refContig.lines)-1:
                              xsectPt = getXSectPtBtwn2LinesCheckIfInRefLine(newRefContigSeg.lines[-1], refContig.lines[overlapSeg[1]+1])
                              if xsectPt.all():
                                 print("refContig -> intersect pt " + str(xsectPt) + " exists between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and next refContig line with start pt " + str(refContig.lines[overlapSeg[1]+1].termPt1) + " end pt " + str(refContig.lines[overlapSeg[1]+1].termPt2))
                                 newRefContigSeg.lines[-1].setEndPt(xsectPt)
                                 refContig.lines[overlapSeg[1]+1].setStartPt(xsectPt)
                              else:
                                 print("refContig -> intersect pt does not exist between refContig line with start pt " + str(newRefContigSeg.lines[-1].termPt1) + " end pt " + str(newRefContigSeg.lines[-1].termPt2) + " and next refContig line with start pt " + str(refContig.lines[overlapSeg[1]+1].termPt1) + " end pt " + str(refContig.lines[overlapSeg[1]+1].termPt2))
                                 #newRefContigSeg.lines[-1].setEndPt(secContig.lines[overlapSeg[2]].termPt1)
                                 newRefContigSeg.lines[-1].setEndPt(refContig.lines[overlapSeg[1]+1].termPt1)
                            #  refContig.lines[overlapSeg[1]+1].setStartPt(secContig.lines[overlapSeg[3]].termPt2)
                            #  print("refContig - future adjusting line idx " + str(refContig.lineIdxs[overlapSeg[1]+1]) + " start pt to " + str(refContig.lines[overlapSeg[1]+1].termPt1) + " -> end pt is " + str(refContig.lines[overlapSeg[1]+1].termPt2))

                        refI = overlapSeg[1] + 1

                     # fill out the remaining segs after the last overlap end point
                     for refRemainIndex in range(refI, len(refContig.lines)):
                        print("refContig - filling out remaining lines - add line with idx " + str(refContig.lineIdxs[refRemainIndex]))
                        newRefContigSeg.insertLineToContigSeg(refContig.lineIdxs[refRemainIndex], refContig.lines[refRemainIndex], False)
                      #  if not newRefContigSeg.insertLineToContigSeg(refContig.lineIdxs[refRemainIndex], refContig.lines[refRemainIndex], False):
                      #     newRefContigSeg.finalizeContigSeg()
                      #     if not newRefContigSegCorrespIdxInserted:
                      #        newContigSegIdxToContigSeg[refIdx] = newRefContigSeg
                      #        newRefContigSegCorrespIdxInserted = True
                      #        newRefContigSeg = contigsSegsCls(refContig.lineIdxs[refRemainIndex], refContig.lines[refRemainIndex])
                      #     else:
                      #        newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newRefContigSeg
                      #        addNewIdxToContigSegMap += 1
                      #        newRefContigSeg = contigsSegsCls(refContig.lineIdxs[refRemainIndex], refContig.lines[refRemainIndex])

                     newRefContigSeg.finalizeContigSeg()


             #        if not newRefContigSegCorrespIdxInserted:
             #           newContigSegIdxToContigSeg[refIdx] = newRefContigSeg
             #        else:
             #           newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newRefContigSeg
             #           addNewIdxToContigSegMap += 1

                  # generate contig seg from sec contour
                  # rolling index secI
                  newSecContigSegCorrespIdxInserted = False
                  newSecContigSeg = None
                  if refIsAnchor:
                     newSecContigSeg = contigsSegsCls()

                     secOverlapStartIdx = 2
                     secOverlapEndIdx = 3
                     refOverlapStartIdx = 0
                     refOverlapEndIdx = 1
                     loopChange = 1
                     inclusivityShift = 1
                     flipOverlapLine = False

                     if isSecContigReverse:
                        #refAndSecOverlapSegs.reverse()
                        secOverlapStartIdx = 3
                        secOverlapEndIdx = 2
                        refOverlapStartIdx = 1
                        refOverlapEndIdx = 0
                        loopChange = -1
                        inclusivityShift = -1
                        flipOverlapLine = True

                     secI = 0
                     for overlapSeg in refAndSecOverlapSegs:
                        print("secContig - handle overlapSeg " + str(overlapSeg))
                        for sindex in range(secI, overlapSeg[secOverlapStartIdx]):
                           print("secContig - insert line idx " + str(secContig.lineIdxs[sindex]))
                           # if this is the final line before segment where the 2 contours overlap
                           # need to adjust the end point IF ref contour contig seg is longer
                         #  if sindex == overlapSeg[secOverlapStartIdx] - 1:
                         #     if isSecContigReverse:
                         #        secContig.lines[sindex].setEndPt(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt2)
                         #        print("secContig - reverse adjusting line idx " + str(secContig.lineIdxs[sindex]) + " with start pt " + str(secContig.lines[sindex].termPt1) + " -> end pt to " + str(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt2))
                         #     else:
                         #        secContig.lines[sindex].setEndPt(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt1)
                         #        print("secContig - adjusting line idx " + str(secContig.lineIdxs[sindex]) + " with start pt " + str(secContig.lines[sindex].termPt1) + " -> end pt to " + str(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt1))

                           newSecContigSeg.insertLineToContigSeg(secContig.lineIdxs[sindex], secContig.lines[sindex], False)

                        # need to make sure that the end point of the last line in the newSecContigSeg is equal to the start pt of the refOverlapstartIdx line
                        # Aug 2020 - enhanced -> try to get the intersecPt between the last line of newSecContigSeg and the beginning of refContig overlapSeg
                        #                        and modify both the newSecContigSeg and the connecting point (end pt or start pt depending on whether the secContig is reversed)
                        #                        to the intersect pt: if the intersect pt does NOT lie on the overlap line in the refContig seg -> move the connecting point
                        #                        to the end point of the newSecContigSeg
                        if len(newSecContigSeg.lines) > 0:
                           xsectPt = getXSectPtBtwn2LinesCheckIfInRefLine(refContig.lines[refOverlapStartIdx], newSecContigSeg.lines[-1])
                           if isSecContigReverse:
                              if xsectPt.all():
                                 print("secContig -> intersect pt " + str(xsectPt) + " exists between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and refContig overlap line with start pt " + str(refContig.lines[refOverlapStartIdx].termPt1) + " end pt " + str(refContig.lines[refOverlapStartIdx].termPt2))
                                 newSecContigSeg.lines[-1].setEndPt(xsectPt)
                                 refContig.lines[refOverlapStartIdx].setEndPt(xsectPt)
                              else:
                                 print("secContig -> intersect pt does not exist between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and refContig overlap line with start pt " + str(refContig.lines[refOverlapStartIdx].termPt1) + " end pt " + str(refContig.lines[refOverlapStartIdx].termPt2))
                                 refContig.lines[refOverlapStartIdx].setEndPt(newSecContigSeg.lines[-1].termPt2)
#                              if not np.array_equal(newSecContigSeg.lines[-1].termPt2, refContig.lines[overlapSeg[refOverlapStartIdx]].termPt2):
#                                 print("secContig - reverse -> end pt " + str(newSecContigSeg.lines[-1].termPt2) + " not equal to end pt of refContig overlapSeg " + str(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt2))
#                              newSecContigSeg.lines[-1].setEndPt(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt2)
                           else:
                              if xsectPt.all():
                                 print("secContig -> intersect pt " + str(xsectPt) + " exists between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and refContig overlap line with start pt " + str(refContig.lines[refOverlapStartIdx].termPt1) + " end pt " + str(refContig.lines[refOverlapStartIdx].termPt2))
                                 newSecContigSeg.lines[-1].setEndPt(xsectPt)
                                 refContig.lines[refOverlapStartIdx].setStartPt(xsectPt)
                              else:
                                 print("secContig -> intersect pt does not exist between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and refContig overlap line with start pt " + str(refContig.lines[refOverlapStartIdx].termPt1) + " end pt " + str(refContig.lines[refOverlapStartIdx].termPt2))

                                 refContig.lines[refOverlapStartIdx].setStartPt(newSecContigSeg.lines[-1].termPt2)
                            #  if not np.array_equal(newSecContigSeg.lines[-1].termPt2, refContig.lines[overlapSeg[refOverlapStartIdx]].termPt1):
                            #     print("secContig -> end pt " + str(newSecContigSeg.lines[-1].termPt2) + " not equal to start pt of refContig overlapSeg " + str(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt1))
                            #  newSecContigSeg.lines[-1].setEndPt(refContig.lines[overlapSeg[refOverlapStartIdx]].termPt1)

                          # if not newSecContigSeg.insertLineToContigSeg(secContig.lineIdxs[sindex], secContig.lines[sindex], False):
                          #    newSecContigSeg.finalizeContigSeg()
                          #    if not newSecContigSegCorrespIdxInserted:
                          #       newContigSegIdxToContigSeg[secIdx] = newSecContigSeg
                          #       newSecContigSegCorrespIdxInserted = True
                          #       newSecContigSeg = contigsSegsCls(secContig.lineIdxs[sindex], secContig.lines[sindex])
                          #    else:
                          #       newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newSecContigSeg
                          #       addNewIdxToContigSegMap += 1
                          #       newSecContigSeg = contigsSegsCls(secContig.lineIdxs[sindex], secContig.lines[sindex])

                        # now handle the overlapping segs
                        print("secContig - handling overlap portion")
                        for oindex in range(overlapSeg[refOverlapStartIdx], overlapSeg[refOverlapEndIdx]+inclusivityShift, loopChange):
                           overlapLine = copy.deepcopy(refContig.lines[oindex])
                           if flipOverlapLine:
                              overlapLine.flipLine()
                              print("secContig - reverse adding line with idx " + str(refContig.lineIdxs[oindex]))
                           else:
                              print("secContig - adding line with idx " + str(refContig.lineIdxs[oindex]))

                           newSecContigSeg.insertLineToContigSeg(refContig.lineIdxs[oindex], overlapLine, False)
                           #if not newSecContigSeg.insertLineToContigSeg(refContig.lineIdxs[oindex], overlapLine, False):
                           #   newSecContigSeg.finalizeContigSeg()
                           #   if not newSecContigSegCorrespIdxInserted:
                           #      newContigSegIdxToContigSeg[secIdx] = newSecContigSeg
                           #      newSecContigSegCorrespIdxInserted = True
                           #      newSecContigSeg = contigsSegsCls(refContig.lineIdxs[oindex], overlapLine)
                           #   else:
                           #      newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newSecContigSeg
                           #      addNewIdxToContigSegMap += 1
                           #      newSecContigSeg = contigsSegsCls(refContig.lineIdxs[oindex], overlapLine)

                        # adjust the future NON overlap portion
                        if overlapSeg[secOverlapEndIdx] < len(secContig.lines)-1:
                           xsectPt = getXSectPtBtwn2LinesCheckIfInRefLine(newSecContigSeg.lines[-1], secContig.lines[overlapSeg[secOverlapEndIdx]+1])
                           if isSecContigReverse:
                              if xsectPt.all():
                                 print("secContig - reverse -> intersect pt " + str(xsectPt) + " exists between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and next secContig line with start pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1) + " end pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt2))
                                 newSecContigSeg.lines[-1].setEndPt(xsectPt)
                                 secContig.lines[overlapSeg[secOverlapEndIdx]+1].setStartPt(xsectPt)
                              else:
                                 print("secContig - reverse -> intersect pt does not exist between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and next secContig line with start pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1) + " end pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt2))

                                 newSecContigSeg.lines[-1].setEndPt(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1)
                         #     secContig.lines[overlapSeg[secOverlapEndIdx]+1].setStartPt(refContig.lines[overlapSeg[refOverlapEndIdx]].termPt1)
                         #     print("secContig - reverse future adjusting line idx " + str(secContig.lineIdxs[overlapSeg[secOverlapEndIdx]+1]) + " start pt to " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1) + " -> end pt is " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt2))
                           else:
                              if xsectPt.all():
                                 print("secContig -> intersect pt " + str(xsectPt) + " exists between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and next secContig line with start pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1) + " end pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt2))
                                 newSecContigSeg.lines[-1].setEndPt(xsectPt)
                                 secContig.lines[overlapSeg[secOverlapEndIdx]+1].setStartPt(xsectPt)
                              else:
                                 print("secContig -> intersect pt does not exist between secContig line with start pt " + str(newSecContigSeg.lines[-1].termPt1) + " end pt " + str(newSecContigSeg.lines[-1].termPt2) + " and next secContig line with start pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1) + " end pt " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt2))

                                 newSecContigSeg.lines[-1].setEndPt(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1)
                          #    secContig.lines[overlapSeg[secOverlapEndIdx]+1].setStartPt(refContig.lines[overlapSeg[refOverlapEndIdx]].termPt2)
                          #    print("secContig - future adjusting line idx " + str(secContig.lineIdxs[overlapSeg[secOverlapEndIdx]+1]) + " start pt to " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt1) + " -> end pt is " + str(secContig.lines[overlapSeg[secOverlapEndIdx]+1].termPt2))

                        secI = overlapSeg[secOverlapEndIdx] + 1

                     # fill out the remaining segs after the last overlap end point
                     for secRemainIndex in range(secI, len(secContig.lines)):
                        newSecContigSeg.insertLineToContigSeg(secContig.lineIdxs[secRemainIndex], secContig.lines[secRemainIndex], False)
                       # if not newSecContigSeg.insertLineToContigSeg(secContig.lineIdxs[secRemainIndex], secContig.lines[secRemainIndex], False):
                       #    newSecContigSeg.finalizeContigSeg()
                       #    if not newSecContigSegCorrespIdxInserted:
                       #       newContigSegIdxToContigSeg[secIdx] = newSecContigSeg
                       #       newSecContigSegCorrespIdxInserted = True
                       #       newSecContigSeg = contigsSegsCls(secContig.lineIdxs[secRemainIndex], secContig.lines[secRemainIndex])
                       #    else:
                       #       newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newSecContigSeg
                       #       addNewIdxToContigSegMap += 1
                       #       newSecContigSeg = contigsSegsCls(secContig.lineIdxs[secRemainIndex], secContig.lines[secRemainIndex])

                       # print("secContig - filling out remaining lines - add line with idx " + str(secContig.lineIdxs[secRemainIndex]))

                     if isSecContigReverse:
                        newSecContigSeg.reverseOrientationOfContigSeg()

                     newSecContigSeg.finalizeContigSeg()

               #      if not newSecContigSegCorrespIdxInserted:
               #         newContigSegIdxToContigSeg[secIdx] = newSecContigSeg
               #      else:
               #         newContigSegIdxToContigSeg[addNewIdxToContigSegMap] = newSecContigSeg
               #         addNewIdxToContigSegMap += 1

                  # start here for processing newRefContigSeg and newSecContigSeg
                  if newRefContigSeg:
                     newRefContigSeg = filterOutPtLinesInContigSeg(newRefContigSeg)
                     segToCheck = newRefContigSeg
                     idxToStore = refIdx
                     print("segToCheck is newRefContigSeg")
                  else:
                     newSecContigSeg = filterOutPtLinesInContigSeg(newSecContigSeg)
                     segToCheck = newSecContigSeg
                     idxToStore = secIdx
                     print("segToCheck is newSecContigSeg")

                  if len(segToCheck.lines) < 1:
                     print("segToCheck line is empty - do nothing")
                     continue

                  # check if this segToCheck is completely overlapped by another contig seg in the map
                  for repIdx in range(len(contigSegIdxToContigSeg)):
                     # check if segToCheck is overlapped by any longer contigs in the map
                     repContig = checkIfOneContigCompletelyOverlap(segToCheck, contigSegIdxToContigSeg[repIdx])
                     if repContig is segToCheck:
                        print("found contig idx " + str(repIdx) + " that overlaps with segToCheck")
                        print("segToCheck info ")
                        segToCheck.printContigSegInfo()
                        print("contig idx " + str(repIdx) + " info ")
                        contigSegIdxToContigSeg[repIdx].printContigSegInfo()

                        # if this is newSecContigSeg / secIdx -> store it so that in future if secIdx
                        #  is the ref -> skip it
                        # if this is newRefContigSeg / refIdx -> break out of sec loop and do NOT store it into
                        # newContigSegIdxToContigSeg map at the end
                        if segToCheck is newRefContigSeg:
                           breakRefAndDoNotStore = True
                        else:
                           secAlreadyDetermineIgnore.append(secIdx)
                        break
                  else:
                     print("segToCheck does not get overlapped by existing contigSeg - store into newContigSegIdxToContigSeg")
                     newContigSegIdxToContigSeg[idxToStore] = segToCheck

               # break out of sec contig loop if ref already determined to be completely overlapped
               # by other sec contig
               if breakRefAndDoNotStore:
                  break

            # if refContour has been compared to all sec and none was store into newContigSeg - store the one from orig
            if not newContigSegIdxToContigSeg.get(refIdx, None) and not breakRefAndDoNotStore:
               print("refIdx " + str(refIdx) + " compared to all sec and none was stored into newContigSeg - store orig")
               newContigSegIdxToContigSeg[refIdx] = contigSegIdxToContigSeg[refIdx]

         # put together the ref and sec contig segs into list of contig segs
     #    retRefContigSegs = []
     #    retSecContigSegs = []

     #    for refKey, refContigSegIter in refProcessedContigs.items():
     #       retRefContigSegs.append(refContigSegIter)
     #    for secKey, secContigSegIter in secProcessedContigs.items():
     #       retSecContigSegs.append(secContigSegIter)

     #    retTotalContigsSegs = retRefContigSegs
     #    retTotalContigsSegs.extend(retSecContigSegs)
         contigSegIdxToContigSeg = newContigSegIdxToContigSeg
         newContigSegIdxToContigSeg = {}
         iterCount += 1

      retContigSegs = []
      for contigKey, contig in contigSegIdxToContigSeg.items():
         retContigSegs.append(contig)

      return retContigSegs

   #   return (retRefContigSegs, retSecContigSegs, retTotalContigsSegs)

   def processUniqueContsIntoUniqueContigSegs(self):
      for contourIdx, uniqueEntries in self.contourIdxToUniqueConts.items():
         self.contourIdxToUniqueContigSegs[contourIdx] = self.mergeContour1AndContour2OfSameContour(contourIdx, uniqueEntries[1], uniqueEntries[2])

   def processAndMergeUniqueContigSegsIntoOne(self):
      # for the contour idxs above - create a map where key is the total length of all contig segs for that contour
      totalLenKey = []
      for contourIdx, contourContigSegs in self.contourIdxToUniqueContigSegs.items():
         totalLen = 0
         for contigSeg in contourContigSegs:
            totalLen += contigSeg.length
         if not self.contigSegsTotalLenToContigSegs.get(totalLen, None):
            self.contigSegsTotalLenToContigSegs[totalLen] = [contourContigSegs]
            totalLenKey.append(totalLen)
         else:
            self.contigSegsTotalLenToContigSegs[totalLen].append(contourContigSegs)

      totalLenKey.sort()

      mergedContigSegs = None

      flatSortedContigSegsList = []

      for lenKey in totalLenKey:
         for contigSegsList in self.contigSegsTotalLenToContigSegs[lenKey]:
            flatSortedContigSegsList.append(contigSegsList)

      print("totalLenKey is " + str(totalLenKey))
      print("flatSortedContigSegsList is " + str(flatSortedContigSegsList))

      if len(flatSortedContigSegsList) > 1:
         refContigSegsList = flatSortedContigSegsList[0]
         for i in range(1, len(flatSortedContigSegsList)):
            print("merging contig segs refContigSegsList and contigSegs idx " + str(i))
            refContigSegsList = self.mergeSecContourToRef(refContigSegsList, flatSortedContigSegsList[i])
         self.uniqueContigSegs = refContigSegsList
      else:
         self.uniqueContigSegs = flatSortedContigSegsList[0]

      return self.uniqueContigSegs

   def processContourLineSegs(self):
      contoursToDelete = []
      for idx, contourLineIdxs in self.lineContourToLineIdxs.items():
         print("process contour idx " + str(idx))
         print("contour line idxs are " + str(contourLineIdxs))
         newContourLineIdxs = []

         if len(contourLineIdxs) == 0:
            print("no line in contour idx " + str(idx) + " - remove it")
            contoursToDelete.append(idx)
            continue

     #    for i in range(0, len(contourLineIdxs)-1):
     #       lineIdx = contourLineIdxs[i]
     #       newContourLineIdxs.append(lineIdx)
     #       nextLineIdx = contourLineIdxs[i+1]
     #       print("checking join logic of line idx " + str(lineIdx) + " and line idx " + str(nextLineIdx))
     #       newLine = self.addLineBtwn2Lines(self.lineIdxToLineMap[lineIdx], self.lineIdxToLineMap[nextLineIdx])
     #       if newLine.lineLength > 1:
     #          newLineIdx = self.insertLineToIdxMap(newLine)
     #          print("Adding the following line with idx: " + str(newLineIdx) + " to contour: " + str(idx))
     #          newContourLineIdxs.append(newLineIdx)
     #       else:
     #          self.lineIdxToLineMap[lineIdx].modifyTermPt(self.lineIdxToLineMap[nextLineIdx].termPt1, 2)
    #        join2LinesDict = self.join2LinesInContour(self.lineIdxToLineMap[lineIdx], self.lineIdxToLineMap[nextLineIdx])
    #        if join2LinesDict[0]:  # if the return dict is true - means add extra line between the 2 lines
    #           newLineIdx = self.insertLineToIdxMap(join2LinesDict[1])
    #           print("Adding the following line with idx: " + str(newLineIdx) + " to contour: ")
    #           join2LinesDict[1].displayLineInfo()
    #           newContourLineIdxs.append(newLineIdx)

    #     print(contourLineIdxs[-1])
      #   newContourLineIdxs.append(contourLineIdxs[-1])

       #  self.lineContourToLineIdxs[idx] = newContourLineIdxs

      for idxToDelete in contoursToDelete:
         del self.lineContourToLineIdxs[idxToDelete]

   def processParallelLineSegsInContours(self):
      for idx, contourLineIdxs in self.lineContourToLineIdxs.items():
         print("process contour idx " + str(idx))
         print("contour line idxs are " + str(contourLineIdxs))
         linesToDelete = []
         linesInContourAfterPruning = []
         for i in range(0, len(contourLineIdxs)):
            if i not in linesToDelete:
               currLineIdx = contourLineIdxs[i]
               for j in range(i+1, len(contourLineIdxs)):
                  if j not in linesToDelete:
                     lineIdxToCheck = contourLineIdxs[j]
                     print("line1 idx is: " + str(currLineIdx) + " line2 idx is: " + str(lineIdxToCheck))
                     lineToRemove = self.checkIf2LinesRedundant(self.lineIdxToLineMap[currLineIdx], self.lineIdxToLineMap[lineIdxToCheck])

                     if lineToRemove:
                        if lineToRemove == 1:
                           if i not in linesToDelete:
                              linesToDelete.append(i)
                        elif lineToRemove == 2:
                           if j not in linesToDelete:
                              linesToDelete.append(j)

         if linesToDelete:
            lineIdxToDelete = linesToDelete.pop()
            for lineIndex in range(0, len(contourLineIdxs)):
               if lineIndex == lineIdxToDelete:
                  if linesToDelete:
                     lineIdxToDelete = linesToDelete.pop()
                  else:
                     linesInContourAfterPruning.extend(contourLineIdxs[lineIndex+1:])
                     break
               else:
                  linesInContourAfterPruning.append(contourLineIdxs[lineIndex])

            self.lineContourToLineIdxs[idx] = linesInContourAfterPruning

   def analysis_displayParallelLineSegsBtwn2Contours(self, contour1Idx, contour2Idx):
      contour1 = self.lineContourToLineIdxs.get(contour1Idx, None)
      contour2 = self.lineContourToLineIdxs.get(contour2Idx, None)

      if not contour1 or not contour2:
         print("Error getting contours: contour1 idx is " + str(contour1Idx) + " contour2 idx is " + str(contour2Idx))
         return

      print("analysis process parallel: contour1 idx is " + str(contour1Idx) + " and contour2 idx is " + str(contour2Idx))

      contour1LinesToDelete = []
      contour2LinesToDelete = []
      for contour1LineIdx in range(0, len(contour1)):
         contour2 = self.lineContourToLineIdxs.get(contour2Idx, None)
         for contour2LineIdx in range(0, len(contour2)):
            contour1Line = contour1[contour1LineIdx]
            contour2Line = contour2[contour2LineIdx]
            print("contour1 line idx is " + str(contour1Line) + " contour2 line idx is " + str(contour2Line))
            lineToRemove = self.checkIf2LinesRedundant(self.lineIdxToLineMap[contour1Line], self.lineIdxToLineMap[contour2Line])
            if lineToRemove:
               if contour1LineIdx not in contour1LinesToDelete:
                  contour1LinesToDelete.append(contour1LineIdx)
               if contour2LineIdx not in contour2LinesToDelete:
                  contour2LinesToDelete.append(contour2LineIdx)

      print("contour1 lines that are parallel with contour2 is " + str(contour1LinesToDelete))
      print("contour2 lines that are parallel with contour1 is " + str(contour2LinesToDelete))

      print("contour1 number of redundant lines is " + str(len(contour1LinesToDelete)) + " and total number of lines in contour1 is " + str(len(contour1)))
      print("contour2 number of redundant lines is " + str(len(contour2LinesToDelete)) + " and total number of lines in contour2 is " + str(len(contour2)))

      self.handle2ContoursWithParallelCurves(contour1Idx, contour1LinesToDelete, contour2Idx, contour2LinesToDelete)

   # contour with greater number of lines: more info => delete redundant lines from longer contour then add smaller
   # contour in its entirety into contour with greater number of lines
   def handle2ContoursWithParallelCurves(self, contour1Idx, contour1LinesToDelete, contour2Idx, contour2LinesToDelete):

      contour1RedundantLen = len(contour1LinesToDelete)
      contour2RedundantLen = len(contour2LinesToDelete)

      if (contour1RedundantLen < 1) or \
         (contour2RedundantLen < 1):
         print("contour1RedundantLen is " + str(contour1RedundantLen) + " contour2RedundantLen is " + str(contour2RedundantLen) + " - do nothing")
         return

      contour1 = self.lineContourToLineIdxs.get(contour1Idx, None)
      contour2 = self.lineContourToLineIdxs.get(contour2Idx, None)

      contour1Len = len(contour1)
      contour2Len = len(contour2)

      if contour1Len > contour2Len:
         refContour = contour1
         refContourLinesToDelete = contour1LinesToDelete
         contourToExt = contour2
         refContourIdx = contour1Idx
         delContourIdx = contour2Idx
      else:
         refContour = contour2
         refContourLinesToDelete = contour2LinesToDelete
         contourToExt = contour1
         refContourIdx = contour2Idx
         delContourIdx = contour1Idx

      refContourLinesToDelete.sort()

      linesInContourAfterPruning = []
      print("ref contour idx " + str(refContourIdx) + " before processing has lines " + str(refContour) + " with length " + str(len(refContour)))
      refContourLineToDelete = refContourLinesToDelete.pop(0)

      for lineIndex in range(0, len(refContour)):
         if lineIndex == refContourLineToDelete:
            if refContourLinesToDelete:
               refContourLineToDelete = refContourLinesToDelete.pop(0)
            else:
               linesInContourAfterPruning.extend(refContour[lineIndex+1:])
               break
         else:
            linesInContourAfterPruning.append(refContour[lineIndex])

      print("ref contour idx " + str(refContourIdx) + " before extend sec contour has lines " + str(linesInContourAfterPruning) + " with length " + str(len(linesInContourAfterPruning)))

      linesInContourAfterPruning.extend(contourToExt)

      print("ref contour idx " + str(refContourIdx) + " after processing has lines " + str(linesInContourAfterPruning) + " with length " + str(len(linesInContourAfterPruning)))

      self.lineContourToLineIdxs[refContourIdx] = linesInContourAfterPruning
      self.lineContourToLineIdxs[delContourIdx] = []


   def processParallelLineSegsBtwn2Contours(self, contour1Idx, contour2Idx):
      contour1 = self.lineContourToLineIdxs.get(contour1Idx, None)
      contour2 = self.lineContourToLineIdxs.get(contour2Idx, None)

      if not contour1 or not contour2:
         print("Error getting contours: contour1 idx is " + str(contour1Idx) + " contour2 idx is " + str(contour2Idx))
         return

      print("process parallel: contour1 idx is " + str(contour1Idx) + " and contour2 idx is " + str(contour2Idx))

      contour1LinesToDelete = []
      for contour1LineIdx in range(0, len(contour1)):
         contour2LinesToDelete = []
         contour2 = self.lineContourToLineIdxs.get(contour2Idx, None)
         for contour2LineIdx in range(0, len(contour2)):
            contour1Line = contour1[contour1LineIdx]
            contour2Line = contour2[contour2LineIdx]
            print("contour1 line idx is " + str(contour1Line) + " contour2 line idx is " + str(contour2Line))
            lineToRemove = self.checkIf2LinesRedundant(self.lineIdxToLineMap[contour1Line], self.lineIdxToLineMap[contour2Line])

            if lineToRemove:
               print("line to remove is " + str(lineToRemove))
               if lineToRemove == 1:
                  if contour1LineIdx not in contour1LinesToDelete:
                     contour1LinesToDelete.append(contour1LineIdx)
               elif lineToRemove == 2:
                  if contour2LineIdx not in contour2LinesToDelete:
                     contour2LinesToDelete.append(contour2LineIdx)

         if contour2LinesToDelete:
            print("contour2 lines to delete are: " + str(contour2LinesToDelete))
            linesInContour2AfterPruning = []
            contour2LineToDelete = contour2LinesToDelete.pop(0)
            for line2Index in range(0, len(contour2)):
               if line2Index == contour2LineToDelete:
                  if contour2LinesToDelete:
                     contour2LineToDelete = contour2LinesToDelete.pop(0)
                  else:
                     linesInContour2AfterPruning.extend(contour2[line2Index+1:])
                     break
               else:
                  linesInContour2AfterPruning.append(contour2[line2Index])
            self.lineContourToLineIdxs[contour2Idx] = linesInContour2AfterPruning
            print("resulting contour idx " + str(contour2Idx) + " lines are " + str(linesInContour2AfterPruning))

      if contour1LinesToDelete:
         print("contour1 lines to delete are: " + str(contour1LinesToDelete))
         linesInContour1AfterPruning = []
         # need to sort contour1LinesToDelete since this is in outer loop
         contour1LinesToDelete.sort()
         print("contour1 lines to delete after sorting are: " + str(contour1LinesToDelete))
         contour1LineToDelete = contour1LinesToDelete.pop(0)
         for line1Index in range(0, len(contour1)):
            if line1Index == contour1LineToDelete:
               if contour1LinesToDelete:
                  contour1LineToDelete = contour1LinesToDelete.pop(0)
               else:
                  linesInContour1AfterPruning.extend(contour1[line1Index+1:])
                  break
            else:
               linesInContour1AfterPruning.append(contour1[line1Index])
         print("resulting contour idx " + str(contour1Idx) + " lines are " + str(linesInContour1AfterPruning))
         self.lineContourToLineIdxs[contour1Idx] = linesInContour1AfterPruning

   def processParallelLinesBtwnAllContours(self):
      contourIdxList = []
      for contourIdx in self.lineContourToLineIdxs:
         contourIdxList.append(contourIdx)

      for contour1 in range(0, len(contourIdxList)):
         for contour2 in range(contour1+1, len(contourIdxList)):
            self.processParallelLineSegsBtwn2Contours(contour1, contour2)

   def displayLineContourToLineIdxs(self):
      for contourIdx, lines in self.lineContourToLineIdxs.items():
         print("lines belonging to contour " + str(contourIdx) + " is " + str(lines))

   def displayLineIdxToLineMap(self):
      for idx, line in self.lineIdxToLineMap.items():
         print("line index is " + str(idx))
         line.displayLineInfo()
         if idx+1 < len(self.lineIdxToLineMap):
            print("dist between idx " + str(idx) + " and idx+1 " + str(idx+1) + " is " + str(LA.norm(np.subtract(self.lineIdxToLineMap[idx+1].termPt1, self.lineIdxToLineMap[idx].termPt2))))
      print("number of lines is " + str(len(self.lineIdxToLineMap)))

   def displayContigLineSegs(self):
      for idx, contigs in self.contigSegIdxToContigSeg.items():
         print("contig ling seg idx " + str(idx))
         contigs.printContigSegInfo()

   def drawLinesToImg(self, img, arrowed):
      line_thickness = 1
      for idx, line in self.lineIdxToLineMap.items():
         if arrowed:
            cv.arrowedLine(img, (line.termPt1[0], line.termPt1[1]), (line.termPt2[0], line.termPt2[1]), (0, 255, 0), line_thickness, tipLength=0.5)
         else:
            cv.line(img, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

   def generateImgFromLines(self, outFileName, arrowed=False):
       line_thickness = 1
       imgOutLine = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8)*255
       for idx, line in self.lineIdxToLineMap.items():
          if arrowed:
             cv.arrowedLine(imgOutLine, (line.termPt1[0], line.termPt1[1]), (line.termPt2[0], line.termPt2[1]), (0, 255, 0), line_thickness, tipLength=0.5)
          else:
             cv.line(imgOutLine, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

       cv.imwrite(outFileName, imgOutLine)

   def drawSpecificContourToImg(self, imgName, contourIdx, arrowed):
      line_thickness = 1
      imgOutContourFull = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8) * 255
      contourLineIdxs = self.lineContourToLineIdxs.get(contourIdx, None)
      if not contourLineIdxs:
         print("Error drawing contour idx " + str(contourIdx) + " - not found")
         return

      for lineIdx in contourLineIdxs:
         line = self.lineIdxToLineMap[lineIdx]
         if arrowed:
            cv.arrowedLine(imgOutContourFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
         else:
            cv.line(imgOutContourFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

      cv.imwrite(imgName, imgOutContourFull)

   def drawSpecificListOfContoursToImg(self, imgName, contourList, arrowed):
      line_thickness = 1
      imgOutContourFull = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8) * 255
      for contourIdx in contourList:
         contourLineIdxs = self.lineContourToLineIdxs.get(contourIdx, None)
         if contourLineIdxs:
            for lineIdx in contourLineIdxs:
               line = self.lineIdxToLineMap[lineIdx]
               if arrowed:
                  cv.arrowedLine(imgOutContourFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
               else:
                  cv.line(imgOutContourFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

      cv.imwrite(imgName, imgOutContourFull)

   def drawContoursToImg(self, imgName, arrowed):
      line_thickness = 1
      imgOutContourFull = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8) * 255
      for idx, contourLineIdxs in self.lineContourToLineIdxs.items():
         for lineIdx in contourLineIdxs:
            line = self.lineIdxToLineMap[lineIdx]
            if arrowed:
               cv.arrowedLine(imgOutContourFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContourFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

      cv.imwrite(imgName, imgOutContourFull)

   def drawContigSegsToImg(self, imgName, arrowed):
      line_thickness = 1
      imgOutContigSegsFull = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8) * 255
      for idx, contigSeg in self.contigSegIdxToContigSeg.items():
         for line in contigSeg.lines:
            if arrowed:
               cv.arrowedLine(imgOutContigSegsFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContigSegsFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

      cv.imwrite(imgName, imgOutContigSegsFull)


   def drawContoursToSeparateImgs(self, imgBaseName, imgSuffix, arrowed=False):
      line_thickness = 1
      for contourIdx, lines in self.lineContourToLineIdxs.items():
         imgOutContour = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8)*255
         for lineIdx in lines:
            line = self.lineIdxToLineMap[lineIdx]
            if arrowed:
               cv.arrowedLine(imgOutContour, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContour, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)
         cv.imwrite(imgBaseName + "_" + str(contourIdx) + imgSuffix, imgOutContour)

   def drawContigSegsToSeparateImgs(self, imgBaseName, imgSuffix, arrowed=False):
      line_thickness = 1
      for idx, contigSeg in self.contigSegIdxToContigSeg.items():
         imgOutContigSeg = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8)*255
         for line in contigSeg.lines:
            if arrowed:
               cv.arrowedLine(imgOutContigSeg, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContigSeg, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)
         cv.imwrite(imgBaseName + "_" + str(idx) + imgSuffix, imgOutContigSeg)

   def givenContourDrawImg(self, contour, imgName, arrowed):
      line_thickness = 1
      imgOutContour = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8) * 255
      for lineIdx in contour:
         line = self.lineIdxToLineMap[lineIdx]
         if arrowed:
            cv.arrowedLine(imgOutContour, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
         else:
            cv.line(imgOutContour, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

      cv.imwrite(imgName, imgOutContour)

   def givenContoursDrawToSameImg(self, listOfContours, imgName, arrowed=False):
      line_thickness = 1
      imgOutContour = np.ones([self.imgHeight, self.imgWidth], dtype=np.uint8) * 255
      listOfColors = [(0, 255, 0), (0, 255, 0)]
      for contourIdx in range(len(listOfContours)):
         for lineIdx in listOfContours[contourIdx]:
            line = self.lineIdxToLineMap[lineIdx]
            if arrowed:
               cv.arrowedLine(imgOutContour, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), listOfColors[contourIdx % len(listOfColors)], line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContour, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), listOfColors[contourIdx % len(listOfColors)], line_thickness)

      cv.imwrite(imgName, imgOutContour)

   # 2 lines are considered redundant if:
   #   1) their slopes are parallel (ie. the dot product of their unit vectors
   #      are greater than 0.95 or less than -0.95
   #   2) if the 2 lines have a perpendicular distance than is less than some threshold
   #       ie. the perp dist. is less than some max dist (2 lines parallel but close to each other)
   #   3) if the 2 lines are close to each other in distance as well (meaning the 2 lines overlap in
   #        in parallel direction
   def checkIf2LinesRedundant(self, line1, line2, maxPerpDist=None):

      returnLine = None

      if not maxPerpDist:
         maxPerpDist = self.maxPerpDist

      # as a first pass -> generate bounding box for line1 and line2 -> see if they match
      # to generate bounding box -> take the 2 perp vect of the line and get the points that are displaced by the
      #  perp vect by maxPerpDist*2 from the startPt and endPt -> take these 4 points generated - and create a bounding box
      #
      # generate the bounding box for line1
      bboxFactor = 2
      line1PerpVect1 = np.array([-line1.unitVect[1], line1.unitVect[0]])
      line1PerpVect2 = np.array([line1.unitVect[1], -line1.unitVect[0]])
      line1Pt1 = line1.termPt1 + bboxFactor * maxPerpDist * line1PerpVect1
      line1Pt2 = line1.termPt1 + bboxFactor * maxPerpDist * line1PerpVect2
      line1Pt3 = line1.termPt2 + bboxFactor * maxPerpDist * line1PerpVect1
      line1Pt4 = line1.termPt2 + bboxFactor * maxPerpDist * line1PerpVect2
      bbox1 = boundBoxCls([line1Pt1, line1Pt2, line1Pt3, line1Pt4])

      line2PerpVect1 = np.array([-line2.unitVect[1], line2.unitVect[0]])
      line2PerpVect2 = np.array([line2.unitVect[1], -line2.unitVect[0]])
      line2Pt1 = line2.termPt1 + bboxFactor * maxPerpDist * line2PerpVect1
      line2Pt2 = line2.termPt1 + bboxFactor * maxPerpDist * line2PerpVect2
      line2Pt3 = line2.termPt2 + bboxFactor * maxPerpDist * line2PerpVect1
      line2Pt4 = line2.termPt2 + bboxFactor * maxPerpDist * line2PerpVect2
      bbox2 = boundBoxCls([line2Pt1, line2Pt2, line2Pt3, line2Pt4])

      if checkIf2BoundBoxesOverlap(bbox1, bbox2):
         print("checkIf2LinesRedundant - bounding box for line 1 and line 2 do not overlap or touch - too far away to check if parallel")
         return returnLine

      dotProdBtw2Lines = np.dot(line1.unitVect, line2.unitVect)
      print("dot is " + str(dotProdBtw2Lines) + " line1.unitVect " + str(line1.unitVect) + " line2.unitVect " + str(line2.unitVect))
      if LA.norm(dotProdBtw2Lines) >= self.minDotProdParallel:
         if line1.lineLength > line2.lineLength:
            refLine = copy.deepcopy(line1)
            secLine = copy.deepcopy(line2)
            print("refLine is line1 - secLine is line2")
            delLineToReturn = 2

         else:
            refLine = copy.deepcopy(line2)
            secLine = copy.deepcopy(line1)
            print("refLine is line2 - secLine is line1")
            delLineToReturn = 1

         # check if the 2 lines are oriented in opposite directions
         # if so, flip the secondary line
         if dotProdBtw2Lines < 0:
            secLine.flipLine()
            print("dot product between refLine and secLine is negative - flip it")

         secLinePt1Dist = refLine.getDistBtwnPtAndStartOrEndOfLine(secLine.termPt1, 1)
         secLinePt2Dist = refLine.getDistBtwnPtAndStartOrEndOfLine(secLine.termPt2, 2)

         print("dot product between line1 and line2 is " + str(dotProdBtw2Lines))
         print("secLinePt1Dist is " + str(secLinePt1Dist) + " length of refLine is " + str(refLine.lineLength))
         print("secLinePt2Dist is " + str(secLinePt2Dist) + " length of refLine is " + str(refLine.lineLength))

         normDistFromStartPt = secLinePt1Dist[0]
         perpDistFromStartPt = secLinePt1Dist[1]
         normDistFromEndPt = secLinePt2Dist[0]
         perpDistFromEndPt = secLinePt2Dist[1]

         print("normDistFromStartPt is " + str(normDistFromStartPt) + " perpDistFromStartPt is " + str(perpDistFromStartPt) + " normDistFromEndPt is " + str(normDistFromEndPt) + " perpDistFromEndPt is " + str(perpDistFromEndPt) + " refLine length is " + str(refLine.lineLength))

         if (perpDistFromStartPt < maxPerpDist) and \
            (perpDistFromEndPt < maxPerpDist):
             if (normDistFromStartPt >= 0) and \
                (normDistFromEndPt >= 0):
                returnLine = delLineToReturn
             else:
              #  if ((normDistFromStartPt < 0) and (abs(normDistFromStartPt) < refLine.lineLength * self.dispFactor)) or \
              #     ((normDistFromEndPt < 0) and (abs(normDistFromEndPt) < refLine.lineLength * self.dispFactor)) :
                if ((normDistFromStartPt > 0) and (normDistFromStartPt < refLine.lineLength)) or \
                   ((normDistFromEndPt > 0) and (normDistFromEndPt < refLine.lineLength)):
                   returnLine = delLineToReturn

      return returnLine
