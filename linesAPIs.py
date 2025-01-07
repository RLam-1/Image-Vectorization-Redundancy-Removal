#!/usr/bin/python

import numpy as np
import cv2 as cv
from numpy import linalg as LA
from enum import Enum
from operator import itemgetter, attrgetter
import math
import copy
import json
import traceback
from matplotlib import pyplot as plt

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

   def __init__(self, termPt1=None, termPt2=None):
      if termPt1 is not None:
         self.termPt1 = np.array(termPt1)
      else:
         self.termPt1 = np.array([None, None])
      if termPt2 is not None:
         self.termPt2 = np.array(termPt2)
      else:
         self.termPt2 = np.array([None,None])

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
      self.lineVect = None

      self.maxArea = 0

      self.hash = None

      # this data structure contains reference to parentObj that this line belongs
      # to (if the line belongs to another parentObj) and holds the position of the line
      # within the parentObj
      self.parentObj = None

      if self.termPt1.all() and self.termPt2.all():
         self.calcLineMetadata()
         self.getHash()
         
   # API to check if line is initialized - if the start / end pt are both 0,0 -> this means that the line was not initialized with start and end point
   def isLineInitialized(self):
     return (not (np.array_equal(self.termPt1, np.array([0,0])) and \
                 (np.array_equal(self.termPt2, np.array([0,0])))))

   # Common API to get the start / end pt of the line
   def getStartPt(self):
      return self.termPt1

   def getStartPtAsTuple(self):
      return tuple(self.getStartPt())

   def getEndPt(self):
      return self.termPt2

   def getEndPtAsTuple(self):
      return tuple(self.getEndPt())

   # API that sets reference to parent if the line belongs to a parent curve or contig seg
   def setPosInParent(self, parentObj, pos):
      self.parentObj = [parentObj, pos]

   # API that returns the other endpt of the line given the input endpt
   #
   # INPUT: one endpt of line
   # return - the other end pt of the line
   def getOtherEndPt(self, inPt):
      if np.array_equal(inPt, self.termPt1):
         return self.termPt2
      if np.array_equal(inPt, self.termPt2):
         return self.termPt1

      print("%s is not an end pt of the line" % (inPt))
      return None

   def getTermPt1AsTuple(self):
      return tuple(self.termPt1)

   def getTermPt2AsTuple(self):
      return tuple(self.termPt2)

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
      hashTuple = tuple(sorted([tuple(self.termPt1), tuple(self.termPt2)], key=lambda k: (k[0], k[1])))
      return hash(hashTuple)

   # API that takes as input a line. If the input line is contiguous with the
   # line AND the 2 lines have the same unitVect (or 180 degrees off), then
   # combine the 2 into a new line
   def __add__(self, line2):
      contigPt = checkIf2LinesAreContiguous(self, line2)
      if contigPt:
         if LA.norm(np.dot(self.getUnitVect(), line2.getUnitVect())) < 0.98:
            print("The input line and current line do not have the same unitVect")
            self.displayLineInfo()
            line2.displayLineInfo()
            return self
         else:
            return lineCls(self.getOtherTermPt(contigPt), line2.getOtherTermPt(contigPt))
      else:
         print("The input line is not contiguous with current line")
         self.displayLineInfo()
         line2.displayLineInfo()
         return self

   # API that takes as input a line. If the line is a sub-line of the existing line
   #  (ONLY the subline up to the line itself and NOT a superline of self), subtract the
   #  subline from the line
   #  HERE define SUBLINE as line seg that is part of the line up to the line itself
   #  SUPERLINE is line seg that overlaps and include the current line but extends further
   #  thereby completely overlapping input line
   #  If input line is SUPERLINE - return None
   #  Otherwise return the remaining portions of the line
   def __subtract__(self, line2):
      line2TermPt1OnLine = self.checkIfPointIsOnLine(line2.getStartPt())
      if line2TermPt1OnLine is None:
         line2TermPt2OnLine = self.checkIfPointIsOnLine(line2.getEndPt())
         if line2TermPt2OnLine is None:
            print("end pt of line 2 {} not on line - cannot subtract".format(line2.getEndPt()))
            return self
         else:
           if LA.norm(line2TermPt1OnLine) < LA.norm(line2TermPt2OnLine):
              firstTermPt = line2.getStartPt()
              secondTermPt = line2.getEndPt()
           else:
              firstTermPt = line2.getEndPt()
              secondTermPt = line2.getStartPt()
           # if the conditions for subtraction are met - the resulting lines are
           # 1) self.getStartPt() -> firstTermPt
           # 2) secondTermPt -> self.getEndPt()
           # these lines are returned if they are not points
           line1 = lineCls(self.getStartPt(), firstTermPt)
           line2 = lineCls(secondTermPt, self.getEndPt())
           retLine = []
           if not line1.checkifLineIsPoint():
              retLine.append(line1)
           if not line2.checkIfLineIsPoint():
              retLine.append(line2)
           return retLine

      else:
         print("start pt of line2 {} not on line - cannot subtract".format(line2.getStartPt()))
         return self

   def getHash(self):
      self.hash = self.__hash__()
      return self.hash

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
   #   self.calcLineMetadata()

   def setEndPt(self, pt):
      print("changing end pt of line from " + str(self.termPt2) + " to " + str(pt) + " -- start pt is " + str(self.termPt1))
      self.termPt2 = pt
      self.calcLineMetadata()
      self.getHash()

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
         print("endpt is " + str(self.termPt2))
         status = False

      return status

   def finalizeLine(self):
      if len(self.midPts) > 0:
         endPt = self.midPts[-1]
      else:
         endPt = self.termPt1
      self.setEndPt(endPt)

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
      xMid = ((self.termPt2[0] - self.termPt1[0]) / 2.0) + self.termPt1[0]
      yMid = ((self.termPt2[1] - self.termPt1[1]) / 2.0) + self.termPt1[1]
      self.lineMidPt = np.array([xMid, yMid])
      self.midPts = []

      self.lineVect = self.termPt2 - self.termPt1
      if LA.norm(self.lineVect) > 0:
         self.unitVect = self.lineVect / LA.norm(self.lineVect)
      else:
         self.unitVect = self.lineVect

   def getLength(self):
      if not self.lineLength:
         self.lineLength = LA.norm(self.termPt2 - self.termPt1)
      return self.lineLength

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
      diffVect = pt - self.getStartPt()

      if LA.norm(diffVect) == 0:
         print("pt is actually the start pt of the line " + str(self.termPt1))
         return diffVect

      uDiffVect = diffVect / LA.norm(diffVect)

      if (np.dot(uDiffVect, self.getUnitVect()) >= 0.98) and \
         (LA.norm(diffVect) <= self.getLength()):
         print("pt " + str(pt) + " LIES on line with start pt " + str(self.termPt1) + " and end pt " + str(self.termPt2))
         return diffVect

      print("uDiffVect is " + str(uDiffVect) + " unitVect is " + str(self.unitVect) + " ptDist is " + str(LA.norm(diffVect)) + " lineLength is " + str(self.lineLength))
      print("pt " + str(pt) + " DOES NOT LIE on line with start pt " + str(self.termPt1) + " and end pt " + str(self.termPt2))
      return None

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
   #  return value - (A, abs(B))
   #  for both START AND END PTS - A > 0 if it is along line
   #  whereas A < 0 (for START and END pt) means the pt is projected "behind" the
   #  start and end pt (not along the line but in opposite dir)
   #
   #  INPUT: pt of interest to get dist from
   #         startOrEnd - 1 is startPt (termPt1), 2 is endPt (termPt2)
   def getDistBtwnPtAndStartOrEndOfLine(self, pt, end=False):

      refPt = self.termPt1
      unitVect = self.unitVect
      # set the refPt to be the end point of the line
      # and take the negative of the normal vector since we are using endpt as ref
      # this way, if pt projected to ref line lies within line, A > 0
      if end:
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

      # if the B or the multiplication factor of the unit perp vect is < 0 - this
      # means that the perp vect should've been oriented the other way
      if x[1] < 0:
         perpVect *= -1

      return (x[0], math.fabs(x[1]), perpVect)

   # this API projects:
   #  - the start pt of the input line to the start of the self line
   #    using the unit vect of the self line and the perp vect of the input line
   #  - OR the end pt of the input line to the end of the self line
   #    using the unit vect of the self line and the perp vect of the input line
   # from which this member function is called
   #
   # by using this eqn: termPt1Self + A * UnitSelf = termPt1In + B * UnitPerp(Self or In)
   # OR                 termPt2Self + A * (-UnitSelf) = termPt2In + B * UnitPerp(Self or In)
   #
   #
   # RETURN:
   #   A, abs(B) - where if A > 0 this means that the input seg does NOT completely cover
   #   the self seg
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
   #
   # If A > 0 (when calculating both termPt1 and termPt2), this means that the inLine
   # is "within" the self line and that the pt on the self line where the perpendicular
   # of the inLine bisects the self line is WITHIN the self line (and not having to
   # project the self line outside of the bounds of the termPt1 and termPt2)
   def lineProjInputToSelfWithPerpVect(self, inLine, perpVect, projEndPt=False):
      # check that the secLine and the refLine are oriented in the same direction - if not , spit out warning
      if np.dot(self.unitVect, inLine.unitVect) < 0:
         print("ERROR - lineProjInputToSelfWithPerpVect expects self line and inLine to be in same direction")

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

   def lineProjInputToSelfWithInPerpSelfUnit(self, inLine, projEndPt=False):

      perpVect = np.array([-inLine.unitVect[1], inLine.unitVect[0]])

      return self.lineProjInputToSelfWithPerpVect(inLine, perpVect, projEndPt)

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

      perpVect = np.array([-self.unitVect[1], self.unitVect[0]])

      return self.lineProjInputToSelfWithPerpVect(inLine, perpVect, projEndPt)

   # check if line is a point (if start pt and end pt is the same)
   def checkIfLineIsPoint(self):
      if LA.norm(self.termPt2 - self.termPt1) < 0.02:
         print("line with start pt " + str(self.termPt1) + " and end pt " + str(self.termPt2) + " has become a point")
         return True

      return False

   # API to calculate the POINT EQUATION given a pair of lines with
   # the reference refLine (self) and secondary secLine (line 2)
   # the equation is:
   # A * unitVect_line1 + startPt_line1 + C * perpVect = B * unitVect_line2 + startPt_line2
   #
   # This equation has 4 free variables - A, B, C <- the user can define AT MOST 1
   # FREE VARIABLE. Otherwise the equation is overconstrained
   #  This vector equation is actually a system of 2 equations since vector equations
   #  have x, y component
   #
   #  perpVect is actually also a variable - for now it must be defined by the user
   #  perhaps in the future can make it so specify A,B,C and get the perpVect
   #
   #  rearranging the variables to solve for A, B, C get the eqns in the following form
   #
   #  A * unitVect_line1 + B * perpVect - C *unitVect_line2 = startPt_line2 - startPt_line1
   #
   #  The eqn above is actually 2 eqns - 1 for x and 1 for y
   #
   #  INPUT: line1 - whose unitVect is tied to A
   #         line2 - whose unitVect is tied to C
   #         ABCTuple - the values of (A,B,C) as a tuple in that order
   #         perpVect - perpVect to use
   #
   #  NOTE: the variables are solved in order A, B, C - thus if B is the FIXED variable
   #   the solved variables are A, C in that order
   #   if A is the FIXED variable - the solved variables are B, C in that order
   #   if C is the FIXED variable - the solved variables are A, B in that order
   #
   #  OUTPUT: tuple containing values (A,B,C, and the perpVect used)
   def calcPtEquationGiven2Lines(self, line2, ABCTuple, perpVect):
      # first need to check if there is ONLY 1 FIXED VARIABLE in the ABCTuple
      # ie. there should only be 1 non-None value in the tuple
      fixedVarCount = 0
      fixedVarIdx = None
      for i, x in enumerate(fixedVarCount):
         if x != None:
            fixedVarIdx = i
            fixedVarCount += 1

      if fixedVarCount != 1:
         print("{} - ONLY 1 FIXED VARIABLE in A,B,C tuple accepted - {} entered".format(__name__, fixedVarCount))
         return None

      # make sure perpVect is unit perpVect
      unitPerpVect = perpVect / np.linalg.norm(perpVect)

      eqn1 = []
      eqn2 = []
      consts = np.array([line2.getStartPt()[0] - self.getStartPt()[0], \
                         line2.getStartPt()[1] - self.getStartPt()[1]])

      # check if A is variable to solve or FIXED variable
      if ABCTuple[0]:
         # if FIXED variable need to subtract A * unitVect_line1 from the constants
         consts -= ABCTuple[0]*self.getUnitVect()
      else:
         # otherwise - add the A term to the 2x2 matrix on the LHS to solve
         eqn1.append(self.getUnitVect()[0])
         eqn2.append(self.getUnitVect()[1])

      # check if B is variable to solve or FIXED variable
      if ABCTuple[1]:
         # if FIXED variable need to subtract B * unitPerpVect
         const -= ABCTuple[1]*unitPerpVect
      else:
         # otherwise - add B term to 2x2 matrix on LHS to solve
         eqn1.append(unitPerpVect[0])
         eqn2.append(unitPerpVect[1])

      # check if C is variable to solve or FIXED variable
      if ABCTuple[2]:
         # if FIXED variable need to add C * unitVect_line2 to the constants
         const += ABCTuple[2]*line2.getUnitVect()
      else:
         # otherwise add C term to the 2x2 matrx on LHS to solve
         eqn1.append(-line2.getUnitVect()[0])
         eqn2.append(-line2.getUnitVect()[1])

      # now solve the 2x2 matrix
      L = np.array([eqn1, eqn2])
      try:
         ret = []
         soln = LA.solve(L, consts)
         for i,x in enumerate(soln):
            if i == fixedVarIdx:
               ret.append(ABCTuple[i])
            else:
               ret.append(x)
         ret.append(unitPerpVect)
         return ret

      except Exception as e:
         print("Failed to calculate point equation given {} - {}".format(ABCTuple, e))
         return None

#########################################################
######## GENERAL APIs not part of any class
#########################################################
def getSlopeOfVector(vect):
   if vect[0] == 0:
      if vect[1] < 0:
         return -float("inf")
      else:
         return float("inf")
   else:
      return vect[1] / vect[0]

# API to check if 2 lines are contiguous - if they are contiguous
#  return the shared point otherwise return None
def checkIf2LinesAreContiguous(line1, line2, **kargs):
   if np.array_equal(line1.termPt1, line2.termPt1) or \
      np.array_equal(line1.termPt1, line2.termPt2):
      return line1.termPt1

   if np.array_equal(line1.termPt2, line2.termPt1) or \
      np.array_equal(line1.termPt2, line2.termPt2):
      return line1.termPt2

   return None

def checkIf2LinesOverlap(line1, line2):
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

class contigSegCls:

   def __init__(self, lineIdx=None, line=None):
      # lists to store lines, linesAndCurves, and the original lines
      self.lines = []
      self.lineIdxs = []
      self.combinedLinesAndCurves = []
      self.origLines = []
      ###############
      self.startPt = None
      self.endPt = None
      self.unitVect = None
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

      self.hash = None

   def getStartPt(self):
      return self.startPt

   def getEndPt(self):
      return self.endPt

   def getUnitVect(self):
      try:
         self.unitVect = self.getEndPt() - self.getStartPt()
         self.unitVect /= np.linalg.norm(self.unitVect)
      except Exception as e:
         print("failed to get unit vect - %s" % (e))

      return self.unitVect

   def getPerpVects(self):
      try:
         if not self.perpVects:
            if not self.unitVect:
               self.getUnitVect()
            self.perpVects = [np.array([-self.unitVect[1], self.unitVect[0]]), \
                              np.array([self.unitVect[1], -self.unitVect[0]])]
      except Exception as e:
         print("failed to get perp vects - {}".format(e))

      return self.perpVects

   def getStartPtAsTuple(self):
      return tuple(self.getStartPt())

   def getEndPtAsTuple(self):
      return tuple(self.getEndPt())

   def getLHSMostTermPtAsTuple(self):
      return tuple(self.LHSMostTermPt)

   def getRHSMostTermPtAsTuple(self):
      return tuple(self.RHSMostTermPt)

   def getLength(self):
      if self.length == 0:
         for line in self.lines:
            self.length += line.getLength()
      return self.length

   def __hash__(self):
      ptsList = [self.startPt, self.endPt]
      ptsList.extend(self.intermedPts)
      hashTuple = tuple(sorted(list(map(tuple, ptsList)), key=lambda k: (k[0], k[1])))
      return hash(hashTuple)

   def __eq__(self, other):
      if self.hash and other.hash:
         return self.hash == other.hash

      return self.generateHashValueForContigSeg() == other.generateHashValueForContigSeg()

   def generateHashValueForContigSeg(self):
      self.hash = self.__hash__()
      return self.hash

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
         if insertedLineToContigSeg is not None:
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
      if insertedLineToContigSeg is not None:
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

         # since this API inserts the raw lineCls object from the image (or from an external calc)
         # and not from refactoring of the lineCls obj or internal calcs, insert the lineCls obj into
         # origLines as well
         self.origLines.append(line)
         # put the reference to the contig seg as the parent obj of the lineCls object
         if not insertIdxOnly:
            self.lines[-1].setPosInParent(self, len(self.lines)-1)

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

#### API to return the combined curves of the contig seg in sorted order according to
#### length - either from SHORTEST TO LONGEST or FROM LONGEST TO SHORTEST - by default
#### return from longest to shortest

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

   def setIntermedPts(self):
      self.intermedPts.clear()
      for i in range(len(self.lines)-1):
         self.intermedPts.append(self.lines[i].termPt2)
      self.generateHashValueForContigSeg()

   def clearIntermedData(self):
      self.dotProdBtwnLines.clear()
      self.orientations.clear()
      self.intermedPts.clear()

   def finalizeContigSeg(self):
      if len(self.lines) > 0:
         if self.orientContigSegLines():
            self.startPt = self.lines[0].termPt1
            self.endPt = self.lines[-1].termPt2
            self.unitVect = (self.endPt-self.startPt)/np.linalg.norm(self.endPt-self.startPt)

            self.clearIntermedData()

            self.getDotProdBtwnLines()
            self.getOrientationsOfLines()

            self.setIntermedPts()

            self.generateHashValueForContigSeg()

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

   # API to get the linesAndCombinedCurves in sorted order
   # reversed - from LONGEST to SHORTEST (default option)
   def getCombinedLinesAndCurvesSortedByLen(self, reversed=True):
      sortedCombined = {}
      for combinedObj in self.combinedLinesAndCurves:
         sortedCombined[combinedObj] = combinedObj.getLength()

      return dict(sorted(sortedCombined.items(), key=lambda it: it[1], reverse=reversed))

   # API for checking to see if adjacent lineCls objects (which act as joints)
   # can instead be converted into bezier curves
   #  THEN, check if contiguous bezier curves can be optimized and combined
   def convertContigSegToLinesAndBCurves(self, alphaMax, epsilonVal, verifyCurvesVal):
      # NOTE: the potential joints are generated from the midpts of the line segs
      # except for the 1st and line seg - take the midpt of the lineCls object
      # so that the curves are smooth since the midpt of a line will serve as the end pt of 1
      # curve and start pt of another curve ensuring smoothness
      linesAndCurves = []
      for i in range(len(self.lines)-1):
         # get the common pt between the 2 lines
         commonPt = checkIf2LinesAreContiguous(self.lines[i], self.lines[i+1])
         if commonPt is None:
            print("UNEXPECTED ERROR - lines %s and %s in contig seg are not contigous" % \
                 (i, i+1))
            self.lines[i].displayLineInfo()
            self.lines[i+1].displayLineInfo()
            continue
         # NOTE: for the first lineCls object - use the end pt of the 1st line
         # since it is the beginning of the contig seg
         if i == 0:
            line1Pt = self.lines[i].getOtherEndPt(commonPt)
         else:
            line1Pt = self.lines[i].lineMidPt

         # NOTE: for the last lineCls object - use the endpt of the last line
         # since it is the end of the contig seg
         if (i+1) == (len(self.lines)-1):
            line2Pt = self.lines[i+1].getOtherEndPt(commonPt)
         else:
            line2Pt = self.lines[i+1].lineMidPt

         line1 = lineCls(line1Pt, commonPt)
         line2 = lineCls(line2Pt, commonPt)
         alpha, Li = givenJointReturnAlphaAndLiSimpleVer(line1, line2, unitSquareSize)

         if alpha < alphaMax:
            controlPts = givenJointAndAlphaReturnControlPts(line1, line2, alpha)
            linesAndCurves.append(bezierCls(controlPts))
         else:
            linesAndCurves.append(line1)
            linesAndCurves.append(line2)

      # now - check if the contiguous curves can be combined into longer curves
      startIdx = 0
      endIdx = 1
      while startIdx < len(linesAndCurves):
         # if the startIdx is a line - simply increment both the start
         # and the end idx by 1
         if type(linesAndCurves[startIdx]) == lineCls:
            self.combinedLinesAndCurves.append(linesAndCurves[startIdx])
            startIdx += 1
            endIdx = startIdx + 1
         # the startIdx is a bezier curve - now check the endIdx
         else:
           # if the endIdx is a line must handle the span of curves between
           # startIdx and endIdx-1
           if type(linesAndCurves[endIdx]) == lineCls or \
              endIdx >= len(linesAndCurves):
              # if the span of startIdx and endIdx is greater than 1 (ie. if the startIdx
              # and the endIdx are not adjacent - combine the curves in that span)
              if endIdx-1 != startIdx:
                 # take the curves from startIdx to endIdx-1 and check to see if they can be combined
                 minNumCurves, curvesConfig = getMinNumCurvesConfig(linesAndCurves[startIdx, endIdx], epsilonVal, verifyCurvesVal)
                 # check if the curvesConfig has the same orientation as the uncombined curves
                 if np.array_equal(curvesConfig[0].getFirstPt(), linesAndCurves[startIdx].getFirstPt()):
                    print("first pt %s of first combined curve equal to first pt %s of not combined curve" % \
                          (curvesConfig[0].getFirstPt(), linesAndCurves[startIdx].getFirstPt()))
                 else:
                    print("first pt %s of first combined curve NOT equal to first pt %s of not combined curve" % \
                          (curvesConfig[0].getFirstPt(), linesAndCurves[startIdx].getFirstPt()))
                    curvesConfig.reverse()
                 self.combinedLinesAndCurves.extend(curvesConfig)
              else:
                 self.combinedLinesAndCurves.append(linesAndCurves[startIdx])
                 if endIdx < len(linesAndCurves):
                    self.combinedLinesAndCurves.append(linesAndCurves[endIdx])
              # now that we know endIdx is a line - move the startIdx to endIdx + 1
              # and move the endIdx so that its position is 1 farther than the startIdx
              startIdx = endIdx + 1
              endIdx = startIdx + 1
           # the endIdx is still a bezier curve - expand the span between start idx and end idx
           else:
              endIdx += 1

      # populate the contig seg as the parents of the lines and curves
      # in the combinedLinesAndCurves list
      for i in range(len(self.combinedLinesAndCurves)):
         self.combinedLinesAndCurves[i].setPosInParent(self, i)

      self.setIntermedPts()

   # API that takes in another contig seg and returns:
   #   fragments of the self contig seg that overlaps (is redundant) with the input
   #   contig seg - the fragments are returned as list of new contig segs, each
   #   of which is a fragment
   def getPortionsThatOverlapInputContigSeg(self, inputContigSeg):
      # orient the input contig seg so that it is in the same direction as the
      # self contig seg by taking the dot product of the unit vects of the 2 contig segs
      #  if dot product < 0 - this means that the direction of the 2 contig segs are opposite
      #  in that case - flip the inputContigSeg
      if np.dot(self.unitVect, inputContigSeg.unitVect) < 0:
         inputContigSeg.reverseOrientationOfContigSeg()

      # each line in this contig seg - loop thru each line seg in the inputContigSeg
      # to check if redundant
      remainFrags = []
      for line in self.lines:
         lineFrags = [line]
         for inLine in inpuptContigSegs.lines:
            # check if the line frags from the primary (reference) line seg
            # overlap with any portion of the curr line from the input contig seg
            newLineFrags = []
            for frag in lineFrags:
               if checkIf2LinesRedundant(frag, inLine):
                  # check the proj from the start pt first to see if
                  # the start of self seg is completely covered by input line seg
                  A,B = frag.lineProjInputToSelfWithSelfPerpSelfUnit(inLine, projEndPt=False)
                  if A > 0:
                     # this means that the start portion of the current self
                     # line seg is not cmpletely covered by line from input seg
                     startFragEndPt = frag.getStartPt() + A*frag.getUnitVect()
                     newLineFrags.append(lineCls(frag.getStartPt(), startFragEndPt))
                  # check the proj from the end pt to see if end pt of self seg
                  # is completely covered by input line seg
                  A,B = frag.lineProjInputToSelfWithSelfPerpSelfUnit(inLine, projEndPt=True)
                  if A > 0:
                     # this means that the end portion of the current self line seg
                     # is not completely covered by line from input seg
                     endFragStartPt = frag.getEndPt() - A*frag.getUnitVect()
                     newLineFrags.append(lineCls(endFragStartPt, frag.getEndPt()))
            # assign the new line frags to the "remaining" line frags that have yet to be found redundant
            lineFrags = newLineFrags
         # now that the self line has been checked against all lines from input contig seg
         # store the remaining frags (in lineFrags) in the remain frags
         remainFrags.extend(lineFrags)

      # now that fragments remain - check to see if any are contiguous
      # If so - group them into contig seg.
      # If not - push the singular seg into contig seg
      refIdx = 0
      secIdx = 0
      retContigSegs = []
      contigSeg = contigSegCls(remainFrags[0])
      for idx in range(1, len(remainFrags)):
         if not contigSeg.insertLineToContigSeg(remainFrags[idx].getHash(), remainFrags[idx], False):
            retContigSegs.append(contigSeg)
            contigSeg = contigSegCls(remainFrags[idx])

      return retContigSegs

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
         if len(self.lines) > 0:
            delLines = False
         else:
            delLines = True

         self.lines.clear()
         for idx in self.lineIdxs:
            try:
               self.lines.append(lineSegMap.get(idx))
               self.finalizeContigSeg()
            except:
               print("failed to calcContigSegMetadata at idx " + str(idx))
               print("lineSegMap is " + str(lineSegMap))
               self.displayContigsSegsData()
         #clear lines if previously no lines exist
         if delLines:
            self.lines.clear()

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

   # API to populate the lines in the contig seg given that only the line idxs
   # are stored in the contig seg. MUST TAKE AS INPUT a lineSegMap that maps line idx
   # to lines
   def populateLinesIntoContigSeg(self, lineSegMap):
      if self.lines:
         print("line objects are already stored in this contig seg")
         self.printContigSegInfo()
         return

      for idx in self.lineIdxs:
         if lineSegMap.get(idx):
            self.lines.append(lineSegMap[idx])

      self.printContigSegInfo()

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
########## contigSegCls END #############

def contigSegLenSort(contig):
   return contig.getLength()

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
      if checkIf2LinesOverlap(shortContig.lines[0], longContig.lines[idx]):
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
            if not checkIf2LinesOverlap(shortContig.lines[shortIdx], longContig.lines[longIdx]):
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

def checkIf2LinesIntersect(line1, line2, **kargs):
   retPt = None
   intersectPt = getIntersectPtBtwn2Lines(line1, line2)
   if intersectPt is not None and \
      line1.checkIfPointIsOnLine(intersectPt) and \
      line2.checkIfPointIsOnLine(intersectPt):
         retPt = intersectPt

   return retPt

def filterOutPtLinesInContigSeg(contigSegIn):
   retContigSeg = contigSegCls()
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
#
#  OLD API that only returns the line that is redundant and this condition is only
#  satisfied if the START AND END pts of line1 and line2 satisfy condition (ie.
#  the entire line1 and line2 are redundant)
def checkIf2LinesRedundant(line1, line2, **kargs):

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

      secLinePt1Dist = refLine.getDistBtwnPtAndStartOrEndOfLine(secLine.termPt1)
      secLinePt2Dist = refLine.getDistBtwnPtAndStartOrEndOfLine(secLine.termPt2, True)

      print("dot product between line1 and line2 is " + str(dotProdBtw2Lines))
      print("secLinePt1Dist is " + str(secLinePt1Dist) + " length of refLine is " + str(refLine.lineLength))
      print("secLinePt2Dist is " + str(secLinePt2Dist) + " length of refLine is " + str(refLine.lineLength))

      uVectDistFromStartPt = secLinePt1Dist[0]
      perpDistFromStartPt = secLinePt1Dist[1]
      uVectDistFromEndPt = secLinePt2Dist[0]
      perpDistFromEndPt = secLinePt2Dist[1]

      print("uVectDistFromStartPt is " + str(uVectDistFromStartPt) + " perpDistFromStartPt is " + str(perpDistFromStartPt) + " uVectDistFromEndPt is " + str(uVectDistFromEndPt) + " perpDistFromEndPt is " + str(perpDistFromEndPt) + " refLine length is " + str(refLine.lineLength))

      if (perpDistFromStartPt < kargs.get("maxPerpDist", 5.0)) and \
         (perpDistFromEndPt < kargs.get("maxPerpDist", 5.0)):
         if (uVectDistFromStartPt >= 0) and \
            (uVectDistFromEndPt >= 0):
            returnLine = delLineToReturn
         else:
            #  if ((normDistFromStartPt < 0) and (abs(normDistFromStartPt) < refLine.lineLength * self.dispFactor)) or \
            #     ((normDistFromEndPt < 0) and (abs(normDistFromEndPt) < refLine.lineLength * self.dispFactor)) :
            if ((uVectDistFromStartPt > 0) and (uVectDistFromStartPt < refLine.lineLength)) or \
               ((uVectDistFromEndPt > 0) and (uVectDistFromEndPt < refLine.lineLength)):
               returnLine = delLineToReturn

   return returnLine

# API to check if the 2 lines passed into the API are redundant / parallel
#  meaning that if the 2 lines can be replaced by just one line - this is due to
#  tracing a pencil drawing and you get 2 edges
#
#  This is a newer version of the API where instead of checking that BOTH start and end pt
#  satisfy the condition of perp dist < max - check if either the start pt or the end pt
#  satisfies the condition - if ONLY 1 of them do, move along the line to find the other
#  "last" pt that satisfies this condition so that the redund seg
#  is max len and the perp dist of each pt in that seg is < max
#   - return the 2 portions that are redundant
def checkIf2LinesRedundantV2(line1, line2, **kargs):

   defaultMinDotProdParallel = 0

   dotProdBtw2Lines = np.dot(line1.unitVect, line2.unitVect)
   print("dot is " + str(dotProdBtw2Lines) + " line1.unitVect " + str(line1.unitVect) + " line2.unitVect " + str(line2.unitVect))
   if LA.norm(dotProdBtw2Lines) >= defaultMinDotProdParallel:
      # check if the 2 lines are oriented in opposite directions
      # if so, flip line2
      if dotProdBtw2Lines < 0:
         line2.flipLine()
         print("dot product between line1 and line2 is negative - flip line2")

      # now to determine the segments that are redundant :
      #  to get the start and end pts of the segs that are redundant,
      #  do the following:
      #   start pt -> check which of the lines has start pt that is between other line
      #   the other line shall be the reference line (not refline as above)
      line1TermPt1Disp = line1.getDistBtwnPtAndStartOrEndOfLine(line2.termPt1)
      line2TermPt1Disp = line2.getDistBtwnPtAndStartOrEndOfLine(line1.termPt1)
      # NOTE: getDistBtwnPtAndStartOrEndOfLine returns A, B, perpVect where A is the
      #       len along the unitVect, B is the len along the perpVect, and perpVect
      #       is the perpVect of the line that yields B > 0
      if line1TermPt1Disp[0] >= 0:
         termPt1DispConfig = line1TermPt1Disp
      elif line2TermPt1Disp[0] >= 0:
         termPt1DispConfig = line2TermPt1Disp
      else:
         print("ERROR - termPt1 of neither line1 nor line2 is between the line of the other")
         line1.displayLineInfo()
         line2.displayLineInfo()
         return None

      # end pt -> check which of the lines has end pt that is between other line
      #  the other line shall be the reference line (not refline as above)
      line1TermPt2Disp = line1.getDistBtwnPtAndStartOrEndOfLine(line2.termPt2)
      line2TermPt2Disp = line2.getDistBtwnPtAndStartOrEndOfLine(line1.termPt2)

      if line1TermPt2Disp[0] <= line1.getLength():
         termPt2DispConfig = line1TermPt2Disp
      elif line2TermPt2Disp[0] <= line2.getLength():
         termPt2DispConfig = line2TermPt2Disp
      else:
         print("ERROR - termPt2 of neither line1 nor line2 is between the line of the other")
         line1.displayLineInfo()
         line2.displayLineInfo()
         return None

      # now check:
      #  - if the perp dist of the termPt1 and termPt2 of the 2 lines > maxPerpDist
      #  - if the 2 lines intersect
      #  if none of the above conditions are satisfied - the 2 lines are not redundant
      if calcIntersectionOf2Lines(refLine, secLine) or \
         termPt1DispConfig[1] <= kargs.get("maxPerpDist", 5.0) or \
         termPt2DispConfig[1] <= kargs.get("maxPerpDist", 5.0):
         # to get the termPt1 configuration:
         #  1st - check if the perp dist from the line to the termPt1 of its corresponding line
         #  is <= maxPerpDist - if so then set the termPt1 of the line as the start pt of the
         #  redundant portion
         if termPt1DispConfig[1] <= kargs.get("maxPerpDist", 5.0):
            if termPt1DispConfig is line1TermPt1Disp:
               line1TermPt1 = line1.getStartPt() + termPt1DispConfig[0] * line1.getUnitVect()
               line2TermPt1 = line2.getStartPt()
            elif termPt1DispConfig is line2TermPt2Disp:
               line1TermPt1 = line1.getStartPt()
               line2TermPt1 = line2.getStartPt() + termPt1DispConfig[0] * line2.getUnitVect()
         # if the perp dist from the line to the termPt1 of its corresponding line is
         # > maxPerpDist - need to calculate the point on the 2 lines where the perp dist is
         #  equal to the maxPerpDist
         else:
            if termPt1DispConfig is line1TermPt1Disp:
               termPt1Soln = line1.calcPtEquationGiven2Lines(line2, (None, kargs.get("maxPerpDist", 5.0), None), termPt1DispConfig[2])
               line1TermPt1 = line1.getStartPt() + termPt1Soln[0] * line1.getUnitVect()
               line2TermPt1 = line2.getStartPt() + termPt1Soln[2] * line2.getUnitVect()
            elif termPt1DispConfig is line2TermPt1Disp:
               termPt1Soln = line2.calcPtEquationGiven2Lines(line1, (None, kargs.get("maxPerpDist", 5.0), None), termPt1DispConfig[2])
               line1TermPt1 = line1.getStartPt() + termPt1Soln[2] * line1.getUnitVect()
               line2TermPt1 = line2.getStartPt() + termPt1Soln[0] * line2.getUnitVect()

         # to get the termPt2 configuration:
         # 1st - check if the perp dist from the line to the termPt2 of its corresponding line
         #  is <= maxPerpDist - if so then set the termPt2 of the line as the end pt of the redundant portion
         if termPt2DispConfig[1] <= kargs.get("maxPerpDist", 5.0):
            if termPt2DispConfig is line1TermPt2Disp:
               line1TermPt2 = line1.getStartPt() + termPt2DispConfig[0] * line1.getUnitVect()
               line2TermPt2 = line2.getEndPt()
            elif termPt2DispConfig is line2TermPt2Disp:
               line1TermPt2 = line1.getEndPt()
               line2TermPt2 = line2.getStartPt() + termPt2DispConfig[0] * line2.getUnitVect()
         # if the perp dist from the line to the termPt2 of its corresponding line is
         # > maxPerpDist - need to calculate the point on the 2 lines where the perp dist is
         # equal ot the maxPerpDist
         else:
            if termPt2DispConfig is line1TermPt2Disp:
               termPt2Soln = line1.calcPtEquationGiven2Lines(line2, (None, kargs.get("maxPerpDist", 5.0), None), termPt2DispConfig[2])
               line1TermPt2 = line1.getStartPt() + termPt2Soln[0] * line1.getUnitVect()
               line2TermPt2 = line2.getStartPt() + termPt2Soln[2] * line2.getUnitVect()
            elif termPt2DispConfig is line2TermPt2Disp:
               termPt2Soln = line2.calcPtEquationGiven2Lines(line1, (None, kargs.get("maxPerpDist", 5.0), None), termPt2DispConfig[2])
               line1TermPt2 = line1.getStartPt() + termPt2Soln[2] * line1.getUnitVect()
               line2TermPt2 = line2.getStartPt() + termPt2Soln[0] * line2.getUnitVect()

         line1Redund = lineCls(line1TermPt1, line1TermPt2)
         line2Redund = lineCls(line2TermPt1, line2TermPt2)

         return line1Redund, line2Redund

      else:
         print("line1 and line2 are not redundant - return None")
         line1.displayLineInfo()
         line2.displayLineInfo()
         return None

# API to return reverse of the rotation matrix given a
# rotation matrix
# reverse rotation is simply rotation but with the negative of the angle
#  using properties cos(-A) = cos(A) and sin(-A) = sin(A)
def getReverseRotationMatrix(rotMatrix):
   return(np.array([[rotMatrix[0][0], -rotMatrix[0][1]], \
                    [-rotMatrix[1][0], rotMatrix[1][1]]]))

# API that given 2 connected lines (joint) and alpha, return
#  the control pts that are calculated using potrace algorithm
def givenJointAndAlphaReturnControlPts(self, line1, line2, alpha):
   if checkIf2LinesAreContiguous(line1, line2) is None:
      print("line1 and line2 not connected - cannot reorient")
      line1.displayLineInfo()
      line2.displayLineInfo()
      return None

   if np.array_equal(line1.termPt1, line2.termPt1):
      p0 = line1.termPt2
      p1 = line1.unitVect * (1-alpha) + line1.termPt1
      p2 = line2.unitVect * (1-alpha) + line1.termPt1
      p3 = line2.termPt2
   elif np.array_equal(line1.termPt1, line2.termPt2):
      p0 = line1.termPt2
      p1 = line1.unitVect * (1-alpha) + line1.termPt1
      p2 = line2.unitVect * alpha + line2.termPt1
      p3 = line2.termPt1
   elif np.array_equal(line1.termPt2, line2.termPt1):
      p0 = line1.termPt1
      p1 = line1.unitVect * alpha + line1.termPt1
      p2 = line2.unitVect * (1-alpha) + line2.termPt1
      p3 = line2.termPt2
   elif np.array_equal(line1.termPt2, line2.termPt2):
      p0 = line1.termPt1
      p1 = line1.unitVect * alpha + line1.termPt1
      p2 = line2.unitVect * alpha + line2.termPt1
      p3 = line2.termPt1
   else:
      print("unexpected line1 termPt1 %s termPt2 %s ; line2 termPt1 %s termPt2 %s" % \
            (line1.termPt1, line1.termPt2, line2.termPt1, line2.termPt2))
      return None

   return (p0, p1, p2, p3)

# API to orient 2 connected lines (joint) so that:
#  1) The pt that is common to both lines is (0,0)
#  2) The other pt on the first line is (0,y) where y < 0
#  3) The other pt on the second line is (x,y) where x > 0
#
# return:
#    biMinus1, ai, bi, shift, rotationMat, xFlip
def potraceOrientLines(line1, line2):

   # make sure that the 2 lines are connected - if not,
   # orient the lines to make sure they are
   if checkIf2LinesAreContiguous(line1, line2) is None:
      print("line1 and line2 not connected - cannot reorient")
      line1.displayLineInfo()
      line2.displayLineInfo()
      return None

   # get the pts bi-1, bi and ai
   if np.array_equal(line1.termPt1, line2.termPt1):
      ai = copy.deepcopy(line1.termPt1)
      biMinus1 = copy.deepcopy(line1.termPt2)
      bi = copy.deepcopy(line2.termPt2)
   elif np.array_equal(line1.termPt1, line2.termPt2):
      ai = copy.deepcopy(line1.termPt1)
      biMinus1 = copy.deepcopy(line1.termPt2)
      bi = copy.deepcopy(line2.termPt1)
   elif np.array_equal(line1.termPt2, line2.termPt1):
      ai = copy.deepcopy(line1.termPt2)
      biMinus1 = copy.deepcopy(line1.termPt1)
      bi = copy.deepcopy(line2.termPt2)
   elif np.array_equal(line1.termPt2, line2.termPt2):
      ai = copy.deepcopy(line1.termPt2)
      biMinus1 = copy.deepcopy(line1.termPt1)
      bi = copy.deepcopy(line2.termPt1)
   else:
      print("unexpected line1/2 termPt combination")
      print("line1: termPt1 = %s termPt2 = %s" % (line1.termPt1, line1.termPt2))
      print("line2: termPt1 = %s termPt2 = %s" % (line2.termPt1, line2.termPt2))

   # first do linear translation so that ai is at 0,0
   shift = -ai
   biMinus1 += shift
   bi += shift
   ai += shift

   # now do the rotation so that so that biMinus1 will be at (0, y) where
   # y < 0
   biMinus1UnitVect = biMinus1/np.linalg.norm(biMinus1)
   negYAxisRefVect = np.array([0, -1])
   cosA = np.dot(biMinus1UnitVect, negYAxisRefVect)
   # claculating sinA is more involved since it can be +ve or -ve
   biMinus1UnitVect3d = np.append(biMinus1UnitVect, 0)
   angleDir = np.cross(biMinus1UnitVect3d, negYAxisRefVect)[2]
   try:
      if angleDir == 0:
         angleDir = 1
      else:
         angleDir /= abs(angleDir)
   except Exception as e:
      print("failed to get angle direction - %s" % (e))
      return None

   sinA = math.sqrt(1-cosA**2)*angleDir

   # now - perform rotation of biMinus1 and bi using this matrix
   # cosA    -sinA
   # sinA    cosA
   rotationMat = np.array([[cosA, -sinA], [sinA, cosA]])

   biMinus1 = np.matmul(rotationMat, biMinus1)
   bi = np.matmul(rotationMat, bi)

   # final step - if x- of bi is < 0 -> flip it so that it is > 0 by multiplying -1
   if bi[0] < 0:
      xFlip = -1
      bi[0] *= -1
   else:
      xFlip = 1

   # return both:
   #   1) the transformed points bi-1, ai, bi (where ai is the pt shared
   #      between the 2 lines)
   #   2) the translation / rotation / flip parameters
   return biMinus1, ai, bi, shift, rotationMat, xFlip

# API that takes in a joint (2 lines that are connected) and performs transformations
# so that:
#  the point of connection is moved to 0,0
#  joint is oriented so that the non-connection point of the 1st line is (0,y where y < 0)
#                            the non-connection point of the 2nd line is such that x > 0

# API that uses potrace's method of determining whether 2 connected lines should be
# combined as a cubic bezier curve by determining whether the 2
# lines bend at a specific angle and have certain lengths
#
# ================= (Reference 1)
# The method is:
#  align the 2 connected lines (which we call a joint) such that the pt of connection is translated to (0,0)
#  The 1st line is oriented such that the other pt of the line (that is not)
#  the connection pt is (0, y) where y < 0 - to do this reorientation, the
#  joint is rotated a certain angle after linear shift to (0,0)
#  Since the entire joint is rotated, rotate the second line by that same angle and direction as well
#
#  If the pt on the second line that is not the common point is oriented such that
#  its x < 0, flip that pt along the y axis such that its x > 0
#
# With this orientation - apply the potrace algo to determine whether this should be a curve or remain a joint
# NOTE: check to make sure that the 2 lines cannot be combined in a straight line - otherwise
#  this will cause the matrix calculations to return error
#  With this orientation, now get the line Ci-1_Ci which we denote at Li, where Ci-1 is a pt between
#  bi-1 and ai (where bi-1 is the midpt of the 1st line and a1 is the connection pt)
#  and Ci is a pt between ai and bi, where bi is the midpt of the second line
#  Ci-1_Ci is such that this line touches the unit square surrounding ai but is closest to
#  bi-1_bi AND Ci-1_Ci is // to bi-1_bi
#    -> since we know that the lower right corner of the unit square is 0.5, -0.5
#  we solve system of equation where solve:
#    Ci-1x + AUx = XlowerRHSCorner
#    Ci-1y + AUy = YlowerRHSCorner
#   -> here we get A and Ci-1y
#  Next we calculate pt Ci
#   Ci-1x + BUx = aix + DVx
#   Ci-1y + BUy = aiy + DVy
# NOTE Ux, Uy is the unit vector of the line Ci-1, Ci which is equal to bi-1, bi
#      Vx, Vy is the unit vector of the line ai_bi
#   If A < B, then calculate ALPHA = 4 * gamma / 3 where gamma is
#    bi-1_ci-1 / bi-1_ai
#   if Ci-1y < bi-1 -> set ci-1y = bi
# However, if the solutions for the matrix calc above violate the geometry
#  such as A > B or Ci-1y > 0 then must solve the system of eqns where we check
#  where lines Ci-1_Ci and ai_bi touch the pt where ai_bi cross the bottom edge
# This is because the line Ci-1_Ci that is closest to bi-1_bi now touches
# the bottom edge of the square (Xcross, Y)
#  of the unit square (Xcross, -0.5)
# aix + FVx = Xcross
# aiy + FVy = Ycross
# Ci-1x + GUx = Xcross
# Ci-1y + GUy = Ycross
#  If Ci-1y < bi-1y OR Ci-1y > 0 -> set Ci-1y = bi-1
#     if Ci-1 y > -0.5, Ci-1y = -0.5
def givenJointReturnAlphaAndLi(line1, line2):
   biMinus1, ai, bi, shift, rotationMat, xFlip = potraceOrientLines(line1, line2)
   reverseRotationMat = getReverseRotationMatrix(rotationMat)
   # there are 2 classes of the rotated joint:
   #  CLASS 1 - the angle between ai_bi-1 and ai_bi is < 45 degrees
   #   45 degrees is important because that is the angle ai_bi-1 makes with the
   #   line ai_bi if ai_bi crosses the lower RHS corner (0.5, -0.5) of the unit square
   #   surrounding ai - in this case there is no amount of projection where any line Ci-1_Ci
   #   will touch the lower RHS corner of the square - thus in this class of lines we must
   #   check the lines Ci-1_Ci touching the bottom edge of the square (XCross, -0.5)
   # CLASS 2 - the angle between ai_bi-1 and ai_bi is >= 45 dgrees since in this joint
   #   there exists a line Ci-1_Ci that does touch the lower RHS corner (0.5, -0.5)
   #
   #  can only do this if
   #  first get unit vector of line Ci-1_Ci -> we know that this line must be parallel to
   #  bi-1_bi - denote this unit vector as unitVectC
   unitVectC = (bi - biMinus1) / np.linalg.norm(bi - biMinus1)
   # also need to get unit vect of line ai_bi
   unitVectAiBi = (bi - ai) / np.linalg.norm(bi - ai)
   unitVectAiBiMinus1 = (biMinus1 - ai) / np.linalg.norm(biMinus1 - ai)
   # 45 degrees is: cos(45) = 1 / sqrt(2) and 0 <= angle <= 45 would have
   #  cos(angle) >= 1 / sqrt(2) and angle > 45 would have cos(angle) < 1 / sqrt(2)
   # to calculate cos(angle) take the dot product of unitVectAiBiMinus1 and unitVectAiBi
   cosAngle = np.dot(unitVectAiBiMinus1, unitVectAiBi)
   if cosAngle <= (1 / math.sqrt(2)):
      # must solve this set of equations
      #    Ci-1x + AUx = XlowerRHSCorner
      #    Ci-1y + AUy = YlowerRHSCorner
      #   Ci-1x + BUx  = aix + DVx
      #   Ci-1y + BUy  = aiy + DVy
      # where U is the unit vect of line Ci-1_Ci
      #       V is the unit vect of line ai_bi
      # the unknowns are Ci-1y, A, B, D and so the elements of the resulting vector
      #  will be in that order
      A1 = np.array([[0,unitVectC[0],0,0], [1,unitVectC[1],0,0], \
                    [0,0,unitVectC[0],-unitVectAiBi[0]], [1,0,unitVectC[1],-unitVectAiBi[1]]])
      B1 = np.array([0.5,-0.5,0,0])

      Cy, A, B, D = np.linalg.solve(A1,B1)

      # this means that the line Ci-1_Ci touches the corner (0.5, -0.5)
      # which means that it is the closest line to bi-1_bi => take the Ci-1y
      # as calculated and calculate gamma
      if A <= B and Cy > biMinus1[1]:
         gamma = (math.abs(Cy - biMinus1[1])) / (math.abs(biMinus1[1]))
         alpha = 4 * gamma / 3
         CiMinus1_Ci = [Cy, ai + D * unitVectAiBi]
      else:
         alpha = 0
         CiMinus1_Ci = [biMinus1, bi]
   else:
      # Ci-1_Ci does not touch the corner (0.5, -0.5) but is less than biMinus1[1] (y coord of bi-1)
      # Thus, need to calculate the set of equations that tells us where the line Ci-1_Ci
      # touches the lower edge of the cube (Xcross, -0.5)
      # the set of equations to solve is:
      # aix + FVx = Xcross
      # aiy + FVy = Ycross
      # Ci-1x + GUx = Xcross
      # Ci-1y + GUy = Ycross
      # the unknowns in this set of equation are Ci-1y, F, G, Xcross, and so the elements
      # of the resulting vector are in this order
      A2 = np.array([[0,unitVectAiBi[0],0,-1],[0,unitVectAiBi[1],0,0],\
                     [0,0,unitVectC[0],-1],[1,0,unitVectC[1],0]])
      B2 = np.array([0,-0.5,0,-0.5])

      Cy2, F, G, Xcross = np.linalg.solve(A2, B2)

      # if Ci-1y is less than bi-1 OR greater than 0, set Ci-1y to bi-1, which yields
      # gamma = 0 => alpha = 0
      if Cy2 < biMinus1[1] or \
         Cy2 > 0:
         alpha = 0
         CiMinus1_Ci = [biMinus1, bi]

      # need to check if Cy2 is inside the unit square - if it is, then this means
      # that the Ci-1_Ci line that is closer to bi-1_bi is where Ci-1 is at -0.5
      # since there are 2 candidates of the line Ci-1_Ci -> either Ci-1 OR Ci touches the
      # bottom edge of the unit square (x, -0.5)
      else:
         # if Ci-1 > -0.5 - this means that the Ci-1_Ci that is closest to bi-1_bi
         #  and still touches the unit square is if Ci-1 is -0.5. In this case, need to calculate
         #  Ci. Since we know that Ci-1 must be at (0, -0.5), this means that we can use the ratio of
         #  0.5 / len(ai_bi-1) to get Ci by multiplying this ratio by the len of ai_bi (in the direction
         #  of the unit vect of ai_bi) and add to ai to get the pt Ci
         if Cy2 > -0.5:
            Cy2 = -0.5
            Ci = ai + (0.5/np.linalg.norm(bi)) * np.linalg.norm(bi-ai) * unitVectAiBi
         else:
            Ci = ai + F*unitVectAiBi

         CiMinus1_Ci = [Cy2, Ci]
         gamma = (math.abs(Cy2 - biMinus1[1]))/(math.abs(biMinus1[1]))
         alpha = 4 * gamma / 3

   # now that CiMinus_Ci has been calculated in the rotated frame of reference where
   #  ai = (0,0), bi-1 = (-y, 0) and bi = (x,y) where x, y > 0 - reverse the linear
   #  transformation so that the line CiMinus1_Ci is returned in its original frame of reference

   # First do the xFlip back to its original x values if xFlip was done
   if xFlip < 0:
      xForm = np.array([xFlip, 1])
      CiMinus1_Ci *= xFlip

   # Next reverse the rotation that was applied to bi-1, ai, and bi
   CiMinus1_Ci = [np.matmul(reverseRotationMat, CiMinus1_Ci[0]), \
                  np.matmul(reverseRotationMat, CiMinus1_Ci[1])]

   # finally - reverse the shift that translated ai to (0,0)
   CiMinus1_Ci += shift

   # generate the lineCls object for CiMinus_Ci
   Li = lineCls(CiMinus1_Ci[0], CiMinus1_Ci[1])

   return alpha, Li

# API that generates the control pts of a cubic bezier curve given the joint
# Takes as input all of the properties needed to define a joint
#  (biMinus1, ai, bi, alpha)
def getControlPtsFromJoint(biMinus1, ai, bi, alpha):
   z0 = biMinus1
   z3 = bi

   biMinus1_ai_vect = ai-biMinus1
   biMinus1_ai_norm = np.linalg.norm(biMinus1_ai_vect)
   biMinus1_ai_unitVect = biMinus1_ai_vect/biMinus1_ai_norm
   z1 = biMinus1_ai_unitVect*biMinus1_ai_norm*alpha + biMinus1

   bi_ai_vect = ai-bi
   bi_ai_norm = np.linalg.norm(bi_ai_vect)
   bi_ai_unitVect = bi_ai_vect / bi_ai_norm
   z2 = bi_ai_unitVect*bi_ai_norm*alpha + bi

   return [z0, z1, z2, z3]


# API that returns the joint given the length of the arms of the joint
#  and the alpha desired (the alpha in a way indicates the angle between
#  biMinus1_ai and ai_bi)
#  this api has constraint where biMinus1_ai and ai_bi must be the same length
#  will use the angle phi as proxy to indicate the angle of biMinus1_ai and ai_bi
#  phi is the angle between the positive y-axis (0,1) and ai_bi
#
# Definition of standard joint (StdJoint) =
#   joint where biMinus1 = (0,-y), ai = (0,0), bi = (x,y) where x > 0
#
#  This API takes as input armLen - this means that biMinus1 is constrained
#   - also takes in alpha
#   - the variable that must be calculated is bi - which is dependent on alpha
#   - the length of ai_bi is fixed, which means that the angle phi must be determined
#   that would yield a joint with a certain alpha
def givenArmLenAndAlphaRetStdJoint(armLen, alpha, unitSquareSize=1.0):
   line1 = lineCls(np.array([0, -armLen]), np.array([0,0]))

   maxPhi = math.radians(179.0)
   aibiMaxAngle = -(maxPhi - math.radians(90.0))
   biCandidate = np.array([armLen*math.cos(aibiMaxAngle), armLen*math.sin(aibiMaxAngle)])

   line2 = lineCls(np.array([0,0]), biCandidate)

   candidateAlpha, Li = givenJointReturnAlphaAndLiSimpleVer(line1, line2, unitSquareSize)

   if alpha > candidateAlpha:
      print("joint with length %s and unit square size %s cannot have alpha %s - max alpha is %s" % \
            (armLen, unitSquareSize, alpha, candidateAlpha))
      return None

   phiMin = 1.0
   phiMax = 179.0
   while abs(candidateAlpha - alpha) > 0.1:
      phiCandidate = math.radians(phiMin + (phiMax-phiMin)/2.0)
      # aibiAngle is the angle aibi makes with the positive x-axis
      aibiAngle = -(phiCandidate - math.radians(90.0))
      biCandidate = np.array([armLen*math.cos(aibiAngle), armLen*math.sin(aibiAngle)])
      line2 = lineCls(np.array([0,0]), biCandidate)
      candidateAlpha, Li = givenJointReturnAlphaAndLiSimpleVer(line1, line2, unitSquareSize)
      print("candidate alpha is %s - desired alpha is %s" % (candidateAlpha, alpha))
      # if the candidate alpha is GREATER THAN the desired alpha - this means that
      #  phi must be decreased
      # otherwise (if the candidate alpha is LESS THAN desired alpha - this means
      # that phi must be increased)
      if candidateAlpha > alpha:
         phiMax = math.degrees(phiCandidate)
      else:
         phiMin = math.degrees(phiCandidate)

   biMinus1 = np.array([0, -armLen])

   return biMinus1, np.array([0,0]), biCandidate


# Alternate way to calculate alpha, Li from the API above
# Still using the algo from potrace - but simplify calculation
# calculate Ci-1 with the following knowns:
#   1) Ci-1x = 0, 2) line touches lower RHS corner (0.5, -0.5) of unit square
#   calculate Ci-1y - and thus get gamma / alpha
#  To get the line Li - first need to get Ci
#  To get Ci - calculate t (the parametrization of where the pt Ci-1 lies on
#  the line bi-1_ai) using the equation p = (1-t)*p0 + t*p1
#  use this t to calculate Ci - given Ci can calculate Li
def givenJointReturnAlphaAndLiSimpleVer(line1, line2, unitSquareSize=1.0):
   biMinus1, ai, bi, shift, rotationMat, xFlip = potraceOrientLines(line1, line2)
   angle = math.degrees(math.asin(rotationMat[1][0]))
   print("Transformed coordinates: bi-1 = %s : ai = %s : bi = %s" % (biMinus1, ai, bi))
   print("Tranform params: shift = %s : rotationMat = %s : angle = %s : xFlip = %s" \
          % (shift, rotationMat, angle, xFlip))

   biMinus1_bi_unitVect = (bi - biMinus1) / np.linalg.norm(bi - biMinus1)
   #
   #  This algo is divided into 2 cases:
   #   1) if the angle of the joint is greater than 45 degrees - then we must use the lower RHS
   #      corner of the unit square to determine the Li that is closest to biMinus1_bi and is
   #      parallel to bi1Minus1_bi
   #   2) If the angle of the joint is less than 45 degrees - then we must use the bottom edge
   #      of the square to determine the Li that is closest to biMinus1_bi since the joint does not
   #      span over the lower RHS corner of the unit square
   ai_biMinus1_vect = (ai - biMinus1) /  np.linalg.norm(ai - biMinus1)
   ai_bi_vect = (ai - bi) / np.linalg.norm(ai - bi)
   cosTheta = np.dot(ai_biMinus1_vect, ai_bi_vect)

   #  Determine angle by its cosTheta - cos(45) = 1 / sqrt(2) - if cosTheta <= 1 / sqrt(2) - this means
   #  that the angle is greater than 45 degrees
   if cosTheta <= (1 / math.sqrt(2)):
      # Can solve the parameter A calculating Ci-1x + AUx = 0.5, where:
      #   Ci-1x is 0 since the first line is constrained so that it lies on the y-axis
      #   Ux = biMinus1_bi_unitVectX
      #   A is the parameter to solve
      #   0.5 is the lower RHS corner of the unit square
      #  The equation above calculates A which is needed to calculate Ci-1y
      #    which then yields gamma and ultimately alpha since gamma = the ratio
      #   of bi-1_ci-1 / bi-1_ai
      try:
         A = (unitSquareSize/2) / biMinus1_bi_unitVect[0]
         # After solving A - can solve Ci-1y with the equation
         #  Ci-1y + AUy = -0.5, where:
         #   Uy = biMinus1_bi_unitVectY
         #   -0.5 is the lower RHS corner of the unit square
         ciMinus1Y = -(unitSquareSize/2) - A * biMinus1_bi_unitVect[1]
      except ZeroDivisionError:
         print("biMinus1_bi is a vertical line - return original line and alpha as float('inf')")
         alpha = float('inf')
         ciMinus1_ci = np.array([biMinus1, bi])
      except Exception as e:
         print("Error in angle joint spanning lower RHS corner - %s" % (e))
         print(traceback.format_exc())
         alpha = float('inf')
         ciMinus1_ci = np.array([biMinus1, bi])
   #  else case - angle is less than 45 degrees - meaning that the lower RHS corner
   #   of the unit square is not within span of the joint
   else:
      try:
         # if the lower RHS corner is not spanned by the joint - then the line Li
         # that is closest to biMinus1_bi touches the lower edge of the square (x, -0.5)
         # first need to determine if the unit vector of Li has +ve or -ve y-val
         #    if +ve y-val, this means that the endpt on Li that lies between ai_bi
         #    touching the lower edge of the unit square shall be closest to biMinus1_bi
         #    otherwise, the endpt on Li that lies between biMinus_ai shall be closest to
         #    biMinus_bi
         # note that Li is parallel to biMinus_bi so they share the same unit vect
         LiUnitVect = (bi-biMinus1) / np.linalg.norm(bi-biMinus1)
         if LiUnitVect[1] > 0:
            # this means that the endpt of Li that lies on ai_bi and touches the lower
            # edge of the unit square (x, -0.5) is closest to biMinus1_bi
            # Need to get the t value of the end pt of Li along the ai_bi line
            # given the parametrization eqn y = (1-t) * aiy + t * biy
            #   calculate t given:
            #      y = -0.5, aiy = 0, and biy is known
            t = min(-(unitSquareSize/2) / bi[1], 1.0)
         else:
            # this means that the endpt of Li that lies on biMinus1_ai and touches
            # the lower edge of the unit square (x, -0.5) is closest to biMinus1_bi
            # thus - solve t from this equation:
            #   y = (1-t) * aiy + t * biMinus1Y
            #    where y = -0.5 <- lower edge of square, aiy = 0
            t = min(-(unitSquareSize/2) / biMinus1[1], 1.0)
         # with the calculated t can get the ci-1Y since it will have the same t
         #  solve this equation: y = (1-t) * aiy + t * biMinus1y
         #   where t is calculated above and aiy = 0, making y = ciMinus1Y
         ciMinus1Y = t * biMinus1[1]
      except Exception as e:
         print("Error in angle joint contained in lower edge - %s" % (e))
         print(traceback.format_exc())
         alpha = float('inf')
         ciMinus1_ci = np.array([biMinus1, bi])

   # now - calculate alpha thru gamma
   # NOTE that because the joint is constrained such that ai = (0,0)
   #  and biMinus1 is constrained so that it is (0,y) - to get the length
   #  of bi-1_ci-1 simply subtract the y-values of biMinus1 with ciMinus1
   #  and bi-1_ai by taking the y-value of bi-1 since ai is (0,0)
   # if gamma < 0 - this means that ci-1 lies lower on the y-axis than bi-1 -
   #  in which case return alpha = 0 and Li = bi-1_bi
   gamma = max((ciMinus1Y - biMinus1[1]) / -biMinus1[1], 0)
   # to calculate ci (the pt on Li which lies between a and bi) - use parametrization
   # to do so - since we know that Li is parallel to biMinus_bi - whatever the
   # linear factor t is at the pt ciMinus1Y if we start moving from a (0,0), that same
   # t will determine the x, y of ci if we start moving from a to bi
   # given ai = 0 and bi-1X = 0 -> solve eqn
   #   ciMinus1Y = t * bi-1Y
   #  Then ciX = t * biX, ciY = t * biY
   alpha = 4 / 3 * gamma
   t = min(ciMinus1Y / biMinus1[1], 1.0)
   # Do the xFlip back to its original x values if xFlip was done
   ciMinus1_ci = np.array([np.array([0, t*biMinus1[1]]), np.array([xFlip*t*bi[0], t*bi[1]])])

   print("Potrace transformed coords with orig xFlip: ci-1 = %s : ci = %s" % (ciMinus1_ci[0], ciMinus1_ci[1]))

   # now that ciMinus_ci has been calculated in the rotated frame of reference where
   #  ai = (0,0), bi-1 = (-y, 0) and bi = (x,y) where x, y > 0 - reverse the linear
   #  transformation so that the line ciMinus1_ci is returned in its original frame of reference
   reverseRotationMat = getReverseRotationMatrix(rotationMat)

   # Next reverse the rotation that was applied to bi-1, ai, and bi
   # and apply that reverse rotation to ciMinus1_ci
   ciMinus1_ci = np.transpose(np.matmul(reverseRotationMat, np.transpose(ciMinus1_ci)))

   # finally - reverse the shift that translated ai to (0,0)
   ciMinus1_ci -= shift

   print("Orig line coords - ci-1 = %s : ci = %s" % (ciMinus1_ci[0], ciMinus1_ci[1]))

   # generate the lineCls object for ciMinus_ci
   Li = lineCls(ciMinus1_ci[0], ciMinus1_ci[1])

   return alpha, Li

# API to generate joints with a given alpha (that can be used to generate bezier curves)
#  This problem is 5-dimensional problem with the following parameters
#    - alpha (by extension gamma - which is ciMinus1) -> alpha less than 1 means
#      the joint can be representated as a bezier (0 <= alpha <= 1)
#    - Q (the size of the unit square centered at the joint pt ai)
#    - phi -> the angle between the line ai_bi and the unit y vector (0,1)
#            - this vector is on the side where x>0
#    - biMinus1 (note - biMinus1_ai is oriented so that it is along negative y-axis)
#    - l -> the length of the line ai_bi
#  This API contrains (takes as input alpha, Q, theta, biMinus1) and returns
#   the corresponding l
#  Joint returned is centered at (0,0) (ai) with biMinus_ai being aligned at negative y-axis
#  meaning biMinus1 is (0,y) where y < 0 and ai_bi is in region where x > 0
def genJoint(alpha, unitSquareSize, phi, biMinus1):
   # this API only supports families of joints where the lower RHS corner of the
   #  unit square centered at ai is within the span of biMinus1_ai and ai_bi
   # This means that the theta between positive y-axis and ai_bi is at most
   # 135 degrees
   phiInDegrees = math.degrees(phi)
   if phiInDegrees > 135 or phiInDegrees < 0.4:
      print("Unsupported phi (in degrees) - %s" % (phiInDegrees))
      return None
   # make sure phi is in radians
   phi = math.radians(phi)
   # solving the equations in givenJointReturnAlphaAndLiSimpleVer
   # but instead of solving for alpha we are solving for ai_bi
   # the equations in that API are:
   #  1) ciMinus1X + AUx = Q/2 <- Ux,y is the x,y-component of the vector biMinus1_bi
   #  2) ciMinus1Y + AUy = -Q/2
   #   Solve for Ux, Uy to solve for bi
   #  bi is determined by rotating the unit y-vector (0,1) by theta and multiply parameter L
   #  bi = L * [cosTheta -sinTheta * [0
   #            sinTheta  cosTheta]   1]
   # This means that U (biMinus1_bi) = (-L sin Theta, L cos Theta) - biMinus1
   #  Doing a bunch of algebra we get the equation
   # L = Q * biMinus1 / (2*C*sinTheta + Q*cosTheta)
   # where C = constant = (-Q/2) - (biMinus1 * (1-3*alpha/4))
   #
   #  NOTE: ciMinus1X = 0 since the joint is oriented such that biMinus1_ai is (0, -y)
   #        ciMinus1Y -> (ciMinus1Y - biMinus1Y) / biMinus1Y = gamma and gamma = 3 * alpha / 4
   const = (-unitSquareSize/2.0) - (biMinus1 * (1.0 - 3.0 * alpha / 4.0))
   L = unitSquareSize * biMinus1 / (2 * const * math.sin(phi) + unitSquareSize * math.cos(phi))

   bi = np.matmul(np.array([[math.cos(phi), -math.sin(phi)], \
                            [math.sin(phi), math.cos(phi)]]), \
                  np.array([0,1])) * L

   return bi

# This API reflects the set of input points to reflect
#   it also takes in a unit vect that indicates the direction of the line
#   (the unit vect is assumed to have its start pt at the origin 0,0)
#  in addition this API takes as input the shift that is applied to all of
#  the pts before doing the reflection
def reflectPtsAlongLine(listOfPts, unitVect, shift):
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

   reflectedPts = []
   for pt in listOfPts:
      # shift the pt first
      retPt = pt + shift
      retPt = np.matmul(reflectMat, retPt)
      # reverse the shift
      refPt -= shift
      reflectedPts.append(retPt)

   return reflectedPts

# API that takes as input a list of bezier curves and transforms them so that
# each bezier curve is connected to the one before it and that the
# ai_bi of the previous bezier curve and the biMinus1_ai of the next bezier curve
# form a straight line and that bi of the previous bezier curve === biMinus1
# of the next bezier curve
def orientBezierCurvesToFormContiguousCurve(bezierCurves):
   # NOTE: we start from idx = 1 (the second bezier curve in the list)
   # because the first bezier curve serves as the reference
   retCurves = [bezierCurves[0]]
   for i in range(1, len(bezierCurves)):
      # the angle of rotation is the angle direction going from biMinus1_ai
      # of the current bezier curve to the ai_bi of the previous bezier curve
      cosTheta = np.dot(bezierCurves[i].lineZ0_0[0].unitVect, \
                        bezierCurves[i-1].line0_Z3[0].unitVect)
      thetaDir = np.cross(bezierCurves[i].lineZ0_0[0].unitVect, \
                          bezierCurves[i-1].line0_Z3[0].unitVect)

      if thetaDir < 0:
         sinTheta = -math.sqrt(1-cosTheta**2)
      else:
         sinTheta = math.sqrt(1-cosTheta**2)

      rotMat = np.array([[cosTheta, -sinTheta], [sinTheta, cosTheta]])

      bezierCurves[i].rotateBezierCurve(rotMat)

      # after rotation calculate the shift of the bezier curve
      shift = bezierCurves[i-1].getLastPt() - bezierCurves[i].getFirstPt()
      bezierCurves[i].shiftBezierCurve(shift)

      # if the transformed bezier curve does NOT have the same convexity
      # as the previous curve, need to flip the current bezier curve along the
      # biMinus1_ai line
      if bezierCurves[i-1].getConvexityDir() != bezierCurves[i].getConvexityDir():
         # can use the line Z0_0 to reflect the curve against since we want to reflect
         # along the line made of control pts Z0_Z1 and we know Z0_Z1 lies along Z0_0
         # from our construction of bezier curves
         reflectionLine = bezierCurves[i].lineZ0_0.unitVect
         shift = -bezierCurves[i].getFirstPt(0)
         bezierCurves[i].reflectControlPtsAlongLine(reflectionLine, shift)

      retCurves.append(bezierCurves[i])

   return retCurves

# API to plot multiple bezier curves into 1 diagram
def plotMultipleBezierCurves(imgName, bezierCurves, drawJoint=True, delta_t=0.1, drawDeriv=False, unitDU=1):
   fig = plt.figure()
   ax = plt.subplot(111)
   for bCurve in bezierCurves:
      t = 0
      pts = [[] for i in range(bCurve.dataEntries)]
      ptsDerivs = [[] for i in range(bCurve.dataEntries)]
      while t < 1:
         pts[0].append(bCurve.getXYValOfCurve(t, 0))
         pts[1].append(bCurve.getXYValOfCurve(t, 1))
         ptsDerivs[0].append(bCurve.getDYDXgiven_t(t, 0))
         ptsDerivs[1].append(bCurve.getDYDXgiven_t(t, 1))
         t += delta_t
      origPtsX = [pt[0] for pt in pts[0]]
      origPtsY = [pt[1] for pt in pts[0]]

      rotatedPtsX = [pt[0] for pt in pts[1]]
      rotatedPtsY = [pt[1] for pt in pts[1]]

      tangentX = [[] for i in range(bCurve.dataEntries)]
      tangentY = [[] for i in range(bCurve.dataEntries)]
      # calculate the tangent lines if drawDeriv
      if drawDeriv:
         # for each pts deriv get its tangent by taking each pt as calculated in the
         #  pts list and adding 2 new pts to form a tangent line
         #  The 2 new pts are pt +/- du, where du is a vector where its
         #  x-component is 1 and y-component is therefore dy/dx
         #   however - if dy/dx is float(inf) because dx = 0, then du = (0,1)
         if drawRotated:
            rangeEnd = bCurve.dataEntries
         else:
            rangeEnd = bCurve.calcRotated
         for i in range(rangeEnd):
            for idx in range(len(ptsDerivs[i])):
               if ptsDerivs[i] == float("inf"):
                  DU = np.array([0,unitDU])
               else:
                  DU = np.array([unitDU, unitDU*ptsDerivs[i][idx]])
               ptMinusDU = pts[i][idx] - DU
               ptPlusDU = pts[i][idx] + DU
               tangentX[i].append([ptMinusDU[0], pts[i][idx][0], ptPlusDU[0]])
               tangentY[i].append([ptMinusDU[1], pts[i][idx][1], ptPlusDU[1]])

      ax.plot(origPtsX, origPtsY)
      # now plot the guides
      ax.plot([bCurve.lineZ0_0[0].termPt1[0], bCurve.lineZ0_0[0].termPt2[0],\
               bCurve.lineZ3_0[0].termPt1[0]], \
              [bCurve.lineZ0_0[0].termPt1[1], bCurve.lineZ0_0[0].termPt2[1],\
               bCurve.lineZ3_0[0].termPt1[1]], color='red')

      # if drawDeriv is TRUE need to draw tangents
      if drawDeriv:
         for i in range(len(tangentX[0])):
            ax.plot(tangentX[0][i], tangentY[0][i], color='green')

   plt.savefig(imgName)


# API to check whether a given list of lines is contiguous
#  return the groupings of lines that are contiguous
def genGroupOfContiguousLinesFromList(lines):
   # first need to create adjacency graph that contains both the line object and the
   # adjacent pt
   adjMap = {}
   for line in lines:
      termPt1 = tuple(line.termPt1)
      termPt2 = tuple(line.termPt2)
      if not adjMap.get(termPt1):
         adjMap[termPt1] = [(termPt2, line)]
      elif (termPt2, line) not in adjMap[termPt1]:
         adjMap[termPt1].append((termPt2, line))
      if not adjMap.get(termPt2):
         adjMap[termPt2] = [(termPt1, line)]
      elif (termPt1, line) not in adjMap[termPt2]:
         adjMap[termPt2].append(termPt1, line)

   # now that the adjacency map has been filled - traverse lines thru adjacency pts
   #  use BFS because we want to store lines that are contiguous
   processedPts = {}
   lineGroups = []
   for pt in adjMap:
      lineGroup = []
      if not processedPts.get(pt):
         queue = adjMap.get(pt)
         processedPts[pt] = True
         # traverse BFS thru the "graph" via the adjacency pts - store the line
         #  that make up the connection between the 2 pts
         while queue:
            adjPt = queue.pop(0)
            lineGroup.append(adjPt[1])
            if not processedPts.get(adjPt[0]):
               queue.extend(adjMap.get(adjPt[0]))
               processedPts[adjPt[0]] = True
         # store the lines in lineGroup into the lineGroups
         lineGroups.append(tuple(set(lineGroup)))

   return tuple(lineGroups)

# API to return the list of lines in order if line is singly
# contiguous (starting from the line with the start pt not touching anything
# and moving to the line that touches it) and that for each pt there is max 2 lines that
# share the same pt - count a simple cycle (where each pt only has 2 lines touching
# it) as a single contiguous list
def returnSinglyContiguousOrderedList(linesList):
   # first populate the map of pt to lines that the pt belongs to
   ptToLinesMap = {}
   startPt = None
   for line in linesList:
      termPt1 = tuple(line.termPt1)
      termPt2 = tuple(line.termPt2)
      if not ptToLinesMap.get(termPt1):
         ptToLinesMap[termPt1] = line
      elif line not in ptToLinesMap[termPt1]:
         ptToLinesMap[termPt1].append(line)
      if not ptToLinesMap.get(termPt2):
         ptToLinesMap[termPt2] = line
      elif line not in ptToLinesMap[termPt2]:
         ptToLinesMap[termPt2].append(line)

      if not startPt:
         startPt = termPt1

   # now loop thru the map to see if there are pts that belong to > 2 lines
   #  and look for entries that have < 2 lines
   startPts = []
   for pt, lines in ptToLinesMap.items():
      if len(lines) > 2:
         print("pt %s is shared by more than 2 lines - list of lines \
                is not singly contiguous" % (pt))
         return False, linesList
      elif len(lines) < 2:
         startPts.append(pt)

   # a singly contiguous line can only have either 2 start pts (a line seg) or
   # 0 (a simple cycle)
   if len(startPts) == 2:
      startPt = startPts[0]
   elif len(startPts) > 2:
      print("number of pts that only belong to 1 line exceeds 2 - this means \
             that the list of lines is not contiguous")
      return False, linesList

   # first starting from the start pt (either the pt with only 1 line (endpt)
   #  or since all of the pts belong to 2 lines (have simple cycle) the startPt
   #  is just the first entry of the map
   lineInserted = ptToLinesMap.get(startPt)[0]
   retLinesList = [lineInserted]
   linesAlreadyInserted = {lineInserted: True}

   currPt = startPt
   while True:
      if currPt == tuple(lineInserted.termPt1):
         currPt = tuple(lineToInserted.termPt2)
      elif currPt == tuple(lineToInserted.termPt2):
         currPt = tuple(lineToInserted.termPt1)
      linesToInsert = ptToLinesMap.get(currPt)
      # insert the new line if not inserted yet
      if len(linesToInsert) > 2:
         print("ERROR - more than 2 lines sharing the pt %s" % (currPt))
         return False, linesList

      if lineInserted == linesToInsert[0]:
         if len(linesToInsert) < 2:
            print("Finished inserting all of the lines - exit")
            break
         else:
            if linesAlreadyInserted.get(linesToInsert[1]):
               print("Simple cycle detected - all lines already filled")
               break
            retLinesList.append(linesToInsert[1])
            lineInserted = linesToInsert[1]
            linesAlreadyInserted.update({lineInserted: True})
      elif lineInserted == linesToInsert[1]:
         if linesAlreadyInserted.get(linesToInsert[0]):
            print("Simple cycle detected - all lines already filled")
            break
         retLinesList.append(linesToInsert[0])
         lineInserted = linesToInsert[0]
         linesAlreadyInserted.update({lineInserted: True})

   return True, retLinesList

# API to check if the lines in the line list in the order passed in is
# singly contiguous - ie. if the lines are already connected to each other in the order
# in the list and in the orientation that is in the list
def checkIfLinesSinglyContiguous(linesList):
   ret = True
   for idx in range(len(linesList)-1):
      if not np.array_equal(linesList[idx].termPt2, linesList[idx+1].termPt1):
         if np.linalg.norm(linesList[idx].termPt2 - linesList[idx+1].termPt1) > 0.1:
            print("lines %s with endpt %s is not contiguous with line %s with start pt %s" \
                  %(idx, linesList[idx].termPt2, idx+1, linesList[idx+1].termPt1))
            ret = False
         else:
             linesList[idx].setEndPt(linesList[idx+1].termPt1)

   return ret

# given list of singly contiguous lines calculate area of the polygon formed
# by the start / end pts of the lines
#  NOTE - for lines to be singly contiguous - the endpt of 1 line is equal to the
#         start pt of the line after it and that each pt is only shared by 2 lines max
#      - for the singly contiguous line the area of the polygon generated by this line
#         is the singly contiguous line itself AND is closed by the "fictional" edge
#         of the endpt of the last line to the start pt of the first line (the 2 open pts
#         of the singly contiguous line)
def calcAreaUnderSinglyContiguousLine(linesList):
   if not checkIfLinesSinglyContiguous(linesList):
      print("Failed to calculate area of lineList %s \
             list of lines is not singly contiguous" % (linesList))
      return None

   # now that the linesList is confirmed to be singly contiguous - get its points
   ptsList = [line.termPt1 for line in linesList]
   ptsList.append(linesList[-1].termPt2)
   xPts = np.array([pt[0] for pt in ptsList])
   yPts = np.array([pt[1] for pt in ptsList])
   area = 0.5*np.abs(np.dot(xPts,np.roll(yPts,1)) - np.dot(yPts,np.roll(xPts,1)))

   return area

# API to calculate the euclidean vector between a line and a pt
# by solving this system of eqns
#  - pt1 = termPt1 of line
#  - U = unit vect of line
#  - U^ = vect that is perpendicular to unit vect - note that there are 2 candidates
#    that are 180 of each other - whichever we pick does NOT matter because the B
#    term will wash
#  - C = pt on line so that line from C to pt is perpendicular to line
# pt1X + AUx = Cx
# pt1Y + AUy = Cy
# Cx + BU^x = ptX
# Cy + BU^y = ptY
# Return - the euclidean vector between the pt and the line
#        - the dist from the start pt of the line to the origin of the euclidean vector
#          (on the line)
def calcEuclideanProjBtwnLineAndPt(line, pt):
   # solving the variables in the order
   #  A B Cx Cy const terms
   perpVect = np.array([line.unitVect[1], -line.unitVect[0]])
   A1 = np.array([[line.unitVect[0], 0, -1, 0], \
                  [line.unitVect[1], 0, 0, -1], \
                  [0, perpVect[0], 1, 0], \
                  [0, perpVect[1], 0, 1]])

   B1 = np.array([-line.termPt1[0], -line.termPt1[1], pt[0], pt[1]])

   A, B, Cx, Cy = np.linalg.solve(A1, B1)

   return (B * perpVect, A)

# API to check if the orthogonal projection of the pt
# onto the line lies between the line segment passed in
def checkIfOrthoProjOfPtBtwnLine(pt, line):
   termPt1ToPtVect = np.array(pt) - line.termPt1
   dotProd1 = np.dot(termPt1ToPtVect, line.unitVect)
   if dotProd1 < 0:
      print("Dot product between vect from start pt to input pt and line is %s - \
             pt is not within line" % (dotProd1))
      return False

   termPt2ToPtVect = np.array(pt) - line.termPt2
   dotProd2 = np.dot(termPt2ToPtVect, line.unitVect)
   if dotProd2 > 0:
      print("Dot product between vect from end pt to input pt and line is %s - \
             pt is not within line" % (dotProd2))
      return False

   return True

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

      self.imgHeight = 0
      self.imgWidth = 0

      self.minX = self.maxX = self.minY = self.maxY = 0

   def getAllLinesAsList(self):
      return [line for idx, line in self.lineIdxToLineMap.items()]

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
                      "imgHeight" : self.imgHeight,
                      "imgWidth" : self.imgWidth
                   }

      with open(jsonName, 'w') as jsonFile:
         json.dump(jsonToDump, jsonFile)

   def insertLinesToIdxMap(self, lines):
      for line in lines:
         self.insertLineToIdxMap(line)

   def insertLineToIdxMap(self, line):
      retIdx = -1
      if line.lineLength > 0:
         termPtKey1 = (line.termPt1[0], line.termPt1[1], line.termPt2[0], line.termPt2[1])
         termPtKey2 = (line.termPt2[0], line.termPt2[1], line.termPt1[0], line.termPt1[1])

         if not self.lineTermPtsCacheMap.get(termPtKey1, None) and \
            not self.lineTermPtsCacheMap.get(termPtKey2, None):

            self.lineIdxToLineMap[line.hash] = line

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
            self.lineTermPtsCacheMap[termPtKey1] = line.hash

      return line.hash

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
                  line.displayLineInfo()
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
      contigSeg = contigSegCls()
      for idx, contour in self.lineContourToLineIdxs.items():
         print("contigSeg - looking through contour idx " + str(idx) + " with line idxs " + str(contour))
         for lineIdx in contour:
            print("contig seg - line idx is " + str(lineIdx))
            if not contigSeg.insertLineToContigSeg(lineIdx, self.lineIdxToLineMap[lineIdx]):
               contigSeg.finalizeContigSeg()
               retIdx = self.insertContigSegToIdxMap(contigSeg)
               if retIdx < 0:
                  print("contigSeg - failed to insert contigSeg")
               contigSeg = contigSegCls(lineIdx, self.lineIdxToLineMap[lineIdx])

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

   # the equation to calculate whether 2 lines intersect or not is:
   # A * UnitVect_1 + startPt_line1 = B * UnitVect_2 + startPt_line2
   # To rearrange the equation and solve A, B in this 2x2 eqn:
   #  A * UnitVect_1_X - B * UnitVect_2_X = startPt_line2_X - startPt_line1_X
   #  A * UnitVect_1_Y - B * UnitVect_2_Y = startPt_line2_Y - startPt_line1_Y
   def calcIntersectionOf2Lines(self, line1, line2):
      retPt = None

      A1Const = line1.getUnitVect()[0]
      B1Const = -line2.getUnitVect()[0]
      C1Const = line2.getStartPt()[0] - line1.getStartPt()[0]

      A2Const = line1.getUnitVect()[1]
      B2Const = -line2.getUnitVect()[1]
      C2Const = line2.getStartPt()[1] - line1.getStartPt()[1]

      a = np.array([[A1Const, B1Const],[A2Const, B2Const]])
      b = np.array([C1Const, C2Const])

      try:
         x = LA.solve(a,b)
         print("result is " + str(x))
         # A is the first element of x, B is the second element
         if A <= line1.getLength() and B <= line2.getLength():
            retPt = line1.getStartPt() + A * line1.getUnitVect()
         else:
            print("line1 and line2 do not intersect at any pt")
            line1.displayLineInfo()
            line2.displayLineInfo()
      except:
         print("ERROR TRYING TO SOLVE 2x2 matrix - unable to calculate intersection pt")
         print("a is " + str(a) + " and b is " + str(b))

      return retPt

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

   # given a contour of line idxs generate the contig segs for said contour
   def generateContigSegsFromContour(self, contour):
      alreadyInContigSeg = []
      retContigSegs = []
      for i in range(len(contour)):
         if i not in alreadyInContigSeg:
            alreadyInContigSeg.append(i)
            contigSegCandidate = contigSegCls(contour[i], self.lineIdxToLineMap[contour[i]])
            for j in range(i+1, len(contour)):
               if j not in alreadyInContigSeg:
                  if contigSegCandidate.insertLineToContigSeg(contour[j], self.lineIdxToLineMap[contour[j]]):
                     alreadyInContigSeg.append(j)
            contigSegCandidate.finalizeContigSeg()
            retContigSegs.append(contigSegCandidate)

      return retContigSegs

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

      for idxToDelete in contoursToDelete:
         del self.lineContourToLineIdxs[idxToDelete]

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

       try:
          cv.imwrite(outFileName, imgOutLine)
       except Exception as e:
          print("Failed to generate image %s with error %s" % (outFileName, e))

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
