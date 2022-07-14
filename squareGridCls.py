#!/usr/bin/python

import numpy as np
import math
import copy

from linesAPIs import *
from determineContigSegFromLines import *

class genRedundantContigSegCls:
   def __init__(self, startRefIdx, secIdxGroups):
      self.startRefIdx = startIdx
      self.endRefIdx = startIdx
      self.secSegFinalGroups = secIdxGroups

   def getRefSegAndSecSegTuple(self):
      return ((self.startRefIdx, self.endRefIdx), self.secSegFinalGroups)

   def checkIfRedundantSegPairsExtended(self, refIdx, secIdxGroups, lineSegIdxToContigSegMap):
      # check if the input refIdx belongs to the same contigSeg as startRefIdx
      retCode = False
      # if the input refIdx belongs to the same contigSeg as processing refSegs (self.startRefIdx)
      # and the input refIdx is contiguous to the processing refSegs
      if (lineSegIdxToContigSegMap[refIdx] == lineSegIdxToContigSegMap[self.startRefIdx]) and \
         ((refIdx - self.endRefIdx) == 1):
         # loop thru the secIdxGroups to check if the refSeg can be extended
         #   - need to check if there are any input secIdxGroups that can extend
         #     the existing secIdxGroups
         # SYNTAX WILL BE:
         #   X1 = start seg idx of secGroup that is iterated as part of input secIdxGroups
         #   X2 = end seg idx of secGroup that is iterated as part of input secIdxGroups
         #   X3 = start seg idx of secSegFinalGroup that is iterated as part of self.secSegFinalsGroups
         #   X4 = end seg idx of secSegFinalGroup that is iterated as part of self.secSegFinalsGroups
         #
         #  Will use overlap of squares condition X1-1 <= X4 && X2 >= X3-1
         #
         #  To cut short of iterations -> if X2 < X3-1 -> can stop since the secIdxGroups are in order
         #
         #  NOTE: the LHS condition needs to be X1 - 1 or X3 - 1 since we are looking for overlap
         newSecSegFinalGroups = []

         for secSegFinalGroup in self.secSegFinalGroups:
            secSegFinalGroupOverlap = False
            # check if secGroup belongs to any existing group
            for secGroup in secIdxGroups:
               X1 = secGroup[0]
               X2 = secGroup[1]
               X3 = secSegFinalGroup[0]
               X4 = secSegFinalGroup[1]
               if X2 < X3 - 1:
                  print("existing sec group start idx " + str(secGroup[0]) + " end idx " + str(secGroup[1]) + " already exceeds iterating sec group with start idx " + str(secIdxGroup[0]) + " and end idx " + str(secIdxGroup[1]))
                  break
               # check if the 2 sec tuples overlap
               if X1-1 <= X4 and \
                  X2 >= X3-1:
                  if X1 < X3:
                     XMin = X1
                  else:
                     XMin = X3
                  if X2 < X4:
                     XMax = X4
                  else:
                     XMax = X2

                  newSecSegFinalGroups.append((XMin, XMax))

         # sort the newSecSegFinalGroups if it isn't empty
         if newSecSegFinalGroups:
            newSecSegFinalGroups.sort()
            self.endRefIdx = refIdx
            self.secSegFinalGroups = newSecSegFinalGroups
            retCode = True

      return retCode


class squareCls:

   displayDelimiter = "SSSSSSSSSSSSSSSSSSSSS"

   def __init__(self, xMin, yMin, squareDim):
      self.squareDim = squareDim
      self.xMin = xMin
      self.xMax = self.xMin + self.squareDim
      self.yMin = yMin
      self.yMax = self.yMin + self.squareDim

      self.lineSegsInSquare = []

   def displaySquareData(self):
      print(squareCls.displayDelimiter)
      print("square dimension is " + str(self.squareDim))
      print("square XMIN is " + str(self.xMin))
      print("square XMAX is " + str(self.xMax))
      print("square YMIN is " + str(self.yMin))
      print("square YMAX is " + str(self.yMax))
      print("line segs in square is " + str(self.lineSegsInSquare))

   # define if a pt is in square
   #   by seeing if it is >= xMin / yMin
   #   and if it is < xMax / yMax
   # V2 - need to allow for pt to be ON xMax / yMax
   def checkIfPtIsInSquare(self, pt):
      return (pt[0] >= self.xMin and pt[0] <= self.xMax and pt[1] >= self.yMin and pt[1] <= self.yMax)

   def insertLineIdxIntoSquare(self, idx):
      if not idx in self.lineSegsInSquare:
         self.lineSegsInSquare.append(idx)

   # return a tuple of the start pt and end pt
   def getLineThatFitsIntoSquare(self, line):
      if not self.checkIfPtIsInSquare(line.termPt1):
         print("start pt " + str(line.termPt1) + " does not belong in this square - exit")
         self.displaySquareData()
         return None

      retLine = copy.deepcopy(line)

      # need to check if end pt is in square
      if not self.checkIfPtIsInSquare(line.termPt2):
         # need to check which pt is the intersection
         # between the line and the border of the box
         # to do this - will calculate the following 4 eqns
         #  line.termPt1[0] + A1*line.unitVect[0] = xMin (of square)
         #  line.termPt1[0] + A2*line.unitVect[0] = xMax (of square)
         #  line.termPt1[1] + A3*line.unitVect[1] = yMin (of square)
         #  line.termPt1[1] + A4*line.unitVect[1] = yMax (of square)
         # exclude the equations where unitVect[0] or unitVect[1] is 0
         AMin = None

         borderPts = ((self.xMin, self.xMax), (self.yMin, self.yMax))
         for i in range(2):
            if math.fabs(line.unitVect[i]) > 0:
               for borderEntry in borderPts[i]:
                  ACandidate = (borderEntry - line.termPt1[i]) / line.unitVect[i]
                  if ACandidate > 0 and \
                     (not AMin or ACandidate < AMin):
                     AMin = ACandidate

         if not AMin:
            print("Unable to find valid AMin - exit")
            return None

         print("Calculating AMin to be " + str(AMin))
         retLine.setEndPt(line.termPt1 + AMin * line.unitVect)

      return retLine

#######################
### DESIGN NOTES ##
##
## 1) treat pt as belonging in Grid IF:
##    - pt is >= startPt of SQUARE AND < endPt of SQUARE ie. [startPt, endPt)
## 2) HOWEVER, condition above is broken IFF it is a start pt of a line ON a boundary
##     AND the line lies in the adjacent cell
##     eg. if pt is on XMin BUT the line lies on the cell to its left
##
#######################
class gridCls:

   displayDelimiter = "GGGGGGGGGGGGGGGGGGGGG"

   xPosVect = np.array([1,0])
   yPosVect = np.array([0,1])

   def __init__(self, xMin, xMax, yMin, yMax, squareDim):

      self.linesInsertedIntoGrid = 0
      self.linesProcessed = 0

      self.maxContigSegIdx = 0
      self.maxLineSegIdx = 0

      # have a map of lineIdx to the grids they occupy
      self.lineSegIdxToGridsOccupied = {}

      self.contigSegIdxToContigSeg = {}

      self.lineSegIdxToLineSeg = {}

      self.lineSegIdxToContigSegIdxs = {}

      self.holeIdxToHole = {}

      #  compliment of the hole
      #  - this is the portion of the sec segs (from secSegStart to secSegEnd)
      #  - NOTE: the secSegStart may be cropped IF the refSeg start portion is completely
      #    overlapped by the secStartSeg (and for the end portion as well)
      #  thus, the compliment of the hole is ultimately of the form
      #   (startSecSegIdx, endSecSegIdx, modified lineCls obj for startSecSeg, modified lineCls obj for endSecSeg)
      # and the map is of the form {hole IDX : compliment of hole - tuple as defined above}
      self.holeIdxToCompliment = {}
      self.maxHoleSegIdx = 0

      # this is map of hole idx to the idx of the line that the hole came from - ie.
      #  the idx of the line that was cropped or removed entirely, resulting in the hole
      self.holeIdxToOrigLineIdx = {}

      # this is map of line idx and the hole idxs that come from the line seg with idx
      #  NOTE: a line (from line idx) may have multiple holes due to different segments
      #  being cropped
      self.lineIdxToHoleIdxs = {}

      # this map contains as key the ref line seg
      # and as val the list of all line segs that are determined to be redundant
      # with this seg
      self.lineSegToParallelLineSegs = {}
      # this list is contains the sorted keys of the above map that will be used
      #  as reference idxs when crawling for parallel segs
      self.inOrderRefIdxs = []

      # this map contains the exact opposite as above in the same format
      #  key as ref line seg and as val the list of all line segs that are
      #  determined to be NOT REDUNDANT with this seg - this prevents
      #  multiple checks of the SAME pairs that do NOT satisfy condition
      self.lineSegToNONParallelLineSegs = {}

      # this map contains the pairing of the ref contiguous segs
      # with its corresponding sec contiguous segs
      #
      #  contiguous segs are defined as segs that have consecutive idxs AND belong to the same contigSeg
      #
      # NOTE: this map only contains the longest ref contiguous segs ie. with largest span (startIdx, endIdx)
      #       and its corresponding sec contiguous segs
      #   has form {(refSegStartIdx1, refSegEndIdx1) : [(secSegStartIdx1, secSegEndIdx1), ..., (secSegStartIdxN, secSegEndIdxN)]}
      self.maxRefContigSegToRedundantSecContigSegs = {}

      # this map contains the pairing of contig seg to the line idxs within the contig seg to delete
      self.contigSegIdxToLinesToDel = {}

      # this map contains the ref seg idx to lineCls objects which contain the cropped or completely deleted line
      #   that is of idx ref seg idx in the form of
      #  {refIdx : [lineClsObj1, ..., lineClsObjN]}
      self.refSegIdxToCroppedLineSegs = {}

      self.bufferFactor = 2
      self.squareDim = squareDim

      self.gridXMin = xMin - self.bufferFactor * self.squareDim
      self.gridXMax = xMax + self.bufferFactor * self.squareDim
      self.gridYMin = yMin - self.bufferFactor * self.squareDim
      self.gridYMax = yMax + self.bufferFactor * self.squareDim

      # store the number of squares in a tuple
      # self.numOfXSquares = int((self.gridXMax - self.gridXMin)/self.squareDim) + 1
      # self.numOfYSquares = int((self.gridYMax - self.gridYMin)/self.squareDim) + 1
      self.numOfSquares = (int((self.gridXMax - self.gridXMin)/self.squareDim) + 1, int((self.gridYMax - self.gridYMin)/self.squareDim) + 1)

      # store the min x and val values of each square at its idx
    #  self.squareXVals = [self.gridXMin + i*self.squareDim for i in self.numOfXSquares]
    #  self.squareYVals = [self.gridYMin + i*self.squareDim for j in self.numOfYSquares]

      self.squareVals = ([self.gridXMin + i*self.squareDim for i in range(self.numOfSquares[0])], [self.gridYMin + j*self.squareDim for j in range(self.numOfSquares[1])])

      self.Grid = [[squareCls(self.gridXMin + i*self.squareDim, self.gridYMin + j*self.squareDim, self.squareDim) for j in range(self.numOfSquares[1])] for i in range(self.numOfSquares[0])]

   def insertLineIntoMaps(self, line):
      # insert the seg into individSegIdxToIndividSeg
      lineIdx = self.maxLineSegIdx
      self.lineSegIdxToLineSeg[lineIdx] = line
      self.maxLineSegIdx += 1
      return lineIdx

   def insertHoleIntoMaps(self, hole):
      holeIdx = self.maxHoleSegIdx
      self.holeIdxToHole[holeIdx] = hole
      self.maxHoleSegIdx += 1
      return holeIdx

   def insertLineSegIntoContigSegMaps(self, line, lineIdx, contigSegIdx):

      if self.lineSegIdxToContigSegIdxs.get(lineIdx, None):
         print("line idx " + str(lineIdx) + " is NOT unique to contig seg + " + str(contigSegIdx) + " - error")
      else:
         self.lineSegIdxToContigSegIdxs[lineIdx] = contigSegIdx


   def displayGridData(self):
      print(gridCls.displayDelimiter)
      print("square dimensions - " + str(self.squareDim))
      print("grid X min - " + str(self.gridXMin))
      print("grid X max - " + str(self.gridXMax))
      print("grid Y min - " + str(self.gridYMin))
      print("grid Y max - " + str(self.gridYMax))
      print("Number of squares along X - " + str(self.numOfSquares[0]))
      print("Number of squares along Y - " + str(self.numOfSquares[1]))
      print("XMin of squares - " + str(self.squareVals[0]))
      print("YMin of squares - " + str(self.squareVals[1]))
      print(gridCls.displayDelimiter)

   def displayGridSquareData(self):
      for i in range(self.numOfSquares[0]):
         for j in range(self.numOfSquares[1]):
            print("Square with Xind " + str(i) + " and Yind " + str(j) + " info: ")
            self.Grid[i][j].displaySquareData()

   def displayMapNice(self, inputMap):
      for key, vals in inputMap.items():
         print(str(key) + " : " + str(vals))

   def insertContigSegToMap(self, contigSeg):
      self.contigSegIdxToContigSeg[self.maxContigSegIdx] = contigSeg
      self.maxContigSegIdx += 1

   def getSquareXYIdxsThatLineStartsIn(self, line):
      # use idx 0, 1 to determine x, y instead of having the same logic twice

      line.calcLineMetadata()

      startPt = line.termPt1
      retList = [0,0]

      # first look for the square that the start pt of the line is in
      for i in range(2):
         # check x or y square
         startIdx = 0
         endIdx = self.numOfSquares[i]-1
         iterIdx = int((endIdx - startIdx)/2)

         prevIterIdx = iterIdx

         while not ( (startPt[i] >= self.squareVals[i][iterIdx]) and (startPt[i] < (self.squareVals[i][iterIdx]+self.squareDim)) ):

            if startPt[i] >= self.squareVals[i][iterIdx]+self.squareDim:
               print("start pt with idx " + str(i) + " : " + str(startPt[i]) + " is greater than max (" + str(i) + ") " + str(self.squareVals[i][iterIdx]+self.squareDim) + " of square idx " + str(iterIdx) + " - shift startIdx to iterIdx")
               startIdx = iterIdx
            elif startPt[i] < self.squareVals[i][iterIdx]:
               print("start pt with idx " + str(i) + " : " + str(startPt[i]) + " is less than min (" + str(i) + ") " + str(self.squareVals[i][iterIdx]) + " of square idx " + str(iterIdx) + " - shift endIdx to iterIdx")
               endIdx = iterIdx

            iterIdx = int((endIdx - startIdx)/2) + startIdx
            print("iterIdx is " + str(iterIdx) + " prevIdx is " + str(prevIterIdx))
            if prevIterIdx == iterIdx:
               print("coord " + str(i) + " iteration is stuck at value " + str(iterIdx) + " with val " + str(self.squareVals[i][iterIdx]) + " - break out of while loop")
               return None

            prevIterIdx = iterIdx

         retList[i] = iterIdx

      # need special handling for lines with start pts that are on boundary of the squares
      squareCandidate = self.Grid[retList[0]][retList[1]]

      if (startPt[0] == squareCandidate.xMin) and \
         (np.dot(line.unitVect, self.xPosVect) < 0):
         retList[0] -= 1

      if (startPt[0] == squareCandidate.xMax) and \
         (np.dot(line.unitVect, self.xPosVect) > 0):
         retList[0] += 1

      if (startPt[1] == squareCandidate.yMin) and \
         (np.dot(line.unitVect, self.yPosVect) < 0) :
         retList[1] -= 1

      if (startPt[1] == squareCandidate.yMax) and \
         (np.dot(line.unitVect, self.yPosVect) > 0):
         retList[1] += 1


      print("line with info ")
      line.displayLineInfo()
      print(" belongs in square " + str(retList) + " with dimensions ")
      self.Grid[retList[0]][retList[1]].displaySquareData()

      return retList

   def insertLineIdxIntoGridSquare(self, xIdx, yIdx, retLineIdx):
      print("inserting line idx " + str() + " into grid with square xIdx " + str() + " and yIdx " + str())
      self.Grid[xIdx][yIdx].insertLineIdxIntoSquare(retLineIdx)
      self.linesInsertedIntoGrid += 1

      if self.lineSegIdxToGridsOccupied.get(retLineIdx, None):
         (self.lineSegIdxToGridsOccupied[retLineIdx]).append((xIdx, yIdx))
      else:
         grids = [(xIdx, yIdx)]
         self.lineSegIdxToGridsOccupied[retLineIdx] = grids

   # API to insert list of lines into grid coordinates - this API
   # is to insert lines into the Grid and process the lines before
   # these lines have become contigsSegs
   def insertLinesIntoGrid(self, linesMap):

      squareXYIdxs = []

      for lineIdx, line in linesMap.items():
         # check which square the line belongs in
         squareXYIdxs = self.getSquareXYIdxsThatLineStartsIn(line)
         if not squareXYIdxs:
            print("failed to find the square that " + str(line.termPt1) + " belongs to")
            return

         # insert the line into lineSegMap
         retLineIdx = self.insertLineIntoMaps(line)

         #given the square X/Y idxs - now set the line starting from that square X/Y idx
         retLine = self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].getLineThatFitsIntoSquare(line)
         if retLine:
            print("inserting line idx " + str(retLineIdx) + " with start pt " + str(retLine.termPt1) + " and end pt " + str(retLine.termPt2) + " into square with Xidx " + str(squareXYIdxs[0]) + " and Yidx " + str(squareXYIdxs[1]))
            print("orig line start pt " + str(line.termPt1) + " and end pt " + str(line.termPt2))
            self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].displaySquareData()

            self.insertLineIdxIntoGridSquare(squareXYIdxs[0], squareXYIdxs[1], retLineIdx)

         # if the end point of the return line is NOT the same as the original line it means
         #  that the line exceeds the dimensions of the square and that the line has been cut
         #  to the dimensions of the square - continue crawling the grid and fitting the line
         #  until all segments have been fit into their corresponding squares in the grid
         while not np.array_equal(line.termPt2, retLine.termPt2):
            # create line that is the remaining portion of the line
            remainLine = lineCls()
            remainLine.setStartPt(retLine.termPt2)
            remainLine.setEndPt(line.termPt2)
            # note that this must lie on the border (xMin, xMax, yMin, yMax)
            #  if point is on xMin - means the x idx for the next square is curr x idx - 1
            #  if point is on xMax - means the x idx for the next square is curr x idx + 1
            #  if point is on yMin - means the y idx for the next square is curr y idx - 1
            #  if point is on yMax - means the y idx for the next square is curr y idx + 1
            xDelta = yDelta = 0
            if (retLine.termPt2[0] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].xMin) and \
               (np.dot(line.unitVect, self.xPosVect) < 0):
               xDelta = -1
            if (retLine.termPt2[0] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].xMax) and \
               (np.dot(line.unitVect, self.xPosVect) > 0):
               xDelta = 1
            if (retLine.termPt2[1] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].yMin) and \
               (np.dot(line.unitVect, self.yPosVect) < 0):
               yDelta = -1
            if (retLine.termPt2[1] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].yMax) and \
               (np.dot(line.unitVect, self.yPosVect) > 0):
               yDelta = 1

            squareXYIdxs[0] += xDelta
            squareXYIdxs[1] += yDelta

            retLine = self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].getLineThatFitsIntoSquare(remainLine)
            if retLine:
               print("inserting line idx " + str(retLineIdx) + " with start pt " + str(retLine.termPt1) + " and end pt " + str(retLine.termPt2) + " into square with Xidx " + str(squareXYIdxs[0]) + " and Yidx " + str(squareXYIdxs[1]))
               print("orig line start pt " + str(line.termPt1) + " and end pt " + str(line.termPt2))
               self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].displaySquareData()

               self.insertLineIdxIntoGridSquare(squareXYIdxs[0], squareXYIdxs[1], retLineIdx)

         retCode = True

      return retCode

   # the input contig seg has not yet been split
   #  into the grid coordinates
   #
   # NOTE : THIS IS THE EXTERNAL API THAT IS CALLED
   #  BY CODE OUTSIDE OF THIS CLASS TO INSERT CONTIGSEG INTO GRID
   # ie. THIS IS THE PUBLIC MEMBER FUNCTION OF THIS CLASS
   #
   # ALSO NOTE: the line segs in each contig seg is indexed sequentially, meaning
   #  that the line seg idxs in each contig seg should be sequential and be contiguous (ie. differ by 1)
   def insertContigSegIntoGrid(self, contigSeg):

      mapContigSeg = contigsSegsCls()
      mapContigSegIdx = self.maxContigSegIdx

      squareXYIdxs = []
      for i in range(len(contigSeg.lineIdxs)):
         line = contigSeg.lines[i]

         self.linesProcessed += 1

         # check which square the line belongs in
         squareXYIdxs = self.getSquareXYIdxsThatLineStartsIn(line)
         if not squareXYIdxs:
            print("failed to find the square that " + str(line.termPt1) + " belongs to")
            return

         # insert the line into lineSegMap
         retLineIdx = self.insertLineIntoMaps(line)
         contigSeg.lineIdxs[i] = retLineIdx

         #given the square X/Y idxs - now set the line starting from that square X/Y idx
         retLine = self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].getLineThatFitsIntoSquare(line)
         if retLine:
            print("inserting line idx " + str(retLineIdx) + " with start pt " + str(retLine.termPt1) + " and end pt " + str(retLine.termPt2) + " into square with Xidx " + str(squareXYIdxs[0]) + " and Yidx " + str(squareXYIdxs[1]))
            print("orig line start pt " + str(line.termPt1) + " and end pt " + str(line.termPt2))
            self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].displaySquareData()

            self.insertLineIdxIntoGridSquare(squareXYIdxs[0], squareXYIdxs[1], retLineIdx)

         # insert line params into contigSeg maps
         self.insertLineSegIntoContigSegMaps(line, retLineIdx, mapContigSegIdx)

         # if the end point of the return line is NOT the same as the original line it means
         #  that the line exceeds the dimensions of the square and that the line has been cut
         #  to the dimensions of the square - continue crawling the grid and fitting the line
         #  until all segments have been fit into their corresponding squares in the grid
         while not np.array_equal(line.termPt2, retLine.termPt2):
            # create line that is the remaining portion of the line
            remainLine = lineCls()
            remainLine.setStartPt(retLine.termPt2)
            remainLine.setEndPt(line.termPt2)
            # note that this must lie on the border (xMin, xMax, yMin, yMax)
            #  if point is on xMin - means the x idx for the next square is curr x idx - 1
            #  if point is on xMax - means the x idx for the next square is curr x idx + 1
            #  if point is on yMin - means the y idx for the next square is curr y idx - 1
            #  if point is on yMax - means the y idx for the next square is curr y idx + 1
            xDelta = yDelta = 0
            if (retLine.termPt2[0] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].xMin) and \
               (np.dot(line.unitVect, self.xPosVect) < 0):
               xDelta = -1
            if (retLine.termPt2[0] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].xMax) and \
               (np.dot(line.unitVect, self.xPosVect) > 0):
               xDelta = 1
            if (retLine.termPt2[1] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].yMin) and \
               (np.dot(line.unitVect, self.yPosVect) < 0):
               yDelta = -1
            if (retLine.termPt2[1] == self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].yMax) and \
               (np.dot(line.unitVect, self.yPosVect) > 0):
               yDelta = 1

            squareXYIdxs[0] += xDelta
            squareXYIdxs[1] += yDelta

            retLine = self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].getLineThatFitsIntoSquare(remainLine)
            if retLine:
               print("inserting line idx " + str(retLineIdx) + " with start pt " + str(retLine.termPt1) + " and end pt " + str(retLine.termPt2) + " into square with Xidx " + str(squareXYIdxs[0]) + " and Yidx " + str(squareXYIdxs[1]))
               print("orig line start pt " + str(line.termPt1) + " and end pt " + str(line.termPt2))
               self.Grid[squareXYIdxs[0]][squareXYIdxs[1]].displaySquareData()

               self.insertLineIdxIntoGridSquare(squareXYIdxs[0], squareXYIdxs[1], retLineIdx)

         retCode = True

         #insertLineToContigSeg input (lineIdx, line, checkDotProd, insertIdxOnly)
         mapContigSeg.insertLineToContigSeg(retLineIdx, line, False, True)

         # check that contigSeg is contiguous
         if not mapContigSeg.checkContigSegIsContiguous(self.lineSegIdxToLineSeg):
            print("mapContigSeg is NOT contiguous - check logs above for source of error")
            retCode = False
         else:
            # insert mapContigSeg into contigSegIdxToContigSeg
            mapContigSeg.calcContigSegMetadata(self.lineSegIdxToLineSeg)
            mapContigSeg.calcContigSegMetadata()
            self.contigSegIdxToContigSeg[mapContigSegIdx] = mapContigSeg
            self.maxContigSegIdx += 1

      return retCode

   # API to check if the line idxs in the input list belong to the same contigSeg
   #   The lineSegIdxToContigSegIdxs map contains {lineSegIdx : contigSegIdx}
   #   Each lineSegIdx corresponds to a contigSegIdx even if 2 lineSegs are the same (same start pt and end pt)
   #   They will have different lineSegIdx
   def checkIfLineIdxsBelongToSameContigSeg(self, listOfLineSegs):
      firstContigSegIdx = self.lineSegIdxToContigSegIdxs[listOfLineSegs[0]]
      for elemIdx in range(1, len(listOfLineSegs)):
         if self.lineSegIdxToContigSegIdxs[listOfLineSegs[elemIdx]] != firstContigSegIdx:
            return False

      return True

   # API to convert the map from format of {ref seg : [redundantSeg1, redundantSeg2, ..., redundantSegN]}
   #
   #   into {(refSeg, contigSegRefSegBelongTo, lenOfContigSeg) : [(redundSegStart1, redundSegEnd1, contigSegLinesBelongTo, lenOfContigSeg),
   #                                                              (redundSegStart2, ..., redundSegEnd2, contigSegLinesBelongTo, lenOfContigSeg)]}
   #
   #   where redundSegStart1 is the redundant seg with the lowest index belonging to a contigSeg and redundSegEnd1 is
   #   is the highest index belonging to the same contiguous segment (ie. the initial map contains redundSegStart1,
   #   redundSegStart1+1, redundSegStart+2, .., redundSegEnd1)
   #
   # reason for this structure is to move to multiprocessing as a future speedup
   def convertLineSegToLineSegsTupleMap(self, origMap, startIdx, endIdx):
      retMap = {}
      origMapKeyList = list(origMap.keys())
      # loop thru the origMap to convert format to above where store only start and end idx of same contig seg
      for i in range(startIdx, endIdx+1):
         lineSegVals = origMap.get(origMapKeyList[i], None)
         if lineSegVals:
            newLineSegGroups = []
            newLineSegGroup = []
            # first - sort since the line segs are initially generated in order for each contig seg
            #  so line segs belonging in contig seg would have contiguous
            lineSegVals.sort()
            startSeg = endSeg = None
            for valsIdx in range(len(lineSegVals)-1):
               if not startSeg:
                  startSeg = lineSegVals[valsIdx]
                  endSeg = startSeg
                  newLineSegGroup = [startSeg, endSeg, self.lineSegIdxToContigSegIdxs[lineSegVals[valsIdx]], self.contigSegIdxToContigSeg[self.lineSegIdxToContigSegIdxs[lineSegVals[valsIdx]]].length]

               # terminate the tuple if:
               #  1) the lineSegIdx at the current position is NOT contiguous (+1) the lineSegIdx at the immediate prev position
               #  2) check if the lines are contiguous using checkIf2LinesAreContiguous API
               #  3) the lineSegIdx at the current position DOES NOT belong to the same contigSeg as the lineSegIdx at the immediate prev position
               if ((lineSegVals[valsIdx+1] - lineSegVals[valsIdx]) != 1) or \
                  not checkIf2LinesAreContiguous(self.lineSegIdxToLineSeg[lineSegVals[valsIdx]], self.lineSegIdxToLineSeg[lineSegVals[valsIdx+1]]) or \
                  not self.checkIfLineIdxsBelongToSameContigSeg([lineSegVals[valsIdx+1], lineSegVals[valsIdx]]):
                  newLineSegGroup[1] = lineSegVals[valsIdx]
                  newLineSegGroup[2] = self.lineSegIdxToContigSegIdxs[lineSegVals[valsIdx]]
                  newLineSegGroup[3] = self.contigSegIdxToContigSeg[self.lineSegIdxToContigSegIdxs[lineSegVals[valsIdx]]].length
                  newLineSegGroups.append((newLineSegGroup[0], newLineSegGroup[1], newLineSegGroup[2], newLineSegGroup[3]))
                  newLineSegGroup = []
                  startSeg = endSeg = None
               else:
                  newLineSegGroup[1] = lineSegVals[valsIdx+1]


            # check if the last element in lineSegVals is absorbed as newLineSegsGroup as an endpt - if it is, this means
            # that newLineSegGroup is filled - need to push it into newLineSegGroups
            if newLineSegGroup:
               newLineSegGroups.append((newLineSegGroup[0], newLineSegGroup[1], newLineSegGroup[2], newLineSegGroup[3]))
            # if the newLineSegGroup is empty this means that the last seg does NOT belong in the same group as the second to last line seg
            #  need to explicitly create a tuple for the last seg
            else:
               newLineSegGroups.append((lineSegVals[-1], lineSegVals[-1], self.lineSegIdxToContigSegIdxs[lineSegVals[-1]], self.contigSegIdxToContigSeg[self.lineSegIdxToContigSegIdxs[lineSegVals[-1]]].length))

            retMap[(origMapKeyList[i], self.lineSegIdxToContigSegIdxs[origMapKeyList[i]], self.contigSegIdxToContigSeg[self.lineSegIdxToContigSegIdxs[origMapKeyList[i]]].length)] = newLineSegGroups

      return retMap

   # API to loop through square grid to resolve the "parallel/redundant" line segs
   #   loop thru each square to determine which segs are parallel with other segs
   #   Put the parallel into lineSegToParallelLineSegs map and the NOT parallels into lineSegToNONParallelLineSegs
   #
   # input - raw 2d grid of squares containing line indices of lines within each square on grid
   # output - generate map in form of {refSeg : [(redundStart1, ..., redundEnd1), (redundStart2, ..., redundEnd2)]}
   def crawlThruGridToGetRedundantSegs(self):
      foundRedundant = False
      # only need to look at neighbors in :
      #   (0, 1), (1, 0), (1, 1) direction since this API moves from top LHS corner of grid
      #   thus, no need for negative coord since sweep looks at +ve direction
      neighborGrids = [(0,1), (1,0), (1,1), (-1, 1)]
      # loop thru y-coord
      for j in range(self.numOfSquares[1]):
         # loop thru x-coord
         for i in range(self.numOfSquares[0]):
            for neighbor in neighborGrids:
               xneighbor = i + neighbor[0]
               yneighbor = j + neighbor[1]

               # if either the x or the y neighbor is out of bounds of the grid
               # continue to the next potential neighbor
               if xneighbor < 0 or xneighbor > (self.numOfSquares[0]-1) or \
                  yneighbor < 0 or yneighbor > (self.numOfSquares[1]-1):
                  continue

               # compare the line segs in the grid and in the adjacent grid to see if they are redundant
               for lineIdx1 in self.Grid[i][j].lineSegsInSquare:
                  for lineIdx2 in self.Grid[xneighbor][yneighbor].lineSegsInSquare:

                     # if the 2 line segs have already been determined to be parallel / redundant or NOT
                     #  or if the 2 lines have the same idx (ie. are the same line)
                     if ( lineIdx1 == lineIdx2 ) or \
                        ( self.lineSegToParallelLineSegs.get(lineIdx1, None) and \
                          lineIdx2 in self.lineSegToParallelLineSegs.get(lineIdx1, None) ) or \
                        ( self.lineSegToNONParallelLineSegs.get(lineIdx1, None) and \
                          lineIdx2 in self.lineSegToNONParallelLineSegs.get(lineIdx1, None) ):
                        continue

                     # check if the 2 lines are parallel / redundant
                     if checkIf2LinesRedundant(self.lineSegIdxToLineSeg.get(lineIdx1, None), self.lineSegIdxToLineSeg.get(lineIdx2, None), 5.0):
                        # if the 2 lines are redundant - store them into the
                        #  lineSegToParallelLineSegs map
                        foundRedundant = True
                        if not self.lineSegToParallelLineSegs.get(lineIdx1, None):
                           self.lineSegToParallelLineSegs[lineIdx1] = [lineIdx2]
                        else:
                           self.lineSegToParallelLineSegs[lineIdx1].append(lineIdx2)

                        if not self.lineSegToParallelLineSegs.get(lineIdx2, None):
                           self.lineSegToParallelLineSegs[lineIdx2] = [lineIdx1]
                        else:
                           self.lineSegToParallelLineSegs[lineIdx2].append(lineIdx1)

                     else:
                        # if the 2 lines are NOT redundant - store them into the
                        #  lineSegToNONParallelLineSegs map
                        if not self.lineSegToNONParallelLineSegs.get(lineIdx1, None):
                           self.lineSegToNONParallelLineSegs[lineIdx1] = [lineIdx2]
                        else:
                           self.lineSegToNONParallelLineSegs[lineIdx1].append(lineIdx2)

                        if not self.lineSegToNONParallelLineSegs.get(lineIdx2, None):
                           self.lineSegToNONParallelLineSegs[lineIdx2] = [lineIdx1]
                        else:
                           self.lineSegToNONParallelLineSegs[lineIdx2].append(lineIdx1)

      return foundRedundant


   # API to process the lineSegToParallelLineSegs map so that
   #  1) the keys, which serves as the IN ORDER REFERENCE idxs that we will crawl this map
   #     to determine parallel (redundant) segs, are sorted in ascending order
   #  2) sort the idxs in the corresponding list (value) of parallel / redundant segs
   #      in ascending order
   def sortRefIdxs(self):
      self.inOrderRefIdx = list(self.lineSegToParallelLineSegs.keys())
      self.inOrderRefIdx.sort()

      for refIdx in self.inOrderRefIdx:
         if self.lineSegToParallelLineSegs.get(refIdx, None):
            secSegs = self.lineSegToParallelLineSegs.get(refIdx)
            secSegs.sort()
            self.lineSegToParallelLineSegs[refIdx] = secSegs

   # API to crawl the lineSegToParallelLineSegs map to get line segs that should be deleted
   #  1) crawl the ref lines (ie. the key idx of lineSegToParallelLineSegs map)
   #  2) check if the corresponding sec redundant line segs recorded are part of contig seg that is
   #     longer than the contig seg that the ref line idx belongs to
   #  3) if condition (2) above is satisfied - add the ref line idx to ref line that must be deleted along with its contig seg
   #  4) while crawling, group the line segs that must be deleted that are: 1) contiguous 2) belong to the same contig seg
   #  5) Store the line segs to delete in the form of map like {"contigSegIdx" : [(lineIdxToDelStart1, lineIdxToDelEnd1), (lineIdxToDelStart2, lineIdxToDelEnd2)]}
   def crawlLineSegToParallelLineSegsMapForRedundSegsToDelete(self):
      # crawl thru map
      spanToDel = []

      self.sortRefIdxs()

      for refIdxEntry in self.inOrderRefIdx:
         refIdxContigSegLen = self.contigSegIdxToContigSeg[self.lineSegIdxToContigSegIdxs[refIdxEntry]].length

         # loop thru the sec line segs for this ref line to see if the ref line should be deleted
         for secEntry in self.lineSegToParallelLineSegs[refIdxEntry]:
            secIdxContigSegLen = self.contigSegIdxToContigSeg[self.lineSegIdxToContigSegIdxs[secEntry]].length

            # if length of contig seg found in sec is greater than ref contig seg
            if secIdxContigSegLen > refIdxContigSegLen:
               if not spanToDel:
                  spanToDel = [refIdxEntry, refIdxEntry]
               else:
                  # check if this refIdx is contiguous with the previous refEndIdx (spanToDel[1]) and if the 2 line
                  # idxs belong to the same contigSeg - if they do
                  if ( (refIdxEntry == spanToDel[1]+1) and \
                       self.checkIfLineIdxsBelongToSameContigSeg([refIdxEntry, spanToDel[1]]) ):
                     spanToDel[1] = refIdxEntry
                  # if they don't
                  else:
                     # need to insert the existing spanToDel into the map contigSegIdxToLinesToDel
                     if not self.contigSegIdxToLinesToDel.get(self.lineSegIdxToContigSegIdxs[spanToDel[0]], None):
                        self.contigSegIdxToLinesToDel[self.lineSegIdxToContigSegIdxs[spanToDel[0]]] = [(spanToDel[0], spanToDel[1])]
                     else:
                        self.contigSegIdxToLinesToDel[self.lineSegIdxToContigSegIdxs[spanToDel[0]]].append((spanToDel[0], spanToDel[1]))

                     spanToDel[0] = refIdxEntry
                     spanToDel[1] = refIdxEntry
               # once found a sec contig seg longer than ref contig seg - no need to look farther
               break


   # API to crawl the map in the form of
   #
   #   {refLineSeg : [[secLineSegStartContiguous1, secLineSegEndContiguous1, contigTheseSegsBelongTo, lengthOfContigSeg], .., [secLineSegstartContiguousX, secLineSegEndContiguousX, contigTheseSegsBelongTo, lengthOfContigSeg]]}
   #
   # to create groups of contiguous line segs that are parallel redundant
   #
   # 1) Crawl thru (in order of increasing idx) the refLineSegs - go thru the secLineSegs groups -> as crawl thru the next refContigSeg
   #      if the refLineIdx is adjacent to the prev - check the secLineSegs groups to see if the group at the current refContigSeg
   #      is contiguous with secLineSegs group (if 2 groups overlap OR 1 group follows another)
   #      - POSTULATE: ONLY keep redundant pair with longest ref seg since all line segs will be ref seg at one point
   #                    - this will take care of any seg portions that have more than one redundant portion!
   #        DATA storing - map of {priContigSeg : [secContigSeg1, secContigSeg2, .., secContigSegN]}
   #
   # OUTPUT - map of self.maxRefContigSegToRedundantSecContigSegs
   def crawlThruSegMapToGenerateRedundantContigSegs(self):
      refKeys = list(self.lineSegToParallelLineSegs.keys())
      refKeys.sort()
      genRedund = genRedundantContigSegCls(refKeys[0], self.lineSegToParallelLineSegs[refKeys[0]])

      for i in range(1, len(refKeys)):
         # check if this ref and the ref in genRedund are part of the same contig seg - if not terminate the existing genRedund
         #  and create new genRedund item
         if not genRedund.checkIfRedundantSegPairsExtended(refKeys[i], self.lineSegToParallelLineSegs[refKeys[i]], self.lineSegIdxToContigSegIdxs):
            refSegSecSegTuple = genRedund.getRefSegAndSecSegTuple()
            self.maxRefContigSegToRedundantSecContigSegs[refSegSecSegTuple[0]] = refSegSecSegTuple[1]
            genRedund = genRedundantContigSegCls(refKeys[i], self.lineSegToParallelLineSegs[refKeys[i]])

   # API to crawl thru map in the form of
   # {(refSeg, contigSegRefSegBelongTo, lenOfContigSeg) : [(redundSegStart1, redundSegEnd1, contigSegLinesBelongTo, lenOfContigSeg), (redundSegStart2, ..., redundSegEnd2, contigSegLinesBelongTo, lenOfContigSeg)]}
   #
   #   where redundSegStart1 is the redundant seg with the lowest index belonging to a contigSeg and redundSegEnd1 is
   #   is the highest index belonging to the same contiguous segment (ie. the initial map contains redundSegStart1,
   #   redundSegStart1+1, redundSegStart+2, .., redundSegEnd1)
   #
   # 0) convert the map of self.lineSegToParallelLineSegs from {refSeg : [secSeg1, ..., secSegN]} to
   #     {(refSeg, contigSegIdx, contigSegLen) : [(secSeg1, secSegN, contigSegIdx, contigSegLen), ..]}
   #
   # 1) sort the ref seg tuples in order of longest contigSeg that the refSeg belongs to - that way the longest segs are resolved first, providing
   #    anchors for shorter segs
   #
   # 2) for each ref seg - sort the sec segs by len of contig segs len - longest segs 1st
   #
   # 3) For each grouping of (secSegStart, secSegEnd, contigSeg, contigSegLen) - orient secSegStart and secSegEnd such that
   #    the termPt2 of secSegStart == termPt1 of secSegStart + 1 AND termPt2 of secSegEnd - 1 == termPt1 of secSegEnd
   #
   # 4) Check the orientation of the sec seg group by taking termPt1 of secSegStart and termPt2 of secSegEnd and getting the unit vector
   #    of that direction - check if the direction of refSeg is the same (ie. if UnitVectSec dot UnitVectRef >= 0) - IF NOT, flip refSeg
   #
   # 5) Now that secSeg and refSeg are properly oriented - check projection secSeg onto refSeg to see if refSeg is completely overlap OR
   #    if refSeg has segments that stick out at the start or at the end - use projection equation
   #      termPt1SecStart + A * UnitSecStartPerp = termPt1RefStart + B * UnitRef <-- start seg of secSegs
   #      termPt2SecEnd   + A * UnitSecEndPerp   = termPt2RefEnd   + B * -UnitRef <-- end seg of secSegs
   #  IN THIS CASE - THE REF SEG WILL BE THE "REFERENCE" SEG THAT TAKES AS INPUT THE SEC START / SEC END SINCE B is the value we wish to use
   #  to determine - if B < 0 ---> this means that there are outstanding portions of refSeg that must be stored
   #
   #  NOTE: if FUTURE seg uses PROCESSED seg (seg that was previously crawled ref seg) as anchor - use ORIG ref seg as anchor since anchor seg
   #        redundancy is handled when anchor was ref seg
   #
   #        While looping thru the secSegs for the one ref seg - use the resulting cropped refSeg for each secSegGroup as long as the secSegGroup contigSeg
   #        is longer than the refSeg contig seg
   #
   def crawlRefSegsToRmvRedundancy(self):

      refSegTupleToRedundSegTupleMap = self.convertLineSegToLineSegsTupleMap(self.lineSegToParallelLineSegs, 0, len(self.lineSegToParallelLineSegs)-1)

      # sort the ref segs in order of contig seg len (entry with idx = 2 in ref tuple since refSegTuple - (refSeg, contigSegIdx, contigSegLen)) in order of longest to shortest
      orderedRefSegs = sorted(refSegTupleToRedundSegTupleMap, key=lambda ref: ref[2], reverse=True)

      for refEntry in orderedRefSegs:
         refSegContigSegLen = refEntry[2]
         secSegs = refSegTupleToRedundSegTupleMap.get(refEntry)
         # sort the sec segs in order of contig seg len (entry with idx = 3 in sec seg tuple - secSegStart, secSegEnd, contigSegIdx, contigSegLen) in order of longest to shortest
         sortedSecSegs = sorted(secSegs, key=lambda sec: sec[3], reverse=True)
         refSeg = self.lineSegIdxToLineSeg.get(refEntry[0])
         refResultSegs = [refSeg]

         for secEntry in sortedSecSegs:
            # check if len of contig seg of secSeg is greater than refSeg contig seg - if not can break since this is sorted by secSeg contigSeg len
            secSegContigSegLen = secEntry[3]
            if secSegContigSegLen < refSegContigSegLen:
               break

            newRefSegs = []
            for refResultSeg in refResultSegs:
               # orient the contig segs
               startSecSeg = self.lineSegIdxToLineSeg.get(secEntry[0])
               endSecSeg = self.lineSegIdxToLineSeg.get(secEntry[1])

               if secEntry[1] != secEntry[0]:
                  startSecSegAdj = self.lineSegIdxToLineSeg.get(secEntry[0]+1)
                  endSecSegAdj = self.lineSegIdxToLineSeg.get(secEntry[1]-1)
               else:
                  startSecSegAdj = endSecSegAdj = None

               # orient the start sec seg such that its end pt is a shared pt with startSecSeg+1
               if (startSecSegAdj and endSecSegAdj):
                  if np.array_equal(startSecSeg.termPt1, startSecSegAdj.termPt1) or \
                     np.array_equal(startSecSeg.termPt1, startSecSegAdj.termPt2):
                     startSecSeg.flipLine()
                  elif not (np.array_equal(startSecSeg.termPt2, startSecSegAdj.termPt1) or \
                            np.array_equal(startSecSeg.termPt2, startSecSegAdj.termPt2)):
                     print("ERROR - startSecSeg does NOT share any pt with startSecSeg+1 - startSecSeg NOT CONTIGUOUS")

                  # orient the end sec seg such that its start pt is a shared pt with endSecSeg-1
                  if np.array_equal(endSecSeg.termPt2, endSecSegAdj.termPt1) or \
                     np.array_equal(endSecSeg.termPt2, endSecSegAdj.termPt2):
                     endSecSeg.flipLine()
                  elif not (np.array_equal(endSecSeg.termPt1, endSecSegAdj.termPt1) or \
                            np.array_equal(endSecSeg.termPt1, endSecSegAdj.termPt2)):
                     print("ERROR - endSecSeg does NOT share any py with endSecSeg-1 - endSecSeg NOT CONTIGUOUS")

               # orient the ref seg so that it is in same direction of secSeg group
               secSegVect = endSecSeg.termPt2 - startSecSeg.termPt1
               secSegUnitVect = (secSegVect) / LA.norm(secSegVect)
               if np.dot(secSegUnitVect, refResultSeg.unitVect) < 0:
                  refResultSeg.flipLine()

               # check the start section of refResultSeg to see if it is completely overlapped by startSecSeg
               startPtProjParams = refResultSeg.lineProjInputToSelfWithInPerpSelfUnit(startSecSeg)
               endPtProjParams = refResultSeg.lineProjInputToSelfWithInPerpSelfUnit(endSecSeg, True)

               if not refResultSeg.lineLength:
                  refResultSeg.calcLineMetadata()

               # need to define the "hole" corresponding to the segment of refResultSeg that is removed
               # because of the redundant sec seg
               refSegHole = lineCls()

               # also need to define the compliment of the hole
               #  - this is the portion of the sec segs (from secSegStart to secSegEnd)
               #  - NOTE: the secSegStart may be cropped IF the refSeg start portion is completely
               #    overlapped by the secStartSeg (and for the end portion as well)
               #  thus, the compliment of the hole is ultimately of the form
               #   (startSecSegIdx, endSecSegIdx, modified lineCls obj for startSecSeg, modified lineCls obj for endSecSeg)
               holeSecSegCompliment = [None, None, None, None]

               # if the ref seg start pt is not completely overlapped within
               # start sec seg
               #
               # calculate the portion of the ref seg that is not covered
               #  by the start sec seg
               if startPtProjParams[0] > 0:
                  newStartLine = lineCls()
                  newStartLine.setStartPt(refResultSeg.termPt1)
                  newStartLine.setEndPt(refResultSeg.termPt1 + startPtProjParams[0] * refResultSeg.unitVect)
                  if not newStartLine.checkIfLineIsPoint():
                     newRefSegs.append(newStartLine)
                  # the hole is the end pt of the ref segment that is not redundant
                  # with the starting sec seg
                  refSegHole.setStartPt(refResultSeg.termPt1 + startPtProjParams[0] * refResultSeg.unitVect)
               else:
                  refSegHole.setStartPt(refResultSeg.termPt1)
                  # if the ref seg is completely overlapped within the
                  # start sec seg - look for the start sec seg that the
                  # the ref seg start pt is "within"
                  # calculate the portion of the that sec seg
                  # that overlaps the ref seg calculating the projection of the ref seg
                  # termPt1 to the starting sec seg
                  for i in range(secEntry[0], secEntry[1]+1):
                     secStartCandidate = self.lineSegIdxToLineSeg.get(i)
                     refProjToSecStartParams = secStartCandidate.lineProjInputToSelfWithSelfPerpSelfUnit(refResultSeg)
                     if(math.fabs(refProjToSecStartParams[0]) < secStartCandidate.lineLength):
                        newSecSegStart = lineCls()
                        newSecSegStart.setStartPt(secStartCandidate.termPt1 + refProjToSecStartParams[0] * secStartCandidate.unitVect)
                        newSecSegStart.setEndPt(secStartCandidate.termPt2)
                        holeSecSegCompliment[0] = i
                        holeSecSegCompliment[2] = newSecSegStart
                        break

               if endPtProjParams[0] > 0:
                  newEndLine = lineCls()
                  newEndLine.setStartPt(refResultSeg.termPt2 + endPtProjParams[0] * -refResultSeg.unitVect)
                  newEndLine.setEndPt(refResultSeg.termPt2)
                  if not newEndLine.checkIfLineIsPoint():
                     newRefSegs.append(newEndLine)
                  refSegHole.setEndPt(refResultSeg.termPt2 + endPtProjParams[0] * -refResultSeg.unitVect)
               else:
                  refSegHole.setEndPt(refResultSeg.termPt2)
                  for i in range(secEntry[1], secEntry[0]-1, -1):
                     secEndCandidate = self.lineSegIdxToLineSeg.get(i)
                     refProjToSecEndParams = secEndCandidate.lineProjInputToSelfWithSelfPerpSelfUnit(refResultSeg, True)
                     if(math.fabs(refProjToSecEndParams[0]) < secEndCandidate.lineLength):
                        newSecSegEnd = lineCls()
                        newSecSegEnd.setStartPt(secEndCandidate.termPt1)
                        newSecSegEnd.setEndPt(secEndCandidate.termPt2 + refProjToSecEndParams[0] * -secEndCandidate.unitVect)
                        holeSecSegCompliment[1] = i
                        holeSecSegCompliment[3] = newSecSegEnd

               # add hole into map
               holeIdx = self.insertHoleIntoMaps(refSegHole)
               self.holeIdxToCompliment[holeIdx] = tuple(holeSecSegCompliment)

               # add entry to map of hole idx to the line that the hole came from
               self.holeIdxToOrigLineIdx[holeIdx] = refEntry[0]
               # add entry to map of line idx (reg seg idx) to the hole idx that it generated
               if not self.lineIdxToHoleIdxs.get(refEntry[0]):
                  self.lineIdxToHoleIdxs[refEntry[0]] = [holeIdx]
               else:
                  self.lineIdxToHoleIdxs[refEntry[0]].append(holeIdx)
            #NOTE - if the ref line is completely overlapped by the sec portions - it will be COMPLETELY
            #       removed because the newRefSegs list will not contain the original refLine
            refResultSegs = newRefSegs

         self.refSegIdxToCroppedLineSegs[refEntry[0]] = refResultSegs

   # This API is called after the redundant segs are removed by doing the following
   #
   #  1) Put the new finalized lines into the new map - remove the lines that have been
   #     trimmed as portions (or all) of the seg has been removed - if this original seg
   #     is not completely gone but has portions of it left, add it to the line map with increasing
   #     idx after max line idx
   #
   #  2) Crawl these line segs above into contig segs that satisfy the "curve" criteria
   #       The criteria are:
   #          - the dir of the potential line seg and the dir of the last line seg
   #            is different by angle of <= 45 degrees (pi / 4 radians) or the dot product
   #            of the 2 unit vects are <= (1/root(2))
   #          - the direction of the angle between 2 curves changes from the rest of the
   #            segments that make up the curve - eg. all segs that have change in direction
   #            must be consistent in direction of change (clockwise or counterclockwise)
   def alignUniqueLinesIntoCurves(self):

      # copy the self.lineSegIdxToLineSeg map
      newLinesMap = copy.deepcopy(self.lineSegIdxToLineSeg)
      newLinesIdx = self.maxLineIdx
      # loop thru the self.refSegIdxToCroppedLineSegs to remove the original seg
      # and add the cropped segs as new segs
      for refSeg in self.refSegIdxToCroppedLineSegs:
         # delete the ref entry
         del newLinesMap[refSeg]
         # enter the cropped segs of the original seg into the map
         for croppedSegs in self.refSegIdxToCroppedLineSegs[refSeg]:
            newLinesMap[newLinesIdx] = croppedSegs
            newLinesIdx += 1

      # generate contig seg curves with this new map
      ptsGraph = determineContigSegFromLines()
      ptsGraph.assignLineMapAndCreateAdjMap(newLinesMap)
      ptsGraph.genContigSegsUsingGraphAdj()

      # orient the contig segs to make sure that the end pt of a line is
      # equal to the start pt of the next line
      for contigSegEntry in ptsGraph.contigsSegsIdxToContigsSegs:
         ptsGraph.contigsSegsIdxToContigsSegs[contigSegEntry].finalizeContigSeg()

      # generate map of open contig seg and its cumulative length
      #
      #  NOTE: open contig seg is one with >= 1 termPts not connected to other contig segs
      #        length of their connected neighbors = length of contig seg + len of its
      #        immediate neighbors and all contig segs connected to those neighbors
      #        and so on and so forth
      openContigSegIdxToCumulativeLen = ptsGraph.genMapOfOpenContigSegIdxToCumulativeLen()

      ptToHolesMap = {}

      # generate map of point to any holes that have that pt as termPt
      for holeEntry in self.holeIdxToHole:
         hole = self.holeIdxToHole[holeEntry]
         if not ptToHolesMap.get(hole.termPt1):
            ptToHolesMap[hole.termPt1] = [holeEntry]
         else:
            if holeEntry not in ptToHolesMap[hole.termPt1]:
               ptToHolesMap[hole.termPt1].append(holeEntry)

         if not ptToHolesMap.get(hole.termPt2):
            ptToHolesMap[hole.termPt2] = [holeEntry]
         else:
            if holeEntry not in ptToHolesMap[hole.termPt2]:
               ptToHolesMap[hole.termPt2].append(holeEntry)

      # crawl the map of openContigSegIdxToCumulativeLen
      for contigIdx in openContigSegIdxToCumulativeLen:
         # get the term pt of the contig seg that is a hole
