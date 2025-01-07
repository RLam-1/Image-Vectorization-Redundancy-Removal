#!/usr/bin/python

import sys
import os
import json
import argparse
import multiprocessing
import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy import linalg as LA

from linesAPIs import *
from squareGridCls import *
from determineContigSegFromLines import *

testOutType = "KCOS"
contigSeg = "CONTIGSEG"
testOutPathPart2 = ".png"

WHITE = (255, 255, 255)

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPPINK = (255, 0, 255)
TURQOISE = (0, 255, 255)
MUDPURP = (129, 97, 131)
ORANGE = (226, 83, 3)

contigSegColorArray = [BLACK, RED, GREEN, BLUE, YELLOW, PURPPINK, TURQOISE, MUDPURP, ORANGE]


def autoCanny(image, sigma=0.33):
   v = np.median(image)

   lower = int(max(0, (1.0 - sigma) * v))
   upper = int(min(255, (1.0 + sigma * v)))
   edges = cv.Canny(image, lower, upper)

   return edges

# assume list is in form (cx, cy, contourIdx)
def mergesortContourCentroidByCoord(centroidIdxList, left, right, coord):

   if coord > 1:
      return

   if left < right:
      mid = int((right - left) / 2 + left)
      print("left is " + str(left) + " mid is " + str(mid) + " right is " + str(right))
      mergesortContourCentroidByCoord(centroidIdxList, left, mid, coord)
      mergesortContourCentroidByCoord(centroidIdxList, mid+1, right, coord)

      # sort between left and right subarrays
      L = centroidIdxList[left: mid+1]
      R = centroidIdxList[mid+1: right+1]

      i = j = 0
      idx = left

      while(i < len(L) and j < len(R)):
         if L[i][coord] < R[j][coord]:
            centroidIdxList[idx] = L[i]
            i += 1
         else:
            centroidIdxList[idx] = R[j]
            j += 1
         idx += 1

      while(i < len(L)):
         centroidIdxList[idx] = L[i]
         idx += 1
         i += 1
      while(j < len(R)):
         centroidIdxList[idx] = R[j]
         idx += 1
         j += 1

# secondary sort coord -> this is sorting the secondary coord
#  for coordinates with the same primary coord
#  x-cord = 0, y-cord = 1
def subsequentMergesortContourCentroidByCoord(centroidIdxList, secCord):
   # if secCord = 1, priCord = 0
   # if secCord = 0, priCord = 1
   priCord = ~(secCord & secCord)
   leftIdx = rightIdx = 0
   segToSort = [centroidIdxList[0]]
   for i in range(1, len(centroidIdxList)):
      if (segToSort[0][priCord] != centroidIdxList[i][priCord]) or \
         (i == (len(centroidIdxList)-1)):
         if len(segToSort) == 1:
            segToSort[0] = centroidIdxList[i]
         else:
            # sort the segToSort based on secondary coordinate
            mergesortContourCentroidByCoord(segToSort, 0, len(segToSort), secCord)
            # double check that number of elements in segToSort is the same as
            # span covered by leftIdx and rightIdx
            if len(segToSort) != (rightIdx - leftIdx + 1):
               print("NOT THE SAME LENGTH - segToSort: " + str(len(segToSort)) + " idxDiff: " + str(rightIdx - leftIdx + 1))
            centroidIdxList[leftIdx:rightIdx+1] = segToSort
            segToSort = [centroidIdxList[i]]

         leftIdx = i
         rightIdx = i
      else:
         segToSort.append(centroidIdxList[i])
         rightIdx = i

maxPixelDist = 10
minDotProdOfMajAxis = 0.9
# API to filter out duplicate contours
#  Criteria for duplicate contours are:
#  1) Centroid is within maxPixelDist (pixels) of each other
#  2) The dot product of the major axis of the contour is greater than minDotProdOfMajAxis
#   If both criterion match, take the contour with the greater number of points
def filterDupContoursOut(contourList, contourIdxToEignValVectMap, contourIdxToContourMap):
   uniqueContours = []

   idx = 1
   refIdx = 0

   while idx < len(contourList):
      # get dist between 2 centroids
      print("contourList at refIdx is ")
      print(str(contourList[refIdx]))
      refCentroid = np.array([contourList[refIdx][0], contourList[refIdx][1]])
      idxCentroid = np.array([contourList[idx][0], contourList[idx][1]])
      centroidDiff = LA.norm(np.subtract(refCentroid, idxCentroid))

      if centroidDiff > maxPixelDist:
         uniqueContours.append(contourList[refIdx])
         refIdx = idx
      else:
         refContourIdx = contourList[refIdx][2]
         idxContourIdx = contourList[idx][2]
         # get dot product between 2 contours
         refEignValVect = contourIdxToEignValVectMap.get(refContourIdx, None)
         idxEignValVect = contourIdxToEignValVectMap.get(idxContourIdx, None)
         if not refEignValVect:
            print("unable to find reference contour " + str(refContourIdx) + " in contour to eigenvector map")
            refIdx = idx
            idx += 1
            continue

         if not idxEignValVect:
            print("unable to find idx contour " + str(idxContourIdx) + " in contour to eigenvector map")
            idx += 1
            continue

         refMajorAxis = refEignValVect[1][0]
         idxMajorAxis = idxEignValVect[1][0]
         dotVal = np.dot(refMajorAxis, idxMajorAxis)

         if (dotVal > minDotProdOfMajAxis) and \
            (len(contourIdxToContourMap[idxContourIdx]) > len(contourIdxToContourMap[refContourIdx])):
            refIdx = idx

         if idx == len(contourList) - 1:
            uniqueContours.append(contourList[refIdx])

      idx += 1

   return uniqueContours

def drawListOfContigSegsToImg(listOfContigSegs, imgHeight, imgWidth, imgName, arrowed, lineIdxMap=None):
   line_thickness = 1
   imgOutContigSegsFull = np.ones([int(imgHeight), int(imgWidth)], dtype=np.uint8) * 255
   for contigSeg in listOfContigSegs:
      for idx in range(len(contigSeg.lineIdxs)):
         line = lines = None
         if lineIdxMap:
            if type(lineIdxMap[contigSeg.lineIdxs[idx]]) == list:
               lines = lineIdxMap[contigSeg.lineIdxs[idx]]
            else:
               line = lineIdxMap[contigSeg.lineIdxs[idx]]
         else:
            if len(contigSeg.lineIdxs) != len(contigSeg.lines):
               print("Drawing contig seg - number of line idxs " + str(len(contigSeg.lineIdxs)) + " does not equal to number of lines " + str(len(contigSeg.lines)))
            line = contigSeg.lines[idx]

         if not lines and line:
            lines = [line]
         print("lines to draw are " + str(lines))
         for line in lines:
            if arrowed:
               cv.arrowedLine(imgOutContigSegsFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContigSegsFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

   cv.imwrite(imgName, imgOutContigSegsFull)

def drawListOfColorContigSegsToImg(listOfContigSegs, imgHeight, imgWidth, imgName, arrowed, lineIdxToLineMap=None):
   line_thickness = 1
   #imgOutContigSegsFull = np.ones([imgHeight, imgWidth], dtype=np.uint8) * 255
   imgOutContigSegsFull = np.zeros((imgHeight, imgWidth, 3), np.uint8)

   # convert image background to white = (255, 255, 255)
   imgOutContigSegsFull[:] = WHITE

   if len(listOfContigSegs) > len(contigSegColorArray):
      print("number of contig segs " + str(len(listOfContigSegs)) + " is greater than number of colors " + str(len(contigSegColorArray)) + " - exit without drawing")

   else:
      for countIdx in range(len(listOfContigSegs)):
         # if line idx is provided - use the line indices from the contig seg to fetch the line from the map and draw
         #   otherwise use the line object directly stored in contigSeg
         if lineIdxToLineMap:
            iterMax = len(listOfContigSegs[countIdx].lineIdxs)
         else:
            iterMax = len(listOfContigSegs[countIdx].lines)

         for lineIter in range(iterMax):
            if lineIdxToLineMap:
               line = lineIdxToLineMap.get(listOfContigSegs[countIdx].lineIdxs[lineIter])
            else:
               line = listOfCOntigSegs[countIdx].lines[lineIter]

            # since opencv uses bgr instead of rgb - need to reverse
            color = tuple(reversed(contigSegColorArray[countIdx]))

            if arrowed:
               cv.arrowedLine(imgOutContigSegsFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), color, line_thickness, tipLength=0.5)
            else:
               cv.line(imgOutContigSegsFull, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), color, line_thickness)

      cv.imwrite(imgName, imgOutContigSegsFull)

def generateUniqueContours(contours):
   contourIdxToContourMap = {}
   contourIdxToCentroidMap = {}
   contourIdxToEignValVectMap = {}
   centroidAndIdxList = []

   for idx, contour in enumerate(contours):
      print("contour idx is " + str(idx))
      #print(str(contour))
      for contourPt in contour:
         print("contour point is " + str(contourPt))
         #print("contour x is " + str(contourPt[0]) + ": contour y is " + str(contourPt[1]))
         cvPt = contourPt[0]
         print("contour x is " + str(cvPt[0]) + ": contour y is " + str(cvPt[1]))

      print("length of contour is " + str(len(contour)))

      print("moments of contour is ")
      M = cv.moments(contour)
      print(str(M))
    #  if idx < len(contours) - 1:
    #     sim = cv.matchShapes(contours[idx], contours[idx+1], cv.CONTOURS_MATCH_I1, 0.0)
    #     print("Similarity between contour " + str(idx) + " and " + str(idx+1) + " is ")
    #     print(sim)

      contourIdxToContourMap[idx] = contour

      if M['m00'] != 0:
         mu20prime = M['mu20'] / M['m00']
         mu02prime = M['mu02'] / M['m00']
         mu11prime = M['mu11'] / M['m00']
         covarMat = np.array([[mu20prime, mu11prime], [mu11prime, mu02prime]])
         w, v = LA.eig(covarMat)
         print("eigenvalues are ")
         print(str(w))
         print("eigenvectors are ")
         print(str(v))

         contourIdxToEignValVectMap[idx] = (w, v)

         # calculate centroid
         cx = M['m10'] / M['m00']
         cy = M['m01'] / M['m00']
         print("centroid x: " + str(cx) + " y: " + str(cy))
         contourIdxToCentroidMap[idx] = (cx, cy)
         centroidAndIdxList.append((cx, cy, idx))

   for hierEntry in hierarchy:
      print("hierarchy entry is ")
      print(str(hierEntry))

   print("length of centroid list BEFORE pri coord sort is " + str(len(centroidAndIdxList)))

   xcoord = 0
   ycoord = 1
   mergesortContourCentroidByCoord(centroidAndIdxList, 0, len(centroidAndIdxList)-1, xcoord)
   print("length of centroid list AFTER FIRST pri coord sort is " + str(len(centroidAndIdxList)))

   subsequentMergesortContourCentroidByCoord(centroidAndIdxList, ycoord)
   print("length of centroid list AFTER SECOND sec coord sort is " + str(len(centroidAndIdxList)))

   print("order of contour centroids are")
   for coord in centroidAndIdxList:
      print(str(coord))

   uniqueContours = filterDupContoursOut(centroidAndIdxList, contourIdxToEignValVectMap, contourIdxToContourMap)

   imgOut = np.ones([height, width], dtype=np.uint8)*255
   uniqueContourList = []
   for idx, uniqueCont in enumerate(uniqueContours):
    #  imgOut = np.ones([height, width], dtype=np.uint8)*255
      print("unique contour " + str(idx) + " is ")
      print(str(uniqueCont))
      uniqueContourList.append(contourIdxToContourMap[uniqueCont[2]])
      print("unique contour value is ")
      print(str(contourIdxToContourMap[uniqueCont[2]]))
      cv.drawContours(imgOut, [contourIdxToContourMap[uniqueCont[2]]], 0, (0, 255, 0), 1)
      testOutImgName = testOutPathPart1 + testOutType + "contours" + testOutPathPart2
      cv.imwrite(testOutImgName, imgOut)

   return uniqueContours

def generateLineMapClsContours(imgName, processContour=False, drawContours=True):
   img = cv.imread(imgName)
   imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

   height, width = imgGray.shape[:2]

  # blurImg = cv.GaussianBlur(imgGray, (3,3), 0)
   wide = cv.Canny(imgGray, 10, 200)
   tight = cv.Canny(imgGray, 225, 250)
   auto = autoCanny(imgGray)

 #  contours, hierarchy = cv.findContours(auto, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   contours, hierarchy = cv.findContours(auto, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS )

   imgOutLine = np.ones([height, width], dtype=np.uint8)*255
   lineMap = lineMapCls()

   lineMap.imgHeight = height
   lineMap.imgWidth = width

   lineMap.detectLinesFromCVContours(contours)
#   lineMap.detectLinesFromCVContours(uniqueContourList)

   if drawContours:
      lineMap.drawLinesToImg(imgOutLine, False)
      cv.imwrite(os.path.splitext(imgName)[0] + testOutType + "PREALIGN" + testOutPathPart2, imgOutLine)

   if processContour:
      #lineMap.alignLinesInContour()
      lineMap.processContourLineSegs()

   # dump firstLineMap maps to JSON
   jsonName = os.path.splitext(imgName)[0] + ".json"
   lineMap.dumpInfoToJSON(jsonName)

   return lineMap

def readLineMapClsContoursFromJSON(jsonName):
   lineMap = lineMapCls()
   lineMap.readInfoFromJSON(jsonName)

   return lineMap

### given list of lines
### get the XMin, XMax, YMin, and YMax in the form of tuple
### (XMin, XMax, YMin, YMax)
def getXYMinMaxFromListOfLines(listOfLines):
   XMin = XMax = YMin = YMax = 0
   for line in listOfLines:
      if line.termPt1[0] < XMin:
         XMin = line.termPt1[0]
      elif line.termPt1[0] > XMax:
         XMax = line.termPt1[0]
      if line.termPt2[0] < XMin:
         XMin = line.termPt2[0]
      elif line.termPt2[0] > XMax:
         XMax = line.termPt2[0]

      if line.termPt1[1] < YMin:
         YMin = line.termPt1[1]
      elif line.termPt1[1] > YMax:
         YMax = line.termPt1[1]
      if line.termPt2[1] < YMin:
         YMin = line.termPt2[1]
      elif line.termPt2[1] > YMax:
         YMax = line.termPt2[1]

   return (XMin, XMax, YMin, YMax)

### given list of contours of the form contigSegCls
### get the XMin, XMax, YMin, and YMax in the form of tuple
def getXYMinMaxFromListOfContigsSegs(inputContigsSegsList):
   XMin = XMax = YMin = YMax = 0
   lineList = []
   for contigsSegs in inputContigsSegsList:
      lineList.extend(contigsSegs.lines)
   return getXYMinMaxFromListOfLines(lineList)

### given map of line idx to lineCls objects
### get the XMin, XMax, YMin, YMax in the form of tuple
def getXYMinMaxFromLineMap(lineMap):
   XMin = XMax = YMin = YMax = 0
   lineList = [line for idx, line in lineMap.items()]
   return getXYMinMaxFromListOfLines(lineList)

### given lineCls objects - generate contig seg
#   using determineContigSegFromLines class
#   that uses pts as nodes in a graph and then traversing graph
####
##  Performance
#
#real	0m2.043s
#user	0m1.981s
#sys	0m0.053s
# takes 3 potential keyword inputs
#  if lineMap is passed in - pass in as kargs.lineMap
#  elif lineList is passed in - pass in as kargs.lineList
#  also pass in kargs.drawContigSegs - if TRUE draw the contig segs
def genContigSegsUsingNodeGraph(**kargs):
   if kargs.get("lineMap"):
      linesList = kargs.get("lineMap").getAllLinesAsList()
   else:
      linesList = kargs.get("lineList",[])

   ptsGraph = determineContigSegFromLines()
   ptsGraph.populateAdjacencyMapFromLines(linesList)

   ptsGraph.genContigSegsUsingGraphAdj()

   if kargs.get("drawContigSegs"):
      drawListOfContigSegsToImg(ptsGraph.getAllContigSegAsList(), \
                                kargs.get("lineMap").imgHeight, \
                                kargs.get("lineMap").imgWidth, \
                                testOutPathPart1 + "CONTIGS_SEGS_FROM_GRAPH" + testOutPathPart2, False)

      imgOutLine = np.ones([kargs.get("lineMap").imgHeight, kargs.get("lineMap").imgWidth], dtype=np.uint8)*255
      ptsGraph.drawLinesToImg(imgOutLine, False)
      cv.imwrite(testOutPathPart1 + "LINES_FROM_GRAPH" + testOutPathPart2, imgOutLine)

   return ptsGraph

# return map of linesIdx to lineObjects
# this API does the following
#  1) insert line objects from linemap into a square grid object
#  2) crawl thru the square grid object to find lines that potentially cross Each
#     other
#  3) handle lines that cross each other by splitting those 2 lines into 4
#      lines and turning the pt of intersection into the end pt for each of those
#      4 new lines
def useGridOverlayToSeparateIntersectLines(inputLineMap):
   # first get square grid dimensions xmin, xmax, ymin, and ymax from the linesMap
   squareGridMinMax = getXYMinMaxFromLineMap(inputLineMap.lineIdxToLineMap)
   # create grid of squares given XMin, XMax, YMin, YMax

   # for now set squareDim = 10.0
   squareDim = 10.0

   # generate grid of squares as gridOverlay
   gridOverlay = gridCls(squareGridMinMax[0], squareGridMinMax[1], \
                         squareGridMinMax[2], squareGridMinMax[3], \
                         squareDim, False, lineMap=inputLineMap)

   gridOverlay.displayGridData()

   gridOverlay.insertLineMapIntoGrid(inputLineMap.lineIdxToLineMap)

   # loop thru the grid of squares to get the redundant segs
   redundFound = gridOverlay.crawlThruGridToResolvePairedSegs(checkIf2LinesIntersect)
   # sort the refIdx, which is the line idx that is used to compare which line segs are redundant
   #  the criteria for redundancy are in the API in the squareGridCls
   gridOverlay.sortRefIdxs()

   # now that the pairs of the lines that intersect are determined - must handle those intersect lines
   gridOverlay.splitLinesThatCrossAndIntersect()

   return gridOverlay.returnLinesInGrid()

# return tuple of (boolean, lineClsList)
#  boolean indicates whether redundant line segs found - TRUE if yes
#  lineClsList - it is only a list of lineCls objects
def useGridOverlayToRemoveRedundantContigSegs(inputContigSegsMap, holesToInsert=[]):

   # first get the square grid dimensions xmin, xmax, ymin, and ymax from the list of contigSegs
   squareGridMinMax = (inputContigSegsMap.minX, inputContigSegsMap.maxX, \
                       inputContigSegsMap.minY, inputContigSegsMap.maxY)
   width = (squareGridMinMax[1] - squareGridMinMax[0]) * 2
   height = (squareGridMinMax[3] - squareGridMinMax[2]) * 2

   # create grid of squares given XMin, XMax, YMin, YMax

   # for now try squareDim = 10.0 as the max perp dist between 2 lines to be considered as overlap is 5.0
   squareDim = 10.0

   # generate the grid of squares as gridOverlay
   gridOverlay = gridCls(squareGridMinMax[0], squareGridMinMax[1], \
                         squareGridMinMax[2], squareGridMinMax[3], \
                         squareDim, True, contigSegsMap=inputContigSegsMap)

   # CHECKPOINT - DISPLAY GRID CHARACTERISTICS TO CHECK IF GRID CORRECTLY SET UP
   gridOverlay.displayGridData()

   for contigsSegs in inputContigSegsMap.getAllContigSegAsList():
      gridOverlay.insertContigSegIntoGrid(contigsSegs)

   for hole in holesToInsert:
      gridOverlay.insertHoleIntoGrid(hole)

 #  gridOverlay.displayGridSquareData()
   # print out logs that show which grids a certain line occupies
   for lineIdx in gridOverlay.lineSegIdxToGridsOccupied:
      gridCoords = gridOverlay.lineSegIdxToGridsOccupied.get(lineIdx)
      print("line " + str(lineIdx) + " occupies the following grids " + str(gridCoords))

      for cIdx in range(len(gridCoords)-1):
         if (abs(gridCoords[cIdx+1][0] - gridCoords[cIdx][0]) > 1) or \
            (abs(gridCoords[cIdx+1][1] - gridCoords[cIdx][1]) > 1):
            print("ERROR - line idx " + str(lineIdx) + " grid " + str(gridCoords[cIdx]) + " and grid " + str(gridCoords[cIdx+1]) + " are not adjacent")


   print("Number of lines processed is " + str(gridOverlay.linesProcessed))
   print("Number of lines inserted into grid is " + str(gridOverlay.linesInsertedIntoGrid))

   # loop thru the grid of squares to get the redundant segs
   redundFound = gridOverlay.crawlThruGridToResolvePairedSegs(checkIf2LinesRedundant, maxPerpDist=5.0)
   # sort the refIdx, which is the line idx that is used to compare which line segs are redundant
   #  the criteria for redundancy are in the API in the squareGridCls
   gridOverlay.sortRefIdxs()

   # log pairs of ref line idx to redundant segs
   for refIdx in gridOverlay.inOrderRefIdx:
      print("refIdx " + str(refIdx) + " has parallel segs " + str(gridOverlay.lineSegToPairedLineSegs.get(refIdx)))

   gridOverlay.crawlRefSegsToRmvRedundancy()

   return gridOverlay.returnLinesInGrid(), gridOverlay.returnRmvedLinesInGrid()

# this API is used to:
#   - given the lines that remain after the 1st pass of removing redundant contig segs
#     there may be cases where at one portion, contig seg A remains but in another portion,
#     contig seg B of the parallel contig segs is used, causing a staggering effect like
#      --------_____ for example - this function attempts to resolve this by
#     1) crawling thru the contig segs in order of len of web that the contig seg is a part of
#        - check if there is a hole or removed redundant portion that used to be attached
#           to the contig seg start / end property
#     2) if there is - check its corresponding redundant line seg that still exists
#         - if the redundant line seg still exists:
#               - restore the removed seg (hole)
#               - delete its corresponding redundant line seg that is still hanging around
def userGridOverlayToAlignCurves(inputContigSegsMap, removedRedundantLines):
   # first get the square grid dimensions xmin, xmax, ymin, and ymax from the list of contigSegs
   rmvRedundLinesMinMax = getXYMinMaxFromListOfLines(removedRedundantLines)

   squareGridMinMax = ( min(rmvRedundLinesMinMax[0], inputContigSegsMap.minX), \
                        max(rmvRedundLinesMinMax[1], inputContigSegsMap.maxX), \
                        min(rmvRedundLinesMinMax[2], inputContigSegsMap.minY), \
                        max(rmvRedundLinesMinMax[3], inputContigSegsMap.maxY) )

   width = (squareGridMinMax[1] - squareGridMinMax[0]) * 2
   height = (squareGridMinMax[3] - squareGridMinMax[2]) * 2

   # create grid of squares given XMin, XMax, YMin, YMax

   # for now try squareDim = 10.0 as the max perp dist between 2 lines to be considered as overlap is 5.0
   squareDim = 10.0

   # generate the grid of squares as gridOverlay
   gridOverlay = gridCls(squareGridMinMax[0], squareGridMinMax[1], \
                         squareGridMinMax[2], squareGridMinMax[3], \
                         squareDim, False, contigSegsMap=inputContigSegsMap)

   for contigSegHash, contigSeg in inputContigSegsMap.contigSegHashToContigSeg.items():
      gridOverlay.insertContigSegIntoGrid(contigSeg)

   # now that the contig segs have been inserted into the grid, now insert
   # the holes into the grid

   #  generate a map of hole terminal pts to holes
   mapOfHolePtsToHoles = {}
   for hole in removedRedundantLines:
      if not mapOfHolePtsToHoles.get(hole.getTermPt1AsTuple()):
         mapOfHolePtsToHoles[hole.getTermPt1AsTuple()] = [hole.hash]
      else:
         if not hole.hash in mapOfHolePtsToHoles[hole.getTermPt1AsTuple()]:
            mapOfHolePtsToHoles[hole.getTermPt1AsTuple()].append(hole.hash)
      if not mapOfHolePtsToHoles.get(hole.getTermPt2AsTuple()):
         mapOfHolePtsToHoles[hole.getTermPt2AsTuple()] = [hole.hash]
      else:
         if not hole.hash in mapOfHolePtsToHoles[hole.getTermPt2AsTuple()]:
            mapOfHolePtsToHoles[hole.getTermPt2AsTuple()].append(hole.hash)

      # insert hole into grid
      gridOverlay.insertHoleIntoGrid(hole)

   # refill holes if needed and remove disjoint shorter contig segs
   gridOverlay.refillParticularHolesWithExistingSegs(mapOfHolePtsToHoles)

# This API uses curve projection to remove any parallel / redundant contig segs
#  INPUT:
#   contigSegsPtsGraph object which contains the contig segs
#  This API does the following:
#   1) For each contig seg - combine the lineCls segments into BCurves
#   2) insert the lineCls portions of the contig seg into the grid - for BCurves
#      generate "representative" lineCls objects of the curve and insert those into
#      the grid
#   3) Starting with the longest contig segs - check in the contig segs for the
#      longest bCurves -> project these bCurves to see if there are redundant
#      lineCls objects - if there are, remove the redundant lineCls object
#   4) TENTATIVE - also use projection to close any gaps
def useGridOverlayAndCurveProjToRmvRedundContigSegs(contigSegsPtsGraph, alpha, epsilonVal, holes=[]):
   # convert each contig seg in the graph to be made up of combined bCurves and lineCls objects
   contigSegsPtsGraph.convertCSegsToLinesAndBCurves(alpha, epsilonVal)

   # first get the square grid dimensions xmin, xmax, ymin, and ymax from the list of contigSegs
   squareGridMinMax = (contigSegsPtsGraph.minX, contigSegsPtsGraph.maxX, \
                       contigSegsPtsGraph.minY, contigSegsPtsGraph.maxY)
   width = (squareGridMinMax[1] - squareGridMinMax[0]) * 2
   height = (squareGridMinMax[3] - squareGridMinMax[2]) * 2

   # create grid of squares given XMin, XMax, YMin, YMax

   # for now try squareDim = 10.0 as the max perp dist between 2 lines to be considered as overlap is 5.0
   squareDim = 10.0

   # generate the grid of squares as gridOverlay
   gridOverlay = gridCls(squareGridMinMax[0], squareGridMinMax[1], \
                         squareGridMinMax[2], squareGridMinMax[3], \
                         squareDim, False, contigSegsMap=contigSegsPtsGraph)

   # CHECKPOINT - DISPLAY GRID CHARACTERISTICS TO CHECK IF GRID CORRECTLY SET UP
   gridOverlay.displayGridData()

   for contigsSegs in inputContigSegsMap.getAllContigSegAsList():
      gridOverlay.insertContigSegIntoGrid(contigsSegs)

   for hole in holes:
      gridOverlay.insertHoleIntoGrid(hole)

   # now populate the lineCls objects adjacency map of the grid
   gridOverlay.insertAllLinesIntoAdjMap()

   # now crawl the contig segs and look for the longest bCurves to start projecting to see
   # if there are redundant lines that may or may not belong to bCurves and delete those lines
   gridOverlay.useBCurvesToProjectAndRmvRedundSegs()

# v2 of API above - this API will not do any processing. It will simply:
#  1) get the lineCls objects from contours and populate them into lineMap
#  2) dump lineMap info to imgName.json
#
#### performance stats to execute this function on HomerSimpson.png
#real	0m9.177s
#user	0m2.664s
#sys	0m0.352s
####
def generateLinesClsObjsFromRawImg(imgName):
   lineMap = generateLineMapClsContours(imgName, False)

   # dump firstLineMap maps to JSON
   jsonName = os.path.splitext(imgName)[0] + ".json"
   lineMap.dumpInfoToJSON(jsonName)

   return lineMap

__prog__ = os.path.basename(sys.argv[0])

def main(argv):
   helpString = """
                   imgProcessing.py - detect edges of images
                                    - and generate 3d model
                                    - takes 1 arguments:
                                     -i: input image name - can be raw image or json file
                                         if the input is json file - means that this image has
                                         already been processed up to a certain point
                """
   parser = argparse.ArgumentParser(
                       prog = __prog__,
                       description=helpString,
                       add_help=True)
   parser.add_argument('-i', action="store", dest="imgName", required=True, default=None, help="name of image input to process")
   args = parser.parse_args()

   # if JSON name passed - use JSON file to generate lineMap
   inputFileExt = os.path.splitext(args.imgName)[1]

   if inputFileExt.lower() == ".json":
      firstLineMap = readLineMapClsContoursFromJSON(args.imgName)
   else:
      firstLineMap = generateLinesClsObjsFromRawImg(args.imgName)

   # before generating / removing redundant segs - first
   # look at lineCls objects that visually cross each other but mathematically
   # register as 2 lines only
   #
   # will convert those 2 lines as 4 lines, where those 2 lines will be split
   # at the pt of the intersection - will use gridCls to look for lines that are
   # within proximity of each other that potentially visually cross each other

   # feed line map into fresh gridCls
   #
   # the modified lines are returned as lists
 #  resolvedIntersectLines = useGridOverlayToSeparateIntersectLines(firstLineMap)


   # contigSegsPtsGraph is determineContigSegFromLines class object
   contigSegsPtsGraph = genContigSegsUsingNodeGraph(lineMap=firstLineMap, drawContigSegs=False)

   # the removed redundant lines (holes) are also returned as a list
   linesThatRemain, removedRedundantLines = useGridOverlayToRemoveRedundantContigSegs(contigSegsPtsGraph)

  # contigSegsPtsGraph = genContigSegsUsingNodeGraph(lineList=linesThatRemain, drawContigSegs=False)
  # realignedLines = userGridOverlayToAlignCurves(contigSegsPtsGraph, removedRedundantLines)

  # for the lines that remain (now that the 1st pass in removing the redundant lines has been removed)
  # remainLineMap = lineMapCls()
  # remainLineMap.insertLinesToIdxMap(linesThatRemain)
  # contigSegsPtsGraph = genContigSegsUsingNodeGraph(lineMap=remainLineMap, drawContigSegs=False)

   # remove redundant lines using curve projection
  # linesThatRemain , removedRedundantLines = useGridOverlayAndCurveProjToRmvRedundContigSegs(contigSegsPtsGraph)
  
   firstLineMap.generateImgFromLines(os.path.splitext(args.imgName)[0] + "GRIDOVERLAY_ORIG_FROMLINEMAP" + testOutPathPart2)

   finalLineMap = lineMapCls()
   for lineEntry in linesThatRemain:
       finalLineMap.insertLineToIdxMap(lineEntry)
   finalLineMap.generateImgFromLines(os.path.splitext(args.imgName)[0] + "GRIDOVERLAY_RMV_REDUNDANT_CONTOURS_FROMLINEMAP" + testOutPathPart2)

   finalHoleMap = lineMapCls()
   for holeEntry in removedRedundantLines:
       finalHoleMap.insertLineToIdxMap(holeEntry)
   finalHoleMap.generateImgFromLines(os.path.splitext(args.imgName)[0] + "GRIDOVERLAY_REDUNDANT_HOLES" + testOutPathPart2)

if __name__ == '__main__':
   main(sys.argv)
