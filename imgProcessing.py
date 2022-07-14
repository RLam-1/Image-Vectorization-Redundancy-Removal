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

testImg = "/home/rudy/imgProjPython/HomerSimpson.png"
testOutImg = "/home/rudy/imgProjPython/HomerSimpsonOut.png"
autoEdgeImg = "/home/rudy/imgProjPython/HomerSimpsonAuto.png"
wideEdgeImg = "/home/rudy/imgProjPython/HomerSimpsonWide.png"
tightEdgeImg = "/home/rudy/imgProjPython/HomerSimpsonTight.png"
testOutPathPart1 = "/home/rudy/imgProjPython/HomerSimpsonOut"
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
      cv.imwrite(testOutPathPart1 + testOutType + "PREALIGN" + testOutPathPart2, imgOutLine)

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

def crawlContourForUniqueSeg(lineMap, contourIdx, delContour1, queue):
   contourUnique = lineMap.crawlContoursForContiguousParallelSegs(contourIdx, contourIdx, delContour1)
   if not delContour1:
      queue.put({1:contourUnique})
   else:
      queue.put({2:contourUnique})

def multiprocessCrawlContourForUniqueSeg(lineMap, contourIdx):
   uniqueContQ = multiprocessing.Queue()
   keepCont1Cont2List = [False, True]

   uniqueProcessesToJoin = []
   for keepEntry in keepCont1Cont2List:
      p = multiprocessing.Process(target=crawlContourForUniqueSeg, args=(lineMap, contourIdx, keepEntry, uniqueContQ))
      time.sleep(5)
      uniqueProcessesToJoin.append(p)
      p.start()

   for process in uniqueProcessesToJoin:
      process.join()

   retDict = {}

   while not uniqueContQ.empty():
      retDict.update(uniqueContQ.get())

   return retDict

### given list of contours of the form contigsSegsCls
### get the XMin, XMax, YMin, and YMax in the form of tuple
def getXYMinMaxFromListOfContigsSegs(inputContigsSegsList):
   XMin = XMax = YMin = YMax = 0
   for contigsSegs in inputContigsSegsList:
      for contigsSegsLine in contigsSegs.lines:
         if contigsSegsLine.termPt1[0] < XMin:
            XMin = contigsSegsLine.termPt1[0]
         elif contigsSegsLine.termPt1[0] > XMax:
            XMax = contigsSegsLine.termPt1[0]
         if contigsSegsLine.termPt2[0] < XMin:
            XMin = contigsSegsLine.termPt2[0]
         elif contigsSegsLine.termPt2[0] > XMax:
            XMax = contigsSegsLine.termPt2[0]

         if contigsSegsLine.termPt1[1] < YMin:
            YMin = contigsSegsLine.termPt1[1]
         elif contigsSegsLine.termPt1[1] > YMax:
            YMax = contigsSegsLine.termPt1[1]
         if contigsSegsLine.termPt2[1] < YMin:
            YMin = contigsSegsLine.termPt2[1]
         elif contigsSegsLine.termPt2[1] > YMax:
            YMax = contigsSegsLine.termPt2[1]

   return (XMin, XMax, YMin, YMax)

### given map of line idx to lineCls objects
### get the XMin, XMax, YMin, YMax in the form of tuple
def getXYMinMaxFromLineMap(lineMap):
   XMin = XMax = YMin = YMax = 0
   for lineEntry in lineMap:
      line = lineMap[lineEntry]
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


### given lineCls objects - generate contig seg
#   using determineContigSegFromLines class
#   that uses pts as nodes in a graph and then traversing graph
####
##  Performance
#
#real	0m2.043s
#user	0m1.981s
#sys	0m0.053s
def genContigSegsUsingNodeGraph(lineMap, drawContigSegs=True):
   ptsGraph = determineContigSegFromLines()
   ptsGraph.populateAdjacencyMapFromLines(lineMap.getAllLinesAsList())
   ptsGraph.genContigSegsUsingGraphAdj()

   if drawContigSegs:
      drawListOfContigSegsToImg(ptsGraph.getAllContigsSegsAsList(), lineMap.imgHeight, lineMap.imgWidth, testOutPathPart1 + "CONTIGS_SEGS_FROM_GRAPH" + testOutPathPart2, False)

      imgOutLine = np.ones([lineMap.imgHeight, lineMap.imgWidth], dtype=np.uint8)*255
      ptsGraph.drawLinesToImg(imgOutLine, False)
      cv.imwrite(testOutPathPart1 + "LINES_FROM_GRAPH" + testOutPathPart2, imgOutLine)

   return ptsGraph

# return tuple of (boolean, lineClsList)
#  boolean indicates whether redundant line segs found - TRUE if yes
#  lineClsList - it is only a list of lineCls objects
def useGridOverlayToRemoveRedundantContigSegs(contigsSegsList):

   # first get the square grid dimensions xmin, xmax, ymin, and ymax from the list of contigSegs
   squareGridMinMax = getXYMinMaxFromListOfContigsSegs(contigsSegsList)
   width = (squareGridMinMax[1] - squareGridMinMax[0]) * 2
   height = (squareGridMinMax[3] - squareGridMinMax[2]) * 2

   # create grid of squares given XMin, XMax, YMin, YMax

   # for now try squareDim = 10.0 as the max perp dist between 2 lines to be considered as overlap is 5.0
   squareDim = 10.0

   # generate the grid of squares as gridOverlay
   gridOverlay = gridCls(squareGridMinMax[0], squareGridMinMax[1], squareGridMinMax[2], squareGridMinMax[3], squareDim)

   # CHECKPOINT - DISPLAY GRID CHARACTERISTICS TO CHECK IF GRID CORRECTLY SET UP
   gridOverlay.displayGridData()

   for contigsSegs in contigsSegsList:
      gridOverlay.insertContigSegIntoGrid(contigsSegs)

 #  gridOverlay.displayGridSquareData()
   # print out logs that show which grids a certain line occupies
   for lineIdx in gridOverlay.lineSegIdxToGridsOccupied:
      gridCoords = gridOverlay.lineSegIdxToGridsOccupied.get(lineIdx, None)
      print("line " + str(lineIdx) + " occupies the following grids " + str(gridCoords))

      for cIdx in range(len(gridCoords)-1):
         if (abs(gridCoords[cIdx+1][0] - gridCoords[cIdx][0]) > 1) or \
            (abs(gridCoords[cIdx+1][1] - gridCoords[cIdx][1]) > 1):
            print("ERROR - line idx " + str(lineIdx) + " grid " + str(gridCoords[cIdx]) + " and grid " + str(gridCoords[cIdx+1]) + " are not adjacent")


   print("Number of lines processed is " + str(gridOverlay.linesProcessed))
   print("Number of lines inserted into grid is " + str(gridOverlay.linesInsertedIntoGrid))

   # loop thru the grid of squares to get the redundant segs
   redundFound = gridOverlay.crawlThruGridToGetRedundantSegs()
   # sort the refIdx, which is the line idx that is used to compare which line segs are redundant
   #  the criteria for redundancy are in the API in the squareGridCls
   gridOverlay.sortRefIdxs()

   # log pairs of ref line idx to redundant segs
   for refIdx in gridOverlay.inOrderRefIdx:
      print("refIdx " + str(refIdx) + " has parallel segs " + str(gridOverlay.lineSegToParallelLineSegs.get(refIdx, None)))

  # gridOverlay.crawlLineSegToParallelLineSegsMapForRedundSegsToDelete()
   gridOverlay.crawlRefSegsToRmvRedundancy()

   resultLineMap = copy.deepcopy(gridOverlay.lineSegIdxToLineSeg)

   for refSegIdx in gridOverlay.refSegIdxToCroppedLineSegs:
      resultLineMap[refSegIdx] = gridOverlay.refSegIdxToCroppedLineSegs[refSegIdx]

   resultLinesList = []
   for refIdx in resultLineMap:
      if type(resultLineMap[refIdx]) == list:
          resultLinesList.extend(resultLineMap[refIdx])
      else:
          resultLinesList.append(resultLineMap[refIdx])

  # for contigIdx in gridOverlay.contigSegIdxToLinesToDel:
  #    contigSegLinesToDel = gridOverlay.contigSegIdxToLinesToDel.get(contigIdx, None)
  #    print("contig idx " + str(contigIdx) + " has the following lines to delete " + str(contigSegLinesToDel))

  #    if contigSegLinesToDel:
  #       gridOverlay.contigSegIdxToContigSeg[contigIdx].deleteLinesFromContigSegByLineIdxs(contigSegLinesToDel)
  #       print("contig idx " + str(contigIdx) + " has the following line idxs " + str(gridOverlay.contigSegIdxToContigSeg[contigIdx].lineIdxs))
  #       print("contig idx " + str(contigIdx) + " has the following groups of segments " + str(gridOverlay.contigSegIdxToContigSeg[contigIdx].contigSegments))

      # draw contig segs 547, 437, 491, 612, 432, 460 from gridOverlay
 #     contigsToDraw = [gridOverlay.contigSegIdxToContigSeg[547], gridOverlay.contigSegIdxToContigSeg[437], gridOverlay.contigSegIdxToContigSeg[491], gridOverlay.contigSegIdxToContigSeg[612], gridOverlay.contigSegIdxToContigSeg[432], gridOverlay.contigSegIdxToContigSeg[460]]
  #    contigsImgName = testOutPathPart1 + "AFTER_DEL_COLORCONTIG_547_437_491_612_432_460" + testOutPathPart2
   #   drawListOfColorContigSegsToImg(contigsToDraw, lineMap.imgHeight, lineMap.imgWidth, contigsImgName, False, gridOverlay.lineSegIdxToLineSeg)

   # draw entire HOMER SIMPSON from gridOverlay contig segs
   allContigSegs = []
   for contigIdx in gridOverlay.contigSegIdxToContigSeg:
      allContigSegs.append(gridOverlay.contigSegIdxToContigSeg[contigIdx])

   #drawListOfContigSegsToImg(allContigSegs, height, width, testOutPathPart1 + "GRIDOVERLAY_HOMER_SIMPSON_RMV_REDUNDANT_CONTOURS" + testOutPathPart2, False, resultLineMap)

   finalLineMap = lineMapCls()
   for lineEntry in resultLinesList:
       finalLineMap.insertLineToIdxMap(lineEntry)
   finalLineMap.generateImgFromLines(testOutPathPart1 + "GRIDOVERLAY_HOMER_SIMPSON_RMV_REDUNDANT_CONTOURS_FROMLINEMAP" + testOutPathPart2)

   return (redundFound, resultLinesList)

def handleProcessingOfContoursInImg(lineMap):

   ### <-THIS IS THE API that generates UNIQUE CONTOURS
   ###   within the lineMapCls and stores them in
   #
   ###   lineMap.contourIdxToUniqueContigSegs
   #
   ### which is of the form {contourIdx : uniqueContour of type contigsSegsCls}
   lineMap.processUniqueContsIntoUniqueContigSegs()

   ### NEXT - TAKE THE UNIQUE CONTOURS AND SEE IF ANY SEGMENTS OVERLAP

   # 1) First - create grid of square units that will overlay all of the contours
   #   grid needs XMin, XMax, YMin, YMax, and the dimension of the square grid
   #   get XMin, XMax, YMin, YMax from of existing contours, which in this case is
   #   stored in lineMap.contourIdxToUniqueContigSegs -> this map is a map of { contourIdx : listOfContigSegsForContour }
   contigsSegsList = []
   for contourIdx in lineMap.contourIdxToUniqueContigSegs:
      contigsSegsList.extend(lineMap.contourIdxToUniqueContigSegs[contourIdx])

   foundNonContigIdx = False

   for contigsIdx in range(len(contigsSegsList)):
      if not contigsSegsList[contigsIdx].checkContigSegIsContiguous():
         print("contigsSegs with idx " + str(contigsIdx) + " is not contiguous")
         foundNonContigIdx = True

   if foundNonContigIdx:
      sys.exit()

   drawListOfContigSegsToImg(contigsSegsList, lineMap.imgHeight, lineMap.imgWidth, testOutPathPart1 + "PRE_GRIDOVERLAY_HOMER_SIMPSON_FILTER_SELF_ONLY_CONTOUR" + testOutPathPart2, False)

   squareGridMinMax = getXYMinMaxFromListOfContigsSegs(contigsSegsList)

   # create grid of squares given XMin, XMax, YMin, YMax

   # for now try squareDim = 10.0 as the max perp dist between 2 lines to be considered as overlap is 5.0
   squareDim = 10.0

   # generate the grid of squares as gridOverlay
   gridOverlay = gridCls(squareGridMinMax[0], squareGridMinMax[1], squareGridMinMax[2], squareGridMinMax[3], squareDim)

   # CHECKPOINT - DISPLAY GRID CHARACTERISTICS TO CHECK IF GRID CORRECTLY SET UP
   gridOverlay.displayGridData()

   for contigsSegs in contigsSegsList:
      gridOverlay.insertContigSegIntoGrid(contigsSegs)

 #  gridOverlay.displayGridSquareData()
   for lineIdx in gridOverlay.lineSegIdxToGridsOccupied:
      gridCoords = gridOverlay.lineSegIdxToGridsOccupied.get(lineIdx, None)
      print("line " + str(lineIdx) + " occupies the following grids " + str(gridCoords))

      for cIdx in range(len(gridCoords)-1):
         if (abs(gridCoords[cIdx+1][0] - gridCoords[cIdx][0]) > 1) or \
            (abs(gridCoords[cIdx+1][1] - gridCoords[cIdx][1]) > 1):
            print("line idx " + str(lineIdx) + " grid " + str(gridCoords[cIdx]) + " and grid " + str(gridCoords[cIdx+1]) + " are not adjacent")


   print("Number of lines processed is " + str(gridOverlay.linesProcessed))
   print("Number of lines inserted into grid is " + str(gridOverlay.linesInsertedIntoGrid))

   gridOverlay.crawlThruGridToGetRedundantSegs()
   gridOverlay.sortRefIdxs()

   for refIdx in gridOverlay.inOrderRefIdx:
      print("refIdx " + str(refIdx) + " has parallel segs " + str(gridOverlay.lineSegToParallelLineSegs.get(refIdx, None)))

   priSegToSecTuples = gridOverlay.convertLineSegToLineSegsTupleMap(gridOverlay.lineSegToParallelLineSegs, 0, len(list(gridOverlay.lineSegToParallelLineSegs.keys()))-1)

   gridOverlay.crawlLineSegToParallelLineSegsMapForRedundSegsToDelete()

   for refIdx in gridOverlay.inOrderRefIdx:
      print("refIdx " + str(refIdx) + " belonging to contig seg " + str(gridOverlay.lineSegIdxToContigSegIdxs[refIdx]) + " with contig seg length " + str(gridOverlay.contigSegIdxToContigSeg[gridOverlay.lineSegIdxToContigSegIdxs[refIdx]].length) + " has parallel tupled segs " + str(priSegToSecTuples.get(refIdx, None)))

   for contigIdx in gridOverlay.contigSegIdxToLinesToDel:
      contigSegLinesToDel = gridOverlay.contigSegIdxToLinesToDel.get(contigIdx, None)
      print("contig idx " + str(contigIdx) + " has the following lines to delete " + str(contigSegLinesToDel))

      if contigSegLinesToDel:
         gridOverlay.contigSegIdxToContigSeg[contigIdx].deleteLinesFromContigSegByLineIdxs(contigSegLinesToDel)
         print("contig idx " + str(contigIdx) + " has the following line idxs " + str(gridOverlay.contigSegIdxToContigSeg[contigIdx].lineIdxs))
         print("contig idx " + str(contigIdx) + " has the following groups of segments " + str(gridOverlay.contigSegIdxToContigSeg[contigIdx].contigSegments))

      # draw contig segs 547, 437, 491, 612, 432, 460 from gridOverlay
 #     contigsToDraw = [gridOverlay.contigSegIdxToContigSeg[547], gridOverlay.contigSegIdxToContigSeg[437], gridOverlay.contigSegIdxToContigSeg[491], gridOverlay.contigSegIdxToContigSeg[612], gridOverlay.contigSegIdxToContigSeg[432], gridOverlay.contigSegIdxToContigSeg[460]]
  #    contigsImgName = testOutPathPart1 + "AFTER_DEL_COLORCONTIG_547_437_491_612_432_460" + testOutPathPart2
   #   drawListOfColorContigSegsToImg(contigsToDraw, lineMap.imgHeight, lineMap.imgWidth, contigsImgName, False, gridOverlay.lineSegIdxToLineSeg)

   # draw entire HOMER SIMPSON from gridOverlay contig segs
   allContigSegs = []
   for contigIdx in gridOverlay.contigSegIdxToContigSeg:
      allContigSegs.append(gridOverlay.contigSegIdxToContigSeg[contigIdx])

   drawListOfContigSegsToImg(allContigSegs, lineMap.imgHeight, lineMap.imgWidth, testOutPathPart1 + "GRIDOVERLAY_HOMER_SIMPSON_FILTER_NOT_SELF_CONTOUR" + testOutPathPart2, False, gridOverlay.lineSegIdxToLineSeg)

 #  for contourIdx, uniqueContigSegs in lineMap.contourIdxToUniqueContigSegs.items():
 #     drawListOfContigSegsToImg(uniqueContigSegs, lineMap.imgHeight, lineMap.imgWidth, testOutPathPart1 + "CONTOUR_" + str(contourIdx) + "_CONTOUR1_CONTOUR2" + testOutPathPart2, False)

 #  cv.imshow("original", imgGray)
 #  cv.imshow("filtered image", imgOut)
 #  cv.waitKey(0)
 #  edges = np.uint8(edges)
  # edges = cv.Sobel(blurImg, cv.CV_8U, 1, 1, ksize=5)

 #  plt.subplot(121),plt.imshow(img,cmap = 'gray')
 #  plt.title('Original Image'), plt.xticks([]), plt.yticks([])
 #  plt.subplot(122),plt.imshow(edges,cmap = 'gray')
 #  plt.title('Edgenume Image'), plt.xticks([]), plt.yticks([])
 #  cv.imshow("original", img)
 #  cv.imshow("edges", np.hstack([wide, tight, auto]))
 #  cv.waitKey(0)

  # plt.show()

# this API is to handle all processing calls that
#  are then dumped into JSON - this is to speed up debugging process
def lineMapStagesBeforeDumpIntoJSON(imgName):
   lineMap = generateLineMapClsContours(imgName, True)

   # for each contour handle its own parallel segments multiprocessCrawlContourForUniqueSeg(lineMap, 62)
   #  delete both contour1 and contour2 -> whether to delete contour1 (outside for loop)
   #   or contour 2 (inside for loop)
   for contourIdx in lineMap.lineContourToLineIdxs:
      lineMap.contourIdxToUniqueConts[contourIdx] = multiprocessCrawlContourForUniqueSeg(lineMap, contourIdx)

   # dump firstLineMap maps to JSON
   jsonName = os.path.splitext(imgName)[0] + ".json"
   lineMap.dumpInfoToJSON(jsonName)

   return lineMap

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
                                         NOTE: a suggested test image is
                                         /home/rudy/imgProjPython/HomerSimpson.png
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

   imgLineMapClassObj = firstLineMap

   ptsGraph = genContigSegsUsingNodeGraph(firstLineMap, False)
   listOfContigSegs = ptsGraph.getAllContigsSegsAsList()
   #NOTE - resTuple is tuple of (boolean indicating whether redundant line seg found,
   #                             list of lines denoted as lineCls objects)
   resTuple = useGridOverlayToRemoveRedundantContigSegs(listOfContigSegs)

  # handleProcessingOfContoursInImg(firstLineMap)

if __name__ == '__main__':
   main(sys.argv)
