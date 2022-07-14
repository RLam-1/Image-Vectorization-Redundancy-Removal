#!/usr/bin/python

import numpy as np
import math
import copy
import queue


from linesAPIs import *

# NOTE: in this class the convention for what is start and end pt of a lineCls
#       object is that the start pt is defined as:
#            1) having the min X value
#            2) if the X value of both the start and end pt are the same, then
#               it is the min Y value
#
#  SHALL use points instead of lines as fundamental unit (ie. treat as graph
#   traversal)
#     cartesian points (start and end pt of lines) are treated as nodes in a graph
#     and their adjacency is determined by whether or not the 2 pts are the start / end pts
#     of a line
#
#  NOTE : need to convert np.array([x,y]) to tuple because np.array is NOT hashable

class determineContigSegFromLines:

   def __init__(self):
      # adjacenyMap will map pt and its adjacent pts
      #  with the form {(x, y) : [(adj1X, adj1Y), (adj2X, adj2Y)]}
      self.adjacencyMap = {}
      # this map will store the pts that have already been visited
      self.visitedPts  = {}
      # this map will store visited lines - note that flipped lines
      # are functionally the same(ie. line with start pt = x1,y1 and end pt = x2,y2)
      # is the same as line with start pt = x2,y2 and end pt = x1,y1 - thus need to
      # create APIs to translate the line so that it is consistent to avoid storing
      # the same line 2x and creating a map that is 2x as large as needed
      self.visitedLines = {}
      # this map will store generated contigsSegs with their corresponding contigsSegsID
      self.maxContigSegsIdx = 0
      self.contigsSegsIdxToContigsSegs = {}

      # define lineIdx to lineCls map
      self.maxLineIdx = 0
      self.lineIdxToLineCls = {}
      # this is the map of the lineCls obj to the line idx - this is used
      # during generation of contig seg where 2 points are turned into a line - if
      # line already exists - do NOT add new entry to the lineIdxToLineCls map
      self.lineClsToLineIdx = {}

      # this is map of the line idx to the contig idx that the line belongs to
      #  and the index / position of the line in the contig seg - thus the map is in the form
      #  {line idx : (contig idx, position idx)}
      self.lineIdxToContigSegIdxAndPos = {}

      # this is map of pts to the contig segs that have said pt as a termPt
      self.termPtToContigSegs = {}

      # this is map of contig seg to their LHS, RHS neighbors, where LHS neighbors
      # are contig segs with termPt the same as the contig segs LHSMostTermPt
      # and same for RHSMostTermPt - thus the map has format
      # {contigSegIdx : ([contigSegIdxs of LHS neighbors], [contigSegIdxs of RHS neighbors])}
      self.contigSegIdxToLRHSNeighbors = {}

   def getAllContigsSegsAsList(self):
      retContigSegList = []
      for idx in self.contigsSegsIdxToContigsSegs:
         retContigSegList.append(self.contigsSegsIdxToContigsSegs.get(idx))

      return retContigSegList

   # this API takes in an existing lineMap to be used
   # as the lineMap for the
   def assignLineMapAndCreateAdjMap(self, lineMap):
      self.lineIdxToLineCls = copy.deepcopy(lineMap)
      for lineEntry in lineMap:
         line = lineMap.get(lineEntry)
         if lineEntry > self.maxLineIdx:
            self.maxLineIdx = lineEntry
         self.lineClsToLineidx[line] = lineEntry

         if not line.checkIfLineIsPoint():
            startPt = (line.termPt1[0], line.termPt1[1])
            endPt = (line.termPt2[0], line.termPt2[1])
            newPt = False

            if not self.adjacencyMap.get(startPt, None):
               self.adjacencyMap[startPt] = [endPt]
            else:
               # need to check if pt has already been stored
               # in this map
               if endPt not in self.adjacencyMap[startPt]:
                  self.adjacencyMap[startPt].append(endPt)
                  newPt = True

            if not self.adjacencyMap.get(endPt, None):
               self.adjacencyMap[endPt] = [startPt]
            else:
               if newPt:
                  self.adjacencyMap[endPt].append(startPt)


   def populateAdjacencyMapFromLines(self, listOfLines):
      for line in listOfLines:
         # check if line is a pt
         if not line.checkIfLineIsPoint():
            startPt = (line.termPt1[0], line.termPt1[1])
            endPt = (line.termPt2[0], line.termPt2[1])
            newPt = False

            if not self.adjacencyMap.get(startPt, None):
               self.adjacencyMap[startPt] = [endPt]
            else:
               # need to check if pt has already been stored
               # in this map
               if endPt not in self.adjacencyMap[startPt]:
                  self.adjacencyMap[startPt].append(endPt)
                  newPt = True

            if not self.adjacencyMap.get(endPt, None):
               self.adjacencyMap[endPt] = [startPt]
            else:
               if newPt:
                  self.adjacencyMap[endPt].append(startPt)

   def insertLineIntoMap(self, line):
      lineIdx = self.maxLineIdx
      self.lineIdxToLineCls[lineIdx] = line
      self.maxLineIdx += 1
      return lineIdx

   def insertContigSegIntoMap(self, contigSeg):
      contigSegIdx = self.maxContigSegsIdx
      self.contigsSegsIdxToContigsSegs[contigSegIdx] = contigSeg
      self.maxContigSegsIdx += 1
      return contigSegIdx

   # This API takes as input a tuple ((x1, y1), (x2, y2))
   # where x1, y1 is the start pt and x2, y2 is the end pt of the line
   def insertLineIntoVisitedMap(self, lineTuple):
      if not self.visitedLines.get((lineTuple[0][0], lineTuple[0][1], lineTuple[1][0], lineTuple[1][1]), None) and \
         not self.visitedLines.get((lineTuple[1][0], lineTuple[1][1], lineTuple[0][0], lineTuple[0][1]), None):

         self.visitedLines[(lineTuple[0][0], lineTuple[0][1], lineTuple[1][0], lineTuple[1][1])] = True

   # this API checks if the line is already visited
   #  - returns TRUE if it is, FALSE otherwise
   #
   #  INPUT: tuple of form ((x1, y1), (x2, y2)) where (x1, y1) is the start pt
   #         and (x2, y2) is the end pt
   def checkIfLineIsAlreadyVisited(self, lineTuple):
      return (self.visitedLines.get((lineTuple[0][0], lineTuple[0][1], lineTuple[1][0], lineTuple[1][1])) or \
              self.visitedLines.get((lineTuple[1][0], lineTuple[1][1], lineTuple[0][0], lineTuple[0][1])))

   def givenPtsListCreateContigSegs(self, ptsList):
      continueContigSeg = False
      contigSeg = contigsSegsCls()

      for i in range(len(ptsList)-2):
         line = lineCls()
         line.setStartPt(np.array([ptsList[i][0], ptsList[i][1]]))
         line.setEndPt(np.array([ptsList[i+1][0], ptsList[i+1][1]]))

         lineIdx = self.lineClsToLineIdx.get(line, self.insertLineIntoMap(line))
         continueContigSeg = contigSeg.insertLineToContigSeg(lineIdx, line)

         if not continueContigSeg:
            # push the contig seg into the map
            contigSeg.finalizeContigSeg()
            contigSegIdx = self.insertContigSegIntoMap(contigSeg)
            for i in range(len(contigSeg.lineIdxs)):
               self.lineIdxToContigSegIdxAndPos[contigSeg.lineIdxs[i]] = (contigSegIdx, i)

            contigSeg = contigsSegsCls()
            contigSeg.insertLineToContigSeg(lineIdx, line)

      # after the loop if contig seg is not empty - must push it into map
      if len(contigSeg.lines) > 0:
         self.insertContigSegIntoMap(contigSeg)

      # generate map of {coord pt : [contig idxs that have termPt as pt]}
      for contigEntry in self.contigsSegsIdxToContigsSegs:
         contigSeg = self.contigsSegsIdxToContigsSegs[contigEntry]
         if not self.termPtToContigSegs.get(contigSeg.LHSMostTermPt):
            self.termPtToContigSegs[contigSeg.LHSMostTermPt] = [contigEntry]
         elif contigEntry not in self.termPtToContigSegs[contigSeg.LHSMostTermPt]:
            self.termPtToContigSegs[contigSeg.LHSMostTermPt].append(contigEntry)
         if not self.termPtToContigSegs.get(contigSeg.RHSMostTermPt):
            self.termPtToContigSegs[contigSeg.RHSMostTermPt] = [contigEntry]
         elif contigEntry not in self.termPtToContigSegs[contigSeg.RHSMostTermPt]:
            self.termPtToContigSegs[contigSeg.RHSMostTermPt].append(contigEntry)

      # generate the map of {contig seg idx : ([LHSNeighbors], [RHSNeighbors])}
      for ptEntry in self.termPtToContigSegs:
         contigSegs = self.termPtToContigSegs[ptEntry]
         for i in range(len(contigSegs)):
            if not self.contigSegIdxToLRHSNeighbors.get(i):
               self.contigSegIdxToLRHSNeighbors[i] = [None, None]
            contigNeighbors = [contigSegs[j] for j in range(len(contigSegs)) if j != i]
            if np.array_equal(ptEntry, self.contigsSegsIdxToContigsSegs.get(contigSegs[i]).LHSMostTermPt):
               self.contigSegIdxToLRHSNeighbors[i][0] = contigNeighbors
            elif np.array_equal(ptEntry, self.contigsSegsIdxToContigsSegs.get(contigSegs[i]).RHSMostTermPt):
               self.contigSegIdxToLRHSNeighbors[i][1] = contigNeighbors

      for contigSegIdx in self.contigSegIdxToLRHSNeighbors:
         self.contigSegIdxToLRHSNeighbors[contigSegIdx] = tuple(self.contigSegIdxToLRHSNeighbors[contigSegIdx])

   # this API generates a map of an open contig seg idx to its cumulative length
   #
   # a contig seg is considered open if >=1 of its termPts is NOT connected to another
   #  contig seg
   #
   # the cumulative length is the length of own contig seg plus all contig segs
   # connected to it and all the contig segs connected to those and so on and so forth
   #  NOTE - each contig seg is only counted ONCE in case of loops
   def genMapOfOpenContigSegIdxToCumulativeLen(self):
      contigSegIdxToCumulativeLenMap = {}
      cumulativeLenToContigSegIdxMap = {}
      for contigSegIdx in self.contigSegIdxToLRHSNeighbors:
         contigSegIdxNeighbors = self.contigSegIdxToLRHSNeighbors[contigSegIdx]
         if not contigSegIdxNeighbors[0]:
            neighborIdx = 1
         elif not contigSegIdxNeighbors[1]:
            neighborIdx = 0
         else:
            neighborIdx = None

         contigCumulativeLen = self.contigSegIdxToContigSeg[contigSegIdx].length

         if neighborIdx:
            neighborQueue = copy.deepcopy(contigSegIdxNeighbors[neighborIdx])
            includedContigSegs = [contigSegIdx]
            while neighborQueue:
               # pop the index we are visiting
               visitingIdx = neighborQueue.pop(0)
               if visitingIdx not in includedContigSegs:
                  includedContigSegs.append(visitingIdx)
                  contigCumulativeLen += self.contigSegIdxToContigSeg[visitingIdx].length
                  # put neighbors in queue
                  fullNeighbors = []
                  if self.contigSegIdxToLRHSNeighbors[visitingIdx][0]:
                     fullNeighbors.extend(self.contigSegIdxToLRHSNeighbors[visitingIdx][0])
                  if self.contigSegIdxToLRHSNeighbors[visitingIdx][1]:
                     fullNeighbors.extend(self.contigSegIdxToLRHSNeighbors[visitingIdx][1])

                  for neighbor in fullNeighbors:
                     if neighbor not in includedContigSegs:
                        neighborQueue.append(neighbor)

         if not cumulativeLenToContigSegIdxMap.get(contigCumulativeLen):
            cumulativeLenToContigSegIdxMap[contigCumulativeLen] = [contigSegIdx]
         else:
            cumulativeLenToContigSegIdxMap[contigCumulativeLen].append(contigSegIdx)

      # sort the map so that the longest segs are first
      contigSegIdxToCumulativeLenMap = {}
      cumulativeLenKeys = sorted(cumulativeLenToContigSegIdxMap, reverse=True)
      for cumulativeLenEntry in cumulativeLenKeys:
         for contigSegEntry in cumulativeLenToContigSegIdxMap[contigSegEntry]:
            contigSegIdxToCumulativeLenMap[contigSegEntry] = cumulativeLenEntry

      return contigSegIdxToCumulativeLenMap

   def drawLinesToImg(self, img, arrowed):
      line_thickness = 1
      for idx, line in self.lineIdxToLineCls.items():
         if arrowed:
            cv.arrowedLine(img, (line.termPt1[0], line.termPt1[1]), (line.termPt2[0], line.termPt2[1]), (0, 255, 0), line_thickness, tipLength=0.5)
         else:
            cv.line(img, (int(line.termPt1[0]), int(line.termPt1[1])), (int(line.termPt2[0]), int(line.termPt2[1])), (0, 255, 0), line_thickness)

   def genContigSegsUsingGraphAdj(self):
      # NOTE: queue and stack can both use Python Lists
      #       - for stack, use append to add to stack
      #                    use pop to remove last item
      #       - for queue, use append to add to queue
      #                    use pop(0) to remove first item
      #
      #  This API crawls thru the adjacency map and performs depth-first
      #   search of contig segs by pushing into the stack ((adjPt),(iterPt)), where
      #  adjPt is the adjacent pt of the iterPt, which is the reference iteration pt that is looped
      #  in the for loop
      #
      #  as we pop the stack, store it into contig seg list and push its adjacent pts into the stack
      #  in the same format - if this pt has more than 1 neighbor (not including the pt that it came from)
      #  then we break the contig seg by passing the contigSegList into another API to generate at least 1 contig seg
      #  depending if the pts are lines that satisfy contig seg requirement (such as not changing direction of more than 90 degrees)
      #
      #  conditions of terminating potential contig seg list:
      #      1) pt has != 2 neighbors
      adjStack = []
      potentialContigSegPtsList = []

      # first pass - handle everything BUT isolated loops
      # next pass - if there are still unvisited pts remaining with 2 neighbors - they are part of isolated loops
      for i in range(2):
         for pt in self.adjacencyMap:
            # only process the pt if it is not already visited AND
            # only if it is definitely an endpt of a contigSeg, meaning it has != 2 neighbors
            #  in the 1st iteration - this will catch the longest contig seg pts
            #
            #  on the 2nd iteration - allow pt with 2 neighbors to start as this is an isolated
            #  closed loop
            if not self.visitedPts.get(pt, None) and \
               not ((len(self.adjacencyMap[pt]) == 2) and (i == 0)):
               # first populate adjStack with immediate neighbors that have not been visited
               for adjPt in self.adjacencyMap[pt]:
                  if not self.visitedPts.get(adjPt, None):
                     adjStack.append((adjPt, pt))

               self.visitedPts[pt] = True

               while len(adjStack) > 0:
                  termPotentialContigSegPtList = False
                  # get the first elem of the stack
                  topElem = adjStack.pop()
                  # handle potentialContigSegPtsList
                  #  if empty - put in the ref pt
                  if not self.checkIfLineIsAlreadyVisited(topElem):
                     self.insertLineIntoVisitedMap(topElem)

                     if not potentialContigSegPtsList:
                        potentialContigSegPtsList.append(topElem[1])

                     # populate the current pt
                     potentialContigSegPtsList.append(topElem[0])

                     # push the neighbors of the current pt into stack
                     for adjPt in self.adjacencyMap[topElem[0]]:
                        if not self.visitedPts.get(adjPt, None):
                           adjStack.append((adjPt, topElem[0]))

                     self.visitedPts[topElem[0]] = True

                     # determine if we should terminate the potentialContigSegPtsList
                     # (ie. if the pts in the potentialContigSegPtsList make up a potential contig seg) based
                     # on conditions above
                     if len(self.adjacencyMap[topElem[0]]) != 2:
                        termPotentialContigSegPtList = True
                     else:
                        # if it has only 2 neighbors - check to see if next pt has been visited or not
                        #   this is to close off loops
                        # get the neighbor that is NOT is stored with this pt
                        neighbors = self.adjacencyMap[topElem[0]]
                        if topElem[1] == neighbors[0]:
                           neighborToCheck = neighbors[1]
                        else:
                           neighborToCheck = neighbors[0]

                        if self.visitedPts.get(neighborToCheck, None):
                           potentialContigSegPtsList.append(neighborToCheck)
                           termPotentialContigSegPtList = True

                     # terminate the potential contig seg pt list (ie.
                     #   the current pt is a boundary condition of the contig seg
                     #   such that the contig seg list must be determined either b/c
                     #  1) the current pt topElem[0] has != 2 neighbors
                     #  2) the current pt has 2 neighbors but the next neighbor (not the ref one)
                     #     has already been visited (loop)
                     if termPotentialContigSegPtList:
                        # process the termPotentialContigSegPtList and push contig segs into map
                        self.givenPtsListCreateContigSegs(potentialContigSegPtsList)
                        # clear the termPotentialContigSegPtList
                        potentialContigSegPtsList = []
