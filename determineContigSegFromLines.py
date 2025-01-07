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
      # store the minX, maxX, minY, maxY of this set of contig segs
      self.minX = self.maxX = self.minY = self.maxY = float(0)
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

      self.contigSegHashToContigSeg = {}

      # define lineIdx to lineCls map
      self.lineHashToLineCls = {}

      # this is map of the line idx to the contig idx that the line belongs to
      #  and the index / position of the line in the contig seg - thus the map is in the form
      #  {line idx : (contig idx, position idx)}
      self.lineIdxToContigSegIdxAndPos = {}

      # this is map of pts to the contig segs that have said pt as a termPt
      self.termPtToContigSegsHash = {}

      # this is a map of contig seg hash to the length of the web that the contig seg
      # belongs to - a web is defined as the contig segs that are connected to each other (hence a web )
      self.contigSegHashToWebLen = {}
      # also have map of contig seg hash to other contig seg hashes in the web
      self.contigSegHashToOtherContigSegsHashInWeb = {}
      # this is map of contig seg to their LHS, RHS neighbors, where LHS neighbors
      # are contig segs with termPt the same as the contig segs LHSMostTermPt
      # and same for RHSMostTermPt - thus the map has format
      # {contigSegIdx : [[contigSegIdxs of LHS neighbors], [contigSegIdxs of RHS neighbors]]}
      self.contigSegIdxToLRHSNeighbors = {}

   def getAllContigSegAsList(self):
      retContigSegList = []
      for hashIdx in self.contigSegHashToContigSeg:
         retContigSegList.append(self.contigSegHashToContigSeg.get(hashIdx))

      return retContigSegList

   # this API takes in an existing lineMap to be used
   # as the lineMap for the
   def assignLineMapAndCreateAdjMap(self, lineMap):
      self.lineIdxToLineCls = copy.deepcopy(lineMap)
      for lineEntry, line in lineMap.items():

         if not line.checkIfLineIsPoint():
            startPt = line.getTermPt1AsTuple()
            endPt = line.getTermPt2AsTuple()
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
            startPt = line.getTermPt1AsTuple()
            endPt = line.getTermPt2AsTuple()
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
      self.lineHashToLineCls[line.hash] = line
      return line.hash

   def insertContigSegIntoMap(self, contigSeg):
      contigSeg.finalizeContigSeg()
      self.contigSegHashToContigSeg[contigSeg.hash] = contigSeg
      return contigSeg.hash

   def convertCSegsToLinesAndBCurves(self, alphaMax, epsilonVal):
      for hash, cs in self.contigSegHashToContigSeg:
         cs.convertContigSegToLinesAndBCurves(alphaMax, epsilonVal)

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
      contigSeg = contigSegCls()

      for i in range(len(ptsList)-2):
         line = lineCls()
         line.setStartPt(np.array([ptsList[i][0], ptsList[i][1]]))
         line.setEndPt(np.array([ptsList[i+1][0], ptsList[i+1][1]]))

         if not line.checkIfLineIsPoint():
            if not self.lineHashToLineCls.get(line.hash):
               self.insertLineIntoMap(line)

            continueContigSeg = contigSeg.insertLineToContigSeg(line.hash, line)

            if continueContigSeg is None:
               # push the contig seg into the map
               contigSegIdx = self.insertContigSegIntoMap(contigSeg)
               for i in range(len(contigSeg.lineIdxs)):
                  self.lineIdxToContigSegIdxAndPos[contigSeg.lineIdxs[i]] = (contigSegIdx, i)

               contigSeg = contigSegCls()
               contigSeg.insertLineToContigSeg(line.hash, line)

      # after the loop if contig seg is not empty - must push it into map
      if len(contigSeg.lines) > 0:
         self.insertContigSegIntoMap(contigSeg)

   def genMapsOfCSNeighbors(self):
      # generate map of {coord pt : [contig idxs that have termPt as pt]}
      for contigEntry, contigSeg in self.contigSegHashToContigSeg.items():
         if not self.termPtToContigSegsHash.get(contigSeg.getLHSMostTermPtAsTuple()):
            self.termPtToContigSegsHash[contigSeg.getLHSMostTermPtAsTuple()] = [contigEntry]
         elif contigEntry not in self.termPtToContigSegsHash[contigSeg.getLHSMostTermPtAsTuple()]:
            self.termPtToContigSegsHash[contigSeg.getLHSMostTermPtAsTuple()].append(contigEntry)
         if not self.termPtToContigSegsHash.get(contigSeg.getRHSMostTermPtAsTuple()):
            self.termPtToContigSegsHash[contigSeg.getRHSMostTermPtAsTuple()] = [contigEntry]
         elif contigEntry not in self.termPtToContigSegsHash[contigSeg.getRHSMostTermPtAsTuple()]:
            self.termPtToContigSegsHash[contigSeg.getRHSMostTermPtAsTuple()].append(contigEntry)

      # generate the map of {contig seg idx : [[LHSNeighbors], [RHSNeighbors]]}
      for ptEntry in self.termPtToContigSegsHash:
         contigSegs = self.termPtToContigSegsHash[ptEntry]
         for i, csHash in enumerate(contigSegs):
            if not self.contigSegIdxToLRHSNeighbors.get(csHash):
               self.contigSegIdxToLRHSNeighbors[csHash] = [None, None]
            contigNeighbors = [contigSegs[j] for j in range(len(contigSegs)) if j != i]
            if np.array_equal(ptEntry, self.contigSegHashToContigSeg.get(csHash).LHSMostTermPt):
               if not self.contigSegIdxToLRHSNeighbors[csHash][0]:
                  self.contigSegIdxToLRHSNeighbors[csHash][0] = contigNeighbors
               else:
                  self.contigSegIdxToLRHSNeighbors[csHash][0].extend(contigNeighbors)
            elif np.array_equal(ptEntry, self.contigSegHashToContigSeg.get(csHash).RHSMostTermPt):
               if not self.contigSegIdxToLRHSNeighbors[csHash][1]:
                  self.contigSegIdxToLRHSNeighbors[csHash][1] = contigNeighbors
               else:
                  self.contigSegIdxToLRHSNeighbors[csHash][1].extend(contigNeighbors)

   # this API generates map of contig seg hash to the length of the 'web'
   # that contig seg belongs to
   #
   #  in this context, a 'web' is defined as a set of contig segs that are connected
   #  to each other, and the length of a web is defined as the total length of the contig segs
   #  that make up this web
   def genMapOfContigSegHashToWebLen(self):
      processedPts = {}
      for NPpt, contigSegsHash in self.termPtToContigSegsHash.items():
         pt = tuple(NPpt)
         if pt not in processedPts:
            stack = [pt]
            webLen = 0
            contigSegsInWeb = []

            while stack:
               currPt = stack.pop()
               for contigSegHash in self.termPtToContigSegsHash.get(currPt):
                  # for the contig seg with unique hash that has a termPt
                  # as currPt, need to get the other termPt
                  if np.array_equal(currPt, self.contigSegHashToContigSeg.get(contigSegHash).startPt):
                     otherPt = self.contigSegHashToContigSeg.get(contigSegHash).getEndPtAsTuple()
                  elif np.array_equal(currPt, self.contigSegHashToContigSeg.get(contigSegHash).endPt):
                     otherPt = self.contigSegHashToContigSeg.get(contigSegHash).getStartPtAsTuple()
                  else:
                     otherPt = None

                  if otherPt and otherPt not in processedPts and \
                     otherPt not in stack:
                     stack.append(otherPt)

                  # now add len of contig seg to len of web
                  webLen += self.contigSegHashToContigSeg.get(contigSegHash).length
                  contigSegsInWeb.append(contigSegHash)

               # now that this currPt has been completely processed (ie. visited all of the contig segs
               #  that this pt touches AND pushed all of its neighboring pts set by contig segs )
               #  push this pt to processedPts
               processedPts[currPt] = True

            # now that this stack is complete - this means that the web that the pt in the initial
            # for loop is part of has been completely crawled - assign each contig seg hash captured
            #  with this total web length
            for webElemIdx, contigSegHash in enumerate(contigSegsInWeb):
               self.contigSegHashToWebLen[contigSegHash] = webLen
               self.contigSegHashToOtherContigSegsHashInWeb[contigSegHash] = contigSegsInWeb[:webElemIdx] + \
                                                                             contigSegsInWeb[webElemIdx+1:]


   def drawLinesToImg(self, img, arrowed):
      line_thickness = 1
      for idx, line in self.lineHashToLineCls.items():
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

            # update the minX / maxX / minY / maxY
            if self.minX > pt[0]:
               self.minX = pt[0]
            if self.maxX < pt[0]:
               self.maxX = pt[0]
            if self.minY > pt[1]:
               self.minY = pt[1]
            if self.maxY < pt[1]:
               self.maxY = pt[1]

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

      # here - generate map of termPtToContigSegsHash (which maps the pt coord
      #        to the contig segs that have this pt as start or end pt - defined
      #        as termPt (short for terminal))
      #        also generate map of contig seg to its neighbors
      #           - this is stored purely using the hash values of key contig seg
      #             and its neighbor values
      self.genMapsOfCSNeighbors()
      # here - generate the contig seg hash to web len
      self.genMapOfContigSegHashToWebLen()
