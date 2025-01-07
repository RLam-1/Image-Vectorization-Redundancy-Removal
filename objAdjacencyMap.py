#!/usr/bin/python

# this file contains all functionalities of graphs such as traversal utilities
#  This file also contains the objAdjacencyMapCls which is used to analyze the
#  positional relationship between given pts and store them in program-usable
#  data structures such as adjacency maps

import math
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

# This section contains graph traversal utilities
# such as: topological sorting, defining condition functions
# for determining the condition for the direction of edges
# and creating the directed graph

# this condition for setting the direction of the graph is that there is an
# edge going from pt1 -> pt2 iff pt2_X > pt1_X OR pt2_Y >= pt1_Y AND pt2_X == pt1_X
def setDirIncXY(pt1, pt2, undirAdjMap):
   return (pt2[0] > pt1[0] or pt2[0] == pt1[0] and pt2[1] >= pt1[1])

# API to return all of the points in the adjacency map as a list
def getAllPtsInAdjMap(adjMap):
   pts = []
   for pt, adjPts in adjMap.items():
      pts.append(pt)
      for adjPt in adjPts:
         pts.append(adjPt)

   return(list(set(pts)))

# API to generate connectivity map from adjacency map
# connectivity map is a map that removes direction from all connections
def getConnectivityMap(inputMap):
   connMap = {}
   for pt, adjPts in inputMap.items():
      for adjPt in adjPts:
         if connMap.get(pt):
            connMap[pt].append(adjPt)
         else:
            connMap[pt] = [adjPt]
         if connMap.get(adjPt):
            connMap[adjPt].append(pt)
         else:
            connMap[adjPt] = [pt]

   # there may be redundant adj pts so make sure to remove all redundant points
   for pt, connPts in connMap.items():
      connMap[pt] = list(set(connPts))

   return connMap

# API that gets the adjacenct pts of an input EXCLUDING the pts specified
# in the input list
#
# INPUT: reference pt to get the adjacenct verts of
#        list of verts to exclude
#        the adjacency map of the graph
#  OUTPUT : list of adjacent points exlucding the pts in the exclude list
def getAdjPtsWithExclusion(refPt, ptsToExclude, adjMap):
   adjPts = []
   if not adjMap.get(refPt):
      print("reference pt {} not in adjacency map {}".format(refPt, adjMap))
      return adjPts

   excludeMap = {}
   # put the pts to exclude in a map
   for excludePt in ptsToExclude:
      excludeMap[excludePt] = True

   for adjPt in adjMap.get(refPt, []):
      if not excludeMap.get(adjPt):
         adjPts.append(adjPt)

   return adjPts

# topological sorting gives the order in which to start traversing the graph
def topologicalSort(adjMap):
   # define the state of the vertex in the graph
   # and store its status in the vertStatus map
   untraversed = 0
   tmp = 1
   perm = 2
   vertStatus = {}
   cycleFound = False

   ptsInMap = getAllPtsInAdjMap(adjMap)
   for pt in ptsInMap:
      vertStatus[pt] = untraversed

   topoSortedList = []

   # this is to get any untraversed pts when traversing the graph to begin
   # each round of recursive topological sort
   def getFirstUntraversedPt(vertStatus):
      for vert, status in vertStatus.items():
         if status == untraversed:
            return vert
      # if all vertices are traversed - return None
      return None

   def setCycleFound(value):
      global cycleFound
      cycleFound = value

   # internal API to traverse the graph - this API is recursive
   def visitGraph(vert):
      # if vert is already visited - terminate
      #  this also includes the case where a cycle is detected
      if vertStatus.get(vert) == perm or cycleFound:
         return

      # if cycle has been detected - this means that it is impossible to
      # do topological sort
      if vertStatus.get(vert) == tmp:
         setCycleFound(True)
         return

      # this is the 1st time the point is traversed - mark it tmp
      vertStatus[vert] = tmp

      # now loop thru the adjacency map and visit neighbors
      for adjPt in adjMap.get(vert, []):
         visitGraph(adjPt)

      # this is the final time this vert is visited
      #
      # Add this point to the head of the topo list
      vertStatus[vert] = perm
      topoSortedList.insert(0, vert)

   # while there are still untraversed pts in the graph - keep doing DFS
   # recursive traversal to perform
   while True:
      startPt = getFirstUntraversedPt(vertStatus)
      if startPt is None:
         break
      visitGraph(startPt)

   if cycleFound:
      topoSortedList.clear()

   return topoSortedList

########### GRAPH APIs end #####################

# This is the object adjacency map class - stores the spatial relationship
# between objects (ie. which objects are adjacent to other objects)
#

class objAdjacencyMapCls:
   def __init__(self):
      self.undirectedPtsAdjMap = {}
      self.directedPtsAdjMap = {}
      self.termPtToObjsMap = {}
      # this obj to adjacent objs map has format {obj : {start pt: [adjObjs1], end pt : [adjObjs2]}}
      # since it needs to store adjaceny objects from its start and end pt
      self.objToAdjObjsMap = {}
      # this map store the obj to the term pts - this is because the termPts can be
      # joined by different objects such as line, contig seg, bezier curve etc.
      # store as {(termPt1, termPt2) : obj} <- the termPt1 and termPt2 may not be
      # in the actual order of termPt1 and termPt2 but is sorted in increasing order
      # of x-coord first and then y-coord
      self.termPtsToObjMap = {}

   def getDirectedAdjMap(self):
      return self.directedPtsAdjMap

   def getUndirectedAdjMap(self):
      return self.undirectedPtsAdjMap

   # this API adds the obj to only the termPtToObjsMap - this cuts down the speed
   # of populating this adjacency map because it doesn't populate objToAdjObjsMap
   def insertObjIntoPtMap(self, obj):
      retCode = True
      # need to convert numpy array to tuple in order to store pts as key in map
      startPt = obj.getStartPtAsTuple()
      endPt = obj.getEndPtAsTuple()
      if not self.termPtToObjsMap.get(startPt):
         self.termPtToObjsMap[startPt] = [obj]
      elif obj not in self.termPtToObjsMap[startPt]:
         self.termPtToObjsMap[startPt].append(obj)
      else:
         print("Failed to insert obj into pts map with start pt %s" % (startPt))
         retCode = False

      if not self.termPtToObjsMap.get(endPt):
         self.termPtToObjsMap[endPt] = [obj]
      elif obj not in self.termPtToObjsMap[endPt]:
         self.termPtToObjsMap[endPt].append(obj)
      else:
         print("Failed to insert obj into pts map with end pt %s" % (endPt))
         retCode = False

      termPtsKey = tuple(sorted([tuple(startPt), tuple(endPt)], key=lambda k : (k[0], k[1])))
      if not self.termPtsToObjMap.get(termPtsKey):
         self.termPtsToObjMap[termPtsKey] = [obj]
      else:
         self.termPtsToObjMap[termPtsKey].append(obj)

      return retCode

   def insertObjIntoAdjMap(self, obj, isDir=False):
      if self.insertObjIntoPtMap(obj):
         objStartPt = obj.getStartPtAsTuple()
         objEndPt = obj.getEndPtAsTuple()
         # now that the object has been inserted into the pt to objs map - insert
         # into the objToAdjObjsMap
         if not self.objToAdjObjsMap.get(obj):
            self.objToAdjObjsMap[obj] = {}

         # get the startPt of the obj and check which curves also contain that start pt
         # and populate the objToAdjObjsMap (obviously excluding the current object)
         startPtObjs = self.termPtToObjsMap.get(objStartPt)
         if startPtObjs:
            self.objToAdjObjsMap[obj][objStartPt] = [object for object in startPtObjs if obj != object]
            print("Found adjacent objs with same term pt as start pt {} of obj".format(objStartPt))

         # do the same thing for the end pt
         endPtObjs = self.termPtToObjsMap.get(objEndPt)
         if endPtObjs:
            self.objToAdjObjsMap[obj][objEndPt] = [object for object in endPtObjs if obj != object]
            print("Found adjacent objs with same term pt as end pt {} of obj".format(objEndPt))

         # insert the term pts of the obj into self.undirectedPtsAdjMap
         if self.undirectedPtsAdjMap.get(objStartPt):
            self.undirectedPtsAdjMap[objStartPt].append(objEndPt)
         else:
            self.undirectedPtsAdjMap[objStartPt] = [objEndPt]

         if self.undirectedPtsAdjMap.get(objEndPt):
            self.undirectedPtsAdjMap[objEndPt].append(objStartPt)
         else:
            self.undirectedPtsAdjMap[objEndPt] = [objStartPt]

         if isDir:
            if not self.directedPtsAdjMap.get(objStartPt):
               self.directedPtsAdjMap[objStartPt] = [objEndPt]
            else:
               self.directedPtsAdjMap[objStartPt].append(objEndPt)

   # this API creates adjacency map that satisfies the condition as passed in by
   # the condition function (shortened as condFunc)
   #  - for example create directed graph where the direction goes in direction
   #     of increasing x (or increasing y if x is the same)
   def createCondAdjMap(self, condFunc):
      self.directedPtsAdjMap.clear()
      for pt, adjPts in self.undirectedPtsAdjMap.items():
         for adjPt in adjPts:
            if condFunc(pt, adjPt, self.undirectedPtsAdjMap):
               if self.directedPtsAdjMap.get(pt):
                  self.directedPtsAdjMap[pt].append(adjPt)
               else:
                  self.directedPtsAdjMap[pt] = [adjPt]
      return self.directedPtsAdjMap
