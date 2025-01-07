#!/usr/bin/python

import sys
import os
import json
import argparse
import multiprocessing
import time
import re
import traceback

fileName = os.path.abspath(__file__)
__prog__ = os.path.basename(fileName)
__dir__ = os.path.dirname(fileName)

sys.path.append(os.path.dirname(__dir__))

import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
from numpy import linalg as LA
from linesAPIs import *
import objAdjacencyMap
import vertGraphTree

import utilAPIs

__prog__ = os.path.basename(sys.argv[0])

helpString = """
             test vertices cluster tree of generating adjacency relationship
             and creating all permutations of connected vertices

             Takes 1 input - text file with lines in the format
             -
             x1,y1 <- termPt1 of line
             x2,y2 <- termPt2 of line
             -
             can store multiple lines separated by '-' -> each line is bookended
             by '-'
             """
parser = argparse.ArgumentParser(
                   prog = __prog__,
                   description=helpString,
                   add_help=True)
parser.add_argument('-i', action="store", dest="inFile", required=True, \
                        default=None, help="file containing lines info")
parser.add_argument('--dir', action="store_true", dest="isDir", required=False, \
                        default=False, help="flag to indicate that the line passed in is directional")
parser.add_argument('--incXincY', action="store_true", dest="incXincY", required=False, \
                        default=False, help="flag to set the direction of the edges in graph by: 1) increasing X 2) increasing Y")

if __name__ == '__main__':
   args = parser.parse_args()
   ptToIdx = {}
   lines = utilAPIs.readLineObjFromTxtFile(args.inFile, ptToIdx)

   # create adjacency map
   #
   # if the direction of the line (from termPt1 to termPt2) is determined to be
   # the direction of the graph, set the isDir flag to TRUE
   adjMapObj = objAdjacencyMap.objAdjacencyMapCls()
   for line in lines:
      adjMapObj.insertObjIntoAdjMap(line, args.isDir)

   # if want to set the direction of the graph so that the direction of the edges
   #  is such that it goes in order of increasing X, and if X is the same, then increasing Y
   if args.incXincY:
      adjMapObj.createCondAdjMap(objAdjacencyMap.setDirIncXY)

   # now with the DG (directed graph) create the vert tree
   print("adjMap is {}".format(adjMapObj.getDirectedAdjMap()))
   vertTree = vertGraphTree.createTreeFromAdjMap(adjMapObj.getDirectedAdjMap())
   print(vertTree.startPtToNode)
   startPtToContigVerts = vertTree.travTreeGenContigVerts()
   for startPt, contigVerts in startPtToContigVerts.items():
      print("Paths with start pt %s : %s" % (startPt, contigVerts))
   contigVertsList = vertGraphTree.getSortedListOfContigClusters(startPtToContigVerts)
   for idx, contigVert in enumerate(contigVertsList):
      print("path %s : %s" % (idx, contigVert))
   if ptToIdx:
      print("pts are labelled with idxs")
      for idx, contigVert in enumerate(contigVertsList):
         vertIdxList = []
         for vert in contigVert:
            vertIdxList.append(ptToIdx.get(tuple(vert)))
         print("index labelled pts path {} : {}".format(idx, vertIdxList))
