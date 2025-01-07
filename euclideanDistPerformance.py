#!/usr/bin/python

import numpy as np
from linesAPIs import *
from numpy import linalg as LA
import time

line1 = lineCls(np.array([-5, -95]), np.array([99, 1026]))
pt = np.array([5, -301])

# first method - doing matrix calculations
begin1 = time.time()
A1 = np.array([[line1.unitVect[0], 0, -1, 0], \
               [line1.unitVect[1], 0, 0, -1], \
               [0, line1.unitVect[1], 1, 0], \
               [0, -line1.unitVect[0], 0, 1]])

B1 = np.array([-line1.termPt1[0], -line1.termPt1[1], pt[0], pt[1]])

A, B, Cx, Cy = np.linalg.solve(A1, B1)

euclideanVect1 = B * np.array([line1.unitVect[1], -line1.unitVect[0]])
end1 = time.time()
print("method 1 mat calc euclideanVect1 = %s time = %s" % (euclideanVect1, end1-begin1))

# second method - using vector calc
begin2 = time.time()
line1TermPt1_pt = lineCls(line1.termPt1, pt)

cosTheta = np.dot(line1TermPt1_pt.unitVect, line1.unitVect)
line1Parallel = lineCls(line1.termPt1, line1TermPt1_pt.lineLength*cosTheta*line1.unitVect + line1.termPt1)

euclideanLine = lineCls(line1Parallel.termPt2, pt)
euclideanVect2 = euclideanLine.lineLength * euclideanLine.unitVect
end2 = time.time()
print("method 2 mat calc euclideanVect2 = %s time = %s" % (euclideanVect2, end2-begin2))
