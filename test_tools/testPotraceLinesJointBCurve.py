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

import utilAPIs

__prog__ = os.path.basename(sys.argv[0])

helpString = """
             test potrace algorithm of detecting if line joints form bezier curve
             REMINDER - the end points of the joint biMinus1, bi are midpts of the
             lineCls lines in the image

             Takes 1 input - text file with joint info in format:
             -
             biMinus1 : 0,-44
             ai : 0,0
             bi : -17,35
             shift : -55,38 <--- shift to be applied
             theta : 40 <--- rotation to be applied
             -
             can store multiple joints separated by '-' -> each joint is bookended
             by '-'
             """
parser = argparse.ArgumentParser(
                   prog = __prog__,
                   description=helpString,
                   add_help=True)
parser.add_argument('-i', action="store", dest="inFile", required=True, \
                        default=None, help="file containing joint info")

if __name__ == '__main__':
   args = parser.parse_args()
   biMinus1_ai_bi_alpha = utilAPIs.getJointsPtsFromTestFile(args.inFile)
   count = 1
   for biMinus1, ai, bi, alpha in biMinus1_ai_bi_alpha:
      try:
         # NOTE - for now the points biMinus1, bi are midpts of the lines that make up
         #        the image. Thus, the end pts of the joints biMinus1 and bi are
         #        the midpt of the input lines to the API givenJointReturnAlphaAndLiSimpleVer
         #
         #   Thus - need to double the lines ai_biMinus1 and ai_bi
         line1EndPt = biMinus1
         line2EndPt = bi
         line1 = lineCls(line1EndPt, ai)
         line2 = lineCls(ai, line2EndPt)
         alpha, Li = givenJointReturnAlphaAndLiSimpleVer(line1, line2)
         print("joint %s =====> alpha = %s" % (count, alpha))
         print("joint %s =====> Li: " % (count))
         Li.displayLineInfo()
         count += 1

      except Exception as e:
         print("Error calculating joint alpha and Li - %s" % (e))
         print(traceback.format_exc())
