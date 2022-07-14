#!/usr/bin/python

import sys
import os

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import cv2 as cv

__prog__ = os.path.abspath(__file__)

import argparse

def main(argv):
   description = """
                   this script takes coordinates of a line and draws them to an img (png)
                   - the format of coordinates is -l xstart ystart xend yend
                 """

   lHelp = "-l xstart ystart xend yend"

   parser = argparse.ArgumentParser(prog=__prog__, usage=description)
   parser.add_argument('-l', '--line', type=int, nargs=4, action='append', help=lHelp)
   parser.add_argument('--arrowed', action='store_true')

   args = parser.parse_args()
   print(args.line)

   yMax = 0
   xMax = 0

   for line in args.line:
      if line[0] > xMax:
         xMax = line[0]
      if line[1] > yMax:
         yMax = line[1]
      if line[2] > xMax:
         xMax = line[2]
      if line[3] > yMax:
         yMax = line[3]

   line_thickness = 1

 #  height = int(float(yMax) * 1.5)
 #  width = int(float(xMax) * 1.5)
   height = 1080
   width = 1920

   imgOut = np.ones([height, width], dtype=np.uint8)*255
   
   imgName = "lineImage"

   for line in args.line:
      if args.arrowed:
         cv.arrowedLine(imgOut, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), line_thickness, tipLength=0.5)
      else:
         cv.line(imgOut, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), line_thickness)

      if(len(imgName) < 30):
         imgName += "_" + str(line[0]) + "_" + str(line[1]) + "_" + str(line[2]) + "_" + str(line[3])

   if args.arrowed:
      imgName += "_arrowed"

   imgName += ".png"

   cv.imwrite(os.path.join(os.path.dirname(__prog__), imgName), imgOut)

if __name__ == '__main__':
   main(sys.argv)
