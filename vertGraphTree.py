#!/usr/bin/python

import numpy as np
import math
import objAdjacencyMap
from linesAPIs import *

# Utility APIs that are not part of the tree class

# This API takes as input a list of lineCls objects and returns
# as output the map of start pt to the combinations of contiguous
# either verts or tree nodes with that start pt
def getStartPtToContigObjsMap(linesList, configsAsNodes=True):
   linesAdjMap = objAdjacencyMapCls()
   for line in linesList:
      linesAdjMap.insertObjIntoAdjMap(redundLine)

   linesVertTree = vertGraphTree.createTreeFromAdjMap(linesAdjMap)
   startPtToContiguousLines = linesVertTree.travTreeGenContigVerts(configsAsNodes)
   return startPtToContigObjs

# This API takes as input the map of start pt to the
# list of combinations of contiguous paths of the vert cluster tree
# with start nodes with that starting point
#
#  INPUT: map of start pt to the list of contig segs with that start pt
#
#  OUTPUT: ordered list of contig segs by pt count from longest to shortest
def getSortedListOfContigClusters(contigsMap, revOrder=True):
   retList = []
   for startPt, contigs in contigsMap.items():
      for contigCluster in contigs:
         retList.append(contigCluster)

   retList = sorted(retList, key=lambda x: len(x), reverse=revOrder)

   return retList

# this is the tree class used to represent the relationship between cluster of vertices
# (representative of multiple lines as contig seg OR a single vertex)
# with other clusters of vertices
#
# Each node is a "cluster" of vertices (this is simply a shorthand as a cluster
# can also refer to a single vertex)
#
# CLUSTER - defined as vertices where all of the vertices:
# 1) are connected
# 2) except for the start / end pt - each vertex is only shared by 2 edges
#
#  The children of a node are clusters that are connected to that node at its end pt
#  The parent of a node is the cluster connected to the node's start pt
#
# the tree can have multiple roots since it can have multiple pts with NO
# incoming edges
# also - leaves can point to root or multiple roots in case of cycles
#
# since the tree maps the graphical adjacency relationship between clusters of
# verts and is not sorted - have map with key and value
#  KEY : start pt of the cluster of verts
#  VALUE : node of the tree
class vertGraphTreeCls:
   def __init__(self):
      self.roots = {}
      self.startPtToNode = {}

   # API to add root
   #  map start pt of root to node
   def addRoot(self, rootNode):
       rootStartPt = rootNode.getStartPtAsTuple()
       if not self.roots.get(rootStartPt):
          self.roots[rootStartPt] = [rootNode]
       elif rootNode not in self.roots[rootStartPt]:
          self.roots[rootStartPt].append(rootNode)

   def setStartPtToNodeMap(self, startPtToNode):
      self.startPtToNode = startPtToNode

   # traverse tree to get combinations of nodes that are connected
   # For example: if given a tree with nodes:
   #  Node 1 with child node 2
   #    Node 2 with child nodes 3, 4
   #      Node 3 with child node 5
   #      Node 4 with child node 6
   #  Would return the node combinations:
   #   5
   #   6
   #   3->5, 3
   #   4->6, 4
   #   2->3->5, 2->4->6, 2->3, 2->4, 2
   #   1->2->3->5, 1->2->4->6, 1->2->3, 1->2->4, 1->2, 1
   #
   #  INPUT: get configs above as the list of verts clusters within each node
   def travTreeGenContigVerts(self, getConfigsAsVerts=True):
      startPtsToContigNodes = {}
      # this map is used to keep track of traversed points for each DFS to avoid cycles
      # run - thus when traversing a node need to set the start pt of the current node to TRUE
      # and after processing current node (meaning after covering all DFS paths from
      # the current node - need to pop that value from the map)
      vertTraverseMap = {}
      nodesWithStartPtProcessed = {}
      # need to define the DFS API used to actually traverse the tree
      def DFSTraversal(curr):
         # traverse child nodes in recursive fashion if that child node has not
         # been visited before
         # NOTE: due to nature of DFS, if a pt is in the startPtsToContigNodes map - this
         # means that all nodes with that start pt has been visited
         # ALSO - need to check if DFS is traversing a cycle as this tree structure connects
         # node to adj node as children even if that node is its ancestor
         vertTraverseMap[curr.getStartPtAsTuple()] = True

         for child in curr.getChildNodes():
            # the property of the tree is that all children of the curr node
            # MUST have the same start pt - since we are doing DFS this means that
            # all of the child nodes with that specific start pts are processed once
            # this for loop terminates
            if nodesWithStartPtProcessed.get(child.getStartPtAsTuple()):
               break
            if not vertTraverseMap.get(child.getStartPtAsTuple()) == True:
               DFSTraversal(child)

         # after having traversed all of the child nodes - now populate the
         # permutations of nodes combinations
         #
         # By nature of DFS - to populate the combinations of contiguous nodes
         # we look for the node combinations with START PT equal to the
         # END PT of the curr node - if none exist this means that the curr node
         # is a leaf
         #
         # first get all the nodes that start with the END PT of the current node
         # if it exists
         contigNodes = startPtsToContigNodes.get(curr.getEndPtAsTuple(), [])
         # now - given the contig nodes that start with the end pt - simply
         #  add the curr node to the beginning to each of contig end nodes combinations
         startPtListOfConfigs = []
         for contigNodesSet in contigNodes:
            contigNodesWithCurr = [curr]
            contigNodesWithCurr.extend(contigNodesSet)
            startPtListOfConfigs.append(contigNodesWithCurr)

         # add the lone entry of the current node
         startPtListOfConfigs.append([curr])

         # add this list of node configurations with this start pt into the
         # startPtsToContigNodes map
         currStartPt = curr.getStartPtAsTuple()
         if not startPtsToContigNodes.get(currStartPt):
            startPtsToContigNodes[currStartPt] = startPtListOfConfigs
         else:
            startPtsToContigNodes[currStartPt].extend(startPtListOfConfigs)

         # now that the current node is completely processed - set its start pt
         # value on the run map to FALSE
         # and set its child nodes to be completed - the structure of the tree
         # is such that all child nodes MUST HAVE SAME START PT - this means that
         # in ONE DFS run ALL of the nodes with that start pt are processed
         # since the end pt of the current node = start pt of all child nodes - set
         # the nodes with start pt processed using the end pt of the curr node
         vertTraverseMap[curr.getStartPtAsTuple()] = False
         nodesWithStartPtProcessed[curr.getEndPtAsTuple()] = True

      # loop thru all of the roots of the tree to do dfs
      for rootStartPt, roots in self.roots.items():
         if not startPtsToContigNodes.get(rootStartPt):
            for root in roots:
               DFSTraversal(root)

      # now that the startPtsToContigNodes are generated - first sort each entry
      # by the configuration of the number of nodes in decreasing order
      for startPt, contigNodes in startPtsToContigNodes.items():
         startPtsToContigNodes[startPt] = sorted(contigNodes, key=lambda x: len(x), reverse=True)

      if getConfigsAsVerts:
         startPtToContigVerts = {}
         for startPt, contigNodes in startPtsToContigNodes.items():
            contigVerts = []
            for entryIdx, contigNodesEntry in enumerate(contigNodes):
               contigVertsEntry = []
               for nodeIdx, node in enumerate(contigNodesEntry):
                  if not contigVertsEntry:
                     contigVertsEntry.extend(node.verts)
                  else:
                     # do check to make sure that the start pt of the curr node
                     # is equal to the last pt that has been pushed into the contig
                     # vertices list
                     if not np.array_equal(node.getStartPt(), contigVertsEntry[-1]):
                        print("ERROR - entry at start pt %s contig nodes entry %s and node idx %s with start pt %s DOES NOT MATCH list last pt %s"\
                              % (startPt, entryIdx, nodeIdx, node.getStartPt(), contigVertsEntry[-1]))
                     else:
                        contigVertsEntry.extend(node.verts[1:])

               # add the contig verts entry into contig verts
               contigVerts.append(contigVertsEntry)
            startPtToContigVerts[startPt] = contigVerts
         return startPtToContigVerts

      return startPtsToContigNodes

class vertGraphNode:
   def __init__(self):
      self.verts = []
      self.children = {}
      self.parent = None

   # add the vertices cluster for the node
   def setVerts(self, verts):
      self.verts = verts

   def getStartPt(self):
      return self.verts[0]

   def getStartPtAsTuple(self):
      return tuple(self.getStartPt())

   def getEndPt(self):
      return self.verts[-1]

   def getEndPtAsTuple(self):
      return tuple(self.getEndPt())

   # the children nodes are stored in certain order
   #  Since the tree is not sorted in any way (it makes no sense to sort the
   #  children by absolute x or absolute y because it depends on the orientation
   #  of the tree)
   #  Thus, will sort children by doing the following:
   #   1) Use the last edge of the vert cluster of the parent node as reference
   #     - the unit vect shall point in direction of end pt to the start pt
   #   2) Use the first edge of the child node vert cluster to determine which position
   #      that child node should be in - using that first node with unit vect pointing
   #      from start pt to end pt
   #   The order of the child nodes shall go in CCW order from the reference unit vect
   #   To calculate this - take dot product of vect 1) and vect 2)
   #   To check whether it is within the 1st half of the CCW block - take the cross product
   #    of child vect to reference vect
   #   and if cross product > 0 k (k being in or out of the page),
   #   it is within the 2st (right) half of CCW - do nothing
   #   If the cross product < 0 k , it is in the 1st (left) half of CCW - in which case
   #   flip that cos(theta) curve alonge the y = -1 line => do to this:
   #   get D = (dot product) - (-1) -> then do -1 - D
   def setChild(self, child):
      refVect = lineCls(self.verts[-1], self.verts[-2]).unitVect
      secVect = lineCls(child.verts[0], child.verts[1]).unitVect
      dot = np.dot(refVect, secVect)
      cross = np.cross(secVect, refVect)
      # if cross product < 0 - this means that the sec vect is on the LHS of the CCW
      #  clock, in which case reflect the dot product along the line y = -1 and
      #  so that the CCW dot product remains in increasing order as we move CCW from
      #  the reference vect
      if cross < 0:
         delta = -1 - dot
         childDot = -1 + delta
      else:
         childDot = dot

      self.children[childDot] = child

   # generator to allow iteration of child nodes
   def getChildNodes(self):
      for dot, childNode in self.children.items():
         yield childNode

# given the adjacency map - create a vertGraphTree - which is an adjacency tree
# that maps clusters of verts to their neighbors - cluster is defined above
def createTreeFromAdjMap(adjMap):
   visitedVerts = {}
   # this maps the start pt of a node to the node itself
   startPtToNode = {}
   # check if it is possible to topologically sort the adjMap to get the order
   # in which to start traversing - if not possible to do topological sort, then
   # return the default topological sorted list of vertices ordered in:
   # 1) increasing X 2) increasing Y if X1==X2
   topoSorted = objAdjacencyMap.topologicalSort(adjMap)
   if not topoSorted:
      topoSorted = sorted(objAdjacencyMap.getAllPtsInAdjMap(adjMap), key=lambda k:(k[0], k[1]))

   # generate the connectivity map because there are cases where a vert has 2 incoming
   # verts but only 1 outgoing edge - to the adjMap that vert would only have 1 outgoing
   # neighbor and be part of the node cluster - however since that vert has 2 incoming edges
   # it should be the start pt of a new node for traversal purposes
   connMap = objAdjacencyMap.getConnectivityMap(adjMap)

   def getUnvisitedNeighbors(pt, map):
      unvisitedNeighbors = []
      for adjPt in map.get(pt, []):
         if not visitedVerts.get(adjPt):
            unvisitedNeighbors.append(adjPt)

      return unvisitedNeighbors

   def genTree(currPt, parent):
      currIter = currPt
      parentIter = parent
      pts = [np.array(parent), np.array(currIter)]
      children = []
      visitedVerts[parent] = True
      visitedVerts[currIter] = True

      # while the pts are part of the cluster (ie. they have only 2 edges)
      # add the clusters to the current node and move the curr iterator to the next
      # neighbor (of which there is only 1)
      newNeighbors = objAdjacencyMap.getAdjPtsWithExclusion(currIter, [parentIter], connMap)
      while(len(newNeighbors) == 1):
         parentIter = currIter
         currIter = newNeighbors[0]
         pts.append(np.array(currIter))
         visitedVerts[currIter] = True
         newNeighbors = objAdjacencyMap.getAdjPtsWithExclusion(currIter, [parentIter], connMap)

      # if the current vert is a start pt of generated subtree(s), then simply create the node with
      # the existing points and take the nodes with that current vert as the start pt as the children
      # of the current node
      if startPtToNode.get(currIter):
         children = startPtToNode.get(currIter)
      else:
         # now look map of nodes already created that has start pt
         #  equal to the visited neighbor - if so - need to create a node with
         #  start pt = curr and next pt = neighbor - then add the existing nodes of that neighbor
         #  to its child list - add this node to current node's child list
         #
         #  need to look thru neighbors and see if pt is already populated in startPtToNode - since
         # this is DFS, any pt in startPtToNode means that all of the subtrees of startPt have been traversed
         for neigh in adjMap.get(currIter, []):
            neighNodes = startPtToNode.get(neigh)
            if neighNodes:
               childNode = vertGraphNode()
               childNode.setVerts([np.array(currIter), np.array(neigh)])
               for neighNode in neighNodes:
                  childNode.setChild(neighNode)
               children.append(childNode)
               childNodeStartPt = childNode.getStartPtAsTuple()
               if startPtToNode.get(childNodeStartPt):
                  startPtToNode[childNodeStartPt].append(childNode)
               else:
                  startPtToNode[childNodeStartPt] = [childNode]

         # if the pt at the current iter has multiple neighbors - it means the
         # pt is the end of the current cluster / node and that the pt is also the start pt
         # for all of the possible children - NOTE this also handles the case where there
         # are 0 children (this is a leaf node)
         for neigh in getUnvisitedNeighbors(currIter, adjMap):
            childNode = genTree(neigh, currIter)
            children.append(childNode)

      # now that we've processed all of the children nodes of the tree
      # add the children to the current node
      node = vertGraphNode()
      node.setVerts(pts)
      for child in children:
         node.setChild(child)
         child.parent = node
      nodeStartPt = node.getStartPtAsTuple()
      if startPtToNode.get(nodeStartPt):
         startPtToNode[nodeStartPt].append(node)
      else:
         startPtToNode[nodeStartPt] = [node]

      return node

   # go thru the unvisited topo sorted pts and use that as the root for generating
   # the tree
   vertTree = vertGraphTreeCls()
   for topoSortedPt in topoSorted:
      if not visitedVerts.get(topoSortedPt):
         for adjPt in adjMap.get(topoSortedPt):
            topoRoot = genTree(adjPt, topoSortedPt)
            vertTree.addRoot(topoRoot)

   vertTree.setStartPtToNodeMap(startPtToNode)

   return vertTree
