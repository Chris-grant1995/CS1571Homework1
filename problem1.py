from Queue import PriorityQueue
from ast import literal_eval as make_tuple
from collections import deque
import math

class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

    def neighbors(self, node):
        return self.edges[node]

    def get_cost(self, from_node, to_node):
        return self.weights[(from_node + to_node)]



def bfs(graph, start):
    visited = []
    queue = deque()
    queue.append(start)
    queueSizes = []
    visitedSizes = []

    while queue:
        node = queue.pop()
        if node not in visited:
            visited.append(node)
            visitedSizes.append(len(visited))

            if checkFinished(graph,visited):
                print visited
                print "Nodes Created: ", len(graph.edges.keys())
                print "Frontier Max Size: ",max(queueSizes)
                print "Visited Max Size: ", max(visitedSizes)
                return
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    queue.appendleft(neighbor)
                    queueSizes.append(len(queue))


def ucs(graph, start):
    visited = []
    queue = PriorityQueue()
    queue.put((0,start))
    queueSizes = []
    visitedSizes = []
    finalCost = 0



    while queue:
        print queue.queue
        cost, node = queue.get()

        visited.append(node)
        visitedSizes.append(len(visited))
        finalCost+=cost

        if checkFinished(graph,visited):
            print visited
            print "Nodes Created: ", len(graph.edges.keys())
            print "Frontier Max Size: ",max(queueSizes)
            print "Visited Max Size: ", max(visitedSizes)
            print "Final Cost: ", finalCost
            return

        for i in graph.neighbors(node):
            total_cost = cost + graph.get_cost(node, i)
            queue.put((total_cost, i))

            queueSizes.append(len(list(queue.queue)))

def ucs2(graph, start):
    visited = []
    queue = PriorityQueue()
    queue.put((0,start))
    queueSizes = []
    visitedSizes = []



    while queue:
        print queue.queue
        cost, node = queue.get()

        visited.append(node)
        visitedSizes.append(len(visited))


        if checkFinished(graph,visited):
            print visited
            print "Nodes Created: ", len(graph.edges.keys())
            print "Frontier Max Size: ",max(queueSizes)
            print "Visited Max Size: ", max(visitedSizes)
            print "Total Cost: ", total_cost
            return

        for i in graph.neighbors(node):
            total_cost = cost + graph.get_cost(node, i)
            queue.put((total_cost, i))

            queueSizes.append(len(list(queue.queue)))

def id_dfs(graph,start):
    import itertools

    def dfs(graph, start,depth ):
        visited = []
        stack = [start]
        stackSizes = []
        visitedSizes = []

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                visitedSizes.append(len(visited))
                if len(visited) > depth:
                    return
                if checkFinished(graph,visited):
                    print visited
                    print "Nodes Created: ", len(graph.edges.keys())
                    print "Frontier Max Size: ", max(stackSizes)
                    print "Visited Max Size: ", max(visitedSizes)
                    return visited
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        stackSizes.append(len(stack))

    for depth in itertools.count():
        route = dfs(graph,start, depth)
        if route:
            return


def checkFinished(graph,visited):
    return
def calculateDistance(p, q):
    q1 = q[0]
    q2 = q[1]
    p1 = p[0]
    p2 = p[1]

    s1 = (q1 - p1)**2
    s2 = (q2 - p2)**2

    return math.sqrt(s1+s2)
def parseInput(file_name_string):
    """
    Parse the input data and fill the class variables in init
    :param file_name_string:
    :return: void
    """

    with open(file_name_string, 'r') as f:
        data = f.read()

    dataArr = data.split("\n")

    if not dataArr[0] == "monitor":
        print "Unknown config file"
        return

    sensors = make_tuple(dataArr[1])

    targets = make_tuple(dataArr[2])
    print sensors
    print targets
parseInput("monitor.config")

print calculateDistance((2,-1),(-2,2))

#ucs(graph,"N_1")