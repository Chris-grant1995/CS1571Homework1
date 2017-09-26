from Queue import PriorityQueue
from ast import literal_eval as make_tuple
from collections import deque
import math

class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}
        self.coords = {}

    def neighbors(self, node):
        return self.edges[node]
    def getCoords(self, node):
        return self.coords[node]

    def get_cost(self, from_node, to_node):
        return self.weights[(from_node + to_node)]

    def getEuclidianDistance(self, node1, node2):
        q = self.getCoords(node1)
        p = self.getCoords(node2)

        q1 = q[0]
        q2 = q[1]
        p1 = p[0]
        p2 = p[1]

        s1 = (q1 - p1) ** 2
        s2 = (q2 - p2) ** 2
        return math.sqrt(s1 + s2)
class Problem2:
    def __init__(self):
        self.test = "test"
    def solveProblem(self,filename, algo):
        graph = self.parseInput(filename)
        if graph == None:
            return
        if algo == "unicost":
            q = PriorityQueue()
            for edge in graph.edges:
                print("Hi From ", edge)
                q.put(self.ucs(graph,edge))
                print q.get()
                return
            print q.get()
        elif algo == "astar":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.astar(graph,edge))
            print q.get()
        elif algo == "greedy":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.greedy(graph,edge))
            print q.get()
        elif algo == "iddfs":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.id_dfs(graph,edge))
            print q.get()
        elif algo == "bfs":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.bfs(graph,edge))
            print q.get()


    def bfs(self,graph, start):
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

                if self.checkFinished(graph,visited):
                    # print visited
                    # print "Nodes Created: ", len(graph.edges.keys())
                    # print "Frontier Max Size: ",max(queueSizes)
                    # print "Visited Max Size: ", max(visitedSizes)
                    ret = (len(graph.edges.keys()), (visited, len(graph.edges.keys()), max(queueSizes), max(visitedSizes)))
                    return ret
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        queue.appendleft(neighbor)
                        queueSizes.append(len(queue))
        return "No Solution"


    def ucs(self,graph, v):
        visited = []                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = len(graph.edges.keys())

        # Because we don't care if we've visited a node before in our algorithm
        # if a node has no edges, the loop will run forever, so running BFS
        # before will tell us if there are any unreachable nodes, and if there are
        # we already know that there is no solution
        v = self.bfs(graph,v)
        if v == "No Solution":
            return "No Solution"
        while not q.empty():             # while the queue is nonempty
            #print q.queue
            f, current_node, path = q.get()
            visited.append(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))
            time+=1

            #print path

            if self.checkFinished(graph,path):
                finalCost = self.calculatePath(graph,path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
            else:
                for edge in graph.neighbors(current_node):
                    #if edge not in visited:
                    q.put((f + graph.get_cost(current_node,edge), edge, path + [edge]))
                    queueSizes.append(len(q.queue))


    def id_dfs(self,graph,start):
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
                    if self.checkFinished(graph,visited):
                        # print visited
                        # print "Nodes Created: ", len(graph.edges.keys())
                        # print "Frontier Max Size: ", max(stackSizes)
                        # print "Visited Max Size: ", max(visitedSizes)
                        ret = (len(graph.edges.keys()), (visited,len(graph.edges.keys()), max(stackSizes),max(visitedSizes) ))
                        return ret
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            stack.append(neighbor)
                            stackSizes.append(len(stack))
            return "No Solution"

        for depth in itertools.count():
            route = dfs(graph,start, depth)
            if route:
                return route
    def greedy(self,graph, v):
        visited = []                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []

        # Because we don't care if we've visited a node before in our algorithm
        # if a node has no edges, the loop will run forever, so running BFS
        # before will tell us if there are any unreachable nodes, and if there are
        # we already know that there is no solution
        v = self.bfs(graph, v)
        if v == "No Solution":
            return "No Solution"

        while not q.empty():             # while the queue is nonempty
            f, current_node, path = q.get()
            visited.append(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))

            #print path

            if self.checkFinished(graph,path):
                finalCost = self.calculatePath(graph,path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes)))
            else:
                for edge in graph.neighbors(current_node):
                    #if edge not in visited:
                    q.put((f + graph.getEuclidianDistance(current_node,edge), edge, path + [edge]))
                    queueSizes.append(len(q.queue))
    def astar(self,graph, v):
        visited = []                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []


        # Because we don't care if we've visited a node before in our algorithm
        # if a node has no edges, the loop will run forever, so running BFS
        # before will tell us if there are any unreachable nodes, and if there are
        # we already know that there is no solution
        v = self.bfs(graph, v)
        if v == "No Solution":
            return "No Solution"


        while not q.empty():             # while the queue is nonempty
            f, current_node, path = q.get()
            visited.append(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))

            # print path

            if self.checkFinished(graph,path):
                finalCost = self.calculatePath(graph,path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes)))
            else:
                for edge in graph.neighbors(current_node):
                    #if edge not in visited:
                    q.put((f + graph.get_cost(current_node,edge) + graph.getEuclidianDistance(current_node,edge), edge, path + [edge]))
                    queueSizes.append(len(q.queue))

    def calculatePath(self,graph, path):
        cost = 0
        for i in range(0,len(path)-1):
            cost += graph.get_cost(path[i], path[i+1])
        return cost
    def checkFinished(self,graph,visited):
        for key in graph.edges.keys():
            if key not in visited:
                return False
        return True
    def parseInput(self,file_name_string):
        """
        Parse the input data and fill the class variables in init
        :param file_name_string:
        :return: void
        """

        with open(file_name_string, 'r') as f:
            data = f.read()

        dataArr = data.split("\n")

        if not dataArr[0].lower() == "aggregation":
            print "Unknown config file"
            return

        locations = make_tuple(dataArr[1])



        g = Graph()
        for location in locations:
            g.edges[location[0]] = []
            g.coords[location[0]] = (location[1], location[2])

        for elem in xrange(2, len(dataArr)):
            action = dataArr[elem]
            action = make_tuple(action)
            #print action
            if action[0] not in g.edges.keys():
                g.edges[action[0]] = []
            g.edges[action[0]].append(action[1])
            if action[1] not in g.edges.keys():
                g.edges[action[1]] = []
            g.edges[action[1]].append(action[0])
            g.weights[action[0]+action[1]] = action[2]
            g.weights[action[1] + action[0]] = action[2]
        return g
