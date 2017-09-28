from Queue import PriorityQueue
from ast import literal_eval as make_tuple
from collections import deque
import math

class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}
        self.coords = {}
        self.battery = {}
        self.targets = []
        self.sensors = []


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


class Problem1:
    def __init__(self):
        self.test = "test"
    def solveProblem(self,filename, algo):
        graph = self.parseInput(filename)
        if graph == None:
            return
        if algo == "unicost":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.ucs(graph,edge))
            print q.get()
        elif algo == "astar":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.astar(graph,edge))
            print q.get()
        elif algo == "greedy":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.greedy(graph,edge))
            print q.get()
        elif algo == "iddfs":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.id_dfs(graph,edge))
            print q.get()
        elif algo == "bfs":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.bfs(graph,edge))
            print q.get()
    def bfs(self,graph, start):
        visited = []
        queue = deque()
        queue.append((start, [start]))
        queueSizes = []
        visitedSizes = []

        while queue:
            node, path = queue.pop()
            if node not in visited:
                visited.append(node)
                visitedSizes.append(len(visited))

                if self.checkFinished(graph,path):
                    # print visited
                    # print "Nodes Created: ", len(graph.edges.keys())
                    # print "Frontier Max Size: ",max(queueSizes)
                    # print "Visited Max Size: ", max(visitedSizes)
                    ret = (len(graph.edges.keys()), (visited, len(graph.edges.keys()), max(queueSizes), max(visitedSizes)))
                    return ret
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        queue.appendleft((neighbor, path + [neighbor]))
                        queueSizes.append(len(queue))
        return "No Solution"


    def ucs(self,graph, v):
        visited = []                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0,v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = len(graph.edges.keys())

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
                finalCost = -self.calculateMaxTime(graph, path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
            else:
                #print graph.sensors
                for edge in graph.neighbors(current_node):
                    if edge not in path:
                        if edge in graph.sensors:
                            q.put((f + graph.get_cost(current_node, edge), edge, path + [edge]))
                        else:
                            q.put((graph.get_cost(current_node, edge), edge, path + [edge]))
                        queueSizes.append(len(q.queue))


        return "No Solution"

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


    def greedy(self,graph, v):
        visited = []                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0,v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = len(graph.edges.keys())

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
                finalCost = -self.calculateMaxTime(graph, path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
            else:
                #print graph.sensors
                for edge in graph.neighbors(current_node):
                    if edge not in path:
                        if edge in graph.sensors:
                            q.put((f + graph.getEuclidianDistance(current_node, edge), edge, path + [edge]))
                        else:
                            q.put((graph.getEuclidianDistance(current_node, edge), edge, path + [edge]))
                        queueSizes.append(len(q.queue))


        return "No Solution"


    def astar(self,graph, v):
        visited = []                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0,v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = len(graph.edges.keys())

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
                finalCost = -self.calculateMaxTime(graph, path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
            else:
                #print graph.sensors
                for edge in graph.neighbors(current_node):
                    if edge not in path:
                        if edge in graph.sensors:
                            q.put((f + graph.get_cost(current_node,edge) +graph.getEuclidianDistance(current_node, edge) , edge, path + [edge]))
                        else:
                            q.put((graph.get_cost(current_node,edge) +graph.getEuclidianDistance(current_node, edge) , edge, path + [edge]))
                        queueSizes.append(len(q.queue))


        return "No Solution"


    def checkFinished(self,graph,monitored):
        for target in graph.targets:
            if target not in monitored:
                return False
        return True
    def calculateDistance(self,p, q):
        q1 = q[0]
        q2 = q[1]
        p1 = p[0]
        p2 = p[1]

        s1 = (q1 - p1)**2
        s2 = (q2 - p2)**2

        return math.sqrt(s1+s2)
    def calculateMaxTime(self,graph,path):
        times = []
        #print path
        for i in range(0,len(path), 2):
            sensor = path[i]
            target = path[i+1]
            battery = graph.battery[sensor]
            distance = graph.getEuclidianDistance(sensor,target)
            times.append( battery //distance)
        #print times
        return min(times)
    def parseInput(self,file_name_string):

        with open(file_name_string, 'r') as f:
            data = f.read()

        dataArr = data.split("\n")

        if not dataArr[0] == "monitor":
            print "Unknown config file"
            return

        sensors = make_tuple(dataArr[1])

        targets = make_tuple(dataArr[2])

        if len(targets) > len(sensors):
            return "No Solution"

        g = Graph()

        for sensor in sensors:
            g.edges[sensor[0]] = []
            g.coords[sensor[0]] = (sensor[1], sensor[2])
            g.battery[sensor[0]] = sensor[3]
            g.sensors.append(sensor[0])

        for target in targets:
            g.edges[target[0]] = []
            g.coords[target[0]] = (target[1], target[2])
            g.targets.append(target[0])

        for sensor in sensors:
            for target in targets:
                g.edges[sensor[0]].append(target[0])
                distance = g.getEuclidianDistance(sensor[0], target[0])
                battery = g.battery[sensor[0]]
                g.weights[sensor[0] + target[0]] = -(battery/distance)
        for target in targets:
            for sensor in sensors:
                g.edges[target[0]].append(sensor[0])

                # weights = []
                # for edge in g.targets:
                #     distance = g.getEuclidianDistance(sensor[0], edge)
                #     battery = g.battery[sensor[0]]
                #     #weights[edge] = -(battery / distance)
                #     weights.append(-(battery/distance))
                # print target[0], " ", sensor[0], " ", weights
                # g.weights[target[0] + sensor[0]] = max(weights)

                # distance = g.getEuclidianDistance(sensor[0], target[0])
                # g.weights[target[0] + sensor[0]] = -distance

                # distance = g.battery[sensor[0]]
                # g.weights[target[0] + sensor[0]] = -distance

                # distance = g.getEuclidianDistance(sensor[0], target[0])
                # battery = g.battery[sensor[0]]
                # g.weights[target[0] + sensor[0]] = -(battery / distance)

                g.weights[target[0] + sensor[0]] = 0

        return g


