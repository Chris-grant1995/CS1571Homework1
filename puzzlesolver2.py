from Queue import PriorityQueue
from ast import literal_eval as make_tuple
from collections import deque
import math




class Problem1:
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
            finalTuple = q.get()
            # print finalTuple
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][4]
            print "Uniform Cost Search Result for ", filename
            print "Assignments: "
            for i in range(0, len(path), 2):
                sensor = path[i]
                target = path[i + 1]
                print sensor, " -> ", target
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", -cost
        elif algo == "astar":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.astar(graph,edge))
            finalTuple = q.get()
            # print finalTuple
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][4]
            print "AStar Result for ", filename
            print "Assignments: "
            for i in range(0, len(path), 2):
                sensor = path[i]
                target = path[i + 1]
                print sensor, " -> ", target
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", -cost
        elif algo == "greedy":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.greedy(graph,edge))
            finalTuple = q.get()
            # print finalTuple
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][4]
            print "Greedy Result for ", filename
            print "Assignments: "
            for i in range(0, len(path), 2):
                sensor = path[i]
                target = path[i + 1]
                print sensor, " -> ", target
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", -cost
        elif algo == "iddfs":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.id_dfs(graph,edge))
            finalTuple = q.get()
            # print finalTuple
            expansion = finalTuple[0]
            cost = finalTuple[1][4]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][2]
            print "IDDFS Result for ", filename
            print "Assignments: "
            for i in range(0, len(path), 2):
                sensor = path[i]
                target = path[i + 1]
                print sensor, " -> ", target
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Max Expansion: ", expansion
            print "Cost: ", cost
        elif algo == "bfs":
            q = PriorityQueue()
            for edge in graph.sensors:
                q.put(self.bfs(graph,edge))
            finalTuple = q.get()
            path = finalTuple[0]
            queueSize = finalTuple[3]
            visitedSize = finalTuple[2]
            time = finalTuple[1]
            print "BFS Result for ", filename
            print "Assignments: "
            for p in path:
                print p
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
    def bfs(self,graph,start):
        visited = []
        queue = deque()
        queue.append(start)
        queueSizes = []
        visitedSizes = []
        time = 0


        while queue:
            node = queue.pop()
            if node not in visited:
                time +=1
                visited.append(node)
                visitedSizes.append(len(visited))

                if self.checkFinished(graph,visited):
                    return (visited, time, max(visitedSizes), max(queueSizes))
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        queue.appendleft(neighbor)
                        queueSizes.append(len(queue))
        return "No Solution"


    def ucs(self,graph, v):
        visited = set()                  # set of visited nodes
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
            visited.add(current_node)    # mark node visited on expansion,
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
            time = 0

            while stack:
                node = stack.pop()
                if node not in visited:
                    time +=1
                    visited.append(node)
                    visitedSizes.append(len(visited))
                    if len(visited) > depth:
                        return
                    if self.checkFinished(graph,visited):
                        # print visited
                        # print "Nodes Created: ", len(graph.edges.keys())
                        # print "Frontier Max Size: ", max(stackSizes)
                        # print "Visited Max Size: ", max(visitedSizes)
                        # return visited
                        cost = self.calculateMaxTime(graph,visited)
                        ret = (visited,time,max(stackSizes),max(visitedSizes),cost)
                        return ret
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            stack.append(neighbor)
                            stackSizes.append(len(stack))

        for depth in itertools.count():
            route = dfs(graph,start, depth)
            if route:
                return (depth,route)


    def greedy(self,graph, v):
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0,v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

        while not q.empty():             # while the queue is nonempty
            #print q.queue
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
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
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0,v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

        while not q.empty():             # while the queue is nonempty
            #print q.queue
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
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

        g = self.Graph()

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


from Queue import PriorityQueue
from ast import literal_eval as make_tuple
from collections import deque
import math


class Problem2:
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
    def __init__(self):
        self.test = "test"
    def solveProblem(self,filename, algo):
        graph = self.parseInput(filename)
        if graph == None:
            return
        if algo == "unicost":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.ucs(graph,edge))
            finalTuple = q.get()
            if finalTuple == "No Solution":
                print "Uniform Cost Search Result for ", filename
                print finalTuple
                return
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][4]

            print "Uniform Cost Search Result for ", filename
            print "Path:"
            for item in path:
                print item
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", cost
        elif algo == "astar":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.astar(graph,edge))
            finalTuple = q.get()
            if finalTuple == "No Solution":
                print "AStar Result for ", filename
                print finalTuple
                return
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][4]

            print "AStar Result for ", filename
            print "Path:"
            for item in path:
                print item
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", cost
        elif algo == "greedy":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.greedy(graph,edge))
            finalTuple = q.get()
            if finalTuple == "No Solution":
                print "Greedy Result for ", filename
                print finalTuple
                return
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][4]

            print "Greedy Result for ", filename
            print "Path:"
            for item in path:
                print item
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", cost
        elif algo == "iddfs":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.id_dfs(graph,edge))
            finalTuple = q.get()
            if finalTuple == "No Solution":
                print "iddfs Result for ", filename
                print finalTuple
                return
            cost = finalTuple[0]
            path = finalTuple[1][0]
            queueSize = finalTuple[1][2]
            visitedSize = finalTuple[1][3]
            time = finalTuple[1][1]
            print "iddfs Result for ", filename
            print "Path:"
            for item in path:
                print item
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Max Expansion: ", cost
        elif algo == "bfs":
            q = PriorityQueue()
            for edge in graph.edges:
                q.put(self.bfs(graph,edge))
            finalTuple = q.get()
            if finalTuple == "No Solution":
                print "BFS Result for ", filename
                print finalTuple
                return
            path = finalTuple[0]
            queueSize = finalTuple[3]
            visitedSize = finalTuple[2]
            time = finalTuple[1]
            print "BFS Result for ", filename
            print "Path:"
            for item in path:
                print item
            print "Time: ", time
            print "Max Frontier Size: ", queueSize
            print "Max Visited Size: ", visitedSize


    def bfs(self,graph,start):
        visited = []
        queue = deque()
        queue.append(start)
        queueSizes = []
        visitedSizes = []
        time = 0


        while queue:
            node = queue.pop()
            if node not in visited:
                time +=1
                visited.append(node)
                visitedSizes.append(len(visited))

                if self.checkFinished(graph,visited):
                    return (visited, time, max(visitedSizes), max(queueSizes))
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        queue.appendleft(neighbor)
                        queueSizes.append(len(queue))
        return "No Solution"


    def ucs(self,graph, v):
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

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
            visited.add(current_node)    # mark node visited on expansion,
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
            time = 0
            while stack:
                node = stack.pop()
                if node not in visited:
                    time +=1
                    visited.append(node)
                    visitedSizes.append(len(visited))
                    if len(visited) > depth:
                        return
                    if self.checkFinished(graph,visited):
                        # print visited
                        # print "Nodes Created: ", len(graph.edges.keys())
                        # print "Frontier Max Size: ", max(stackSizes)
                        # print "Visited Max Size: ", max(visitedSizes)
                        ret = (visited,time, max(stackSizes),max(visitedSizes) )
                        return ret
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            stack.append(neighbor)
                            stackSizes.append(len(stack))
            return "No Solution"

        for depth in itertools.count():
            route = dfs(graph,start, depth)
            if route:
                return (depth,route)
    def greedy(self,graph, v):
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

        # Because we don't care if we've visited a node before in our algorithm
        # if a node has no edges, the loop will run forever, so running BFS
        # before will tell us if there are any unreachable nodes, and if there are
        # we already know that there is no solution
        v = self.bfs(graph, v)
        if v == "No Solution":
            return "No Solution"

        while not q.empty():             # while the queue is nonempty
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))
            time +=1
            #print path

            if self.checkFinished(graph,path):
                finalCost = self.calculatePath(graph,path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
            else:
                for edge in graph.neighbors(current_node):
                    #if edge not in visited:
                    q.put((f + graph.getEuclidianDistance(current_node,edge), edge, path + [edge]))
                    queueSizes.append(len(q.queue))
    def astar(self,graph, v):
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []

        time = 0


        # Because we don't care if we've visited a node before in our algorithm
        # if a node has no edges, the loop will run forever, so running BFS
        # before will tell us if there are any unreachable nodes, and if there are
        # we already know that there is no solution
        v = self.bfs(graph, v)
        if v == "No Solution":
            return "No Solution"


        while not q.empty():             # while the queue is nonempty
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            time +=1
            visitedSizes.append(len(visited))

            # print path

            if self.checkFinished(graph,path):
                finalCost = self.calculatePath(graph,path)
                return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
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
        # print graph.edges.keys()
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



        g = self.Graph()
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

from ast import literal_eval as make_tuple
from collections import deque
from Queue import PriorityQueue

class Problem3:
    def __init__(self):
        self.test = "test"

    def solveProblem(self, filename, algo):
        start, end = self.parseInput(filename)
        if start == None:
            return
        if algo == "unicost":
            ret = self.ucs(start,end)

            path = ret[0]
            time = ret[1]
            queueSize = ret[2]
            visitedSize = ret[3]
            cost = ret[4]

            print "Uniform Cost Search Result for ", filename
            print "Path:"
            for p in path:
                print p
            print "Time: ", time
            print "Max Queue Size: ",queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", cost
        elif algo == "astar":
            ret = self.astar(start, end)

            path = ret[0]
            time = ret[1]
            queueSize = ret[2]
            visitedSize = ret[3]
            cost = ret[4]

            print "AStar Result for ", filename
            print "Path:"
            for p in path:
                print p
            print "Time: ", time
            print "Max Queue Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", cost
        elif algo == "greedy":
            ret = self.greedy(start, end)
            path = ret[0]
            time = ret[1]
            queueSize = ret[2]
            visitedSize = ret[3]
            cost = ret[4]

            print "Greedy Result for ", filename
            print "Path:"
            for p in path:
                print p
            print "Time: ", time
            print "Max Queue Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Cost: ", cost

        elif algo == "iddfs":
            ret = self.id_dfs(start, end)
            print ret

            path = ret[1][0]
            time = ret[1][1]
            queueSize = ret[1][2]
            visitedSize = ret[1][3]
            cost = ret[0]

            print "IDDFS Result for ", filename
            print "Path:"
            for p in path:
                print p
            print "Time: ", time
            print "Max Queue Size: ", queueSize
            print "Max Visited Size: ", visitedSize
            print "Max Expansion: ", cost
        elif algo == "bfs":
            ret = self.bfs(start, end)
            path = ret[0]
            time = ret[1]
            queueSize = ret[2]
            visitedSize = ret[3]

            print "BFS Result for ", filename
            print "Path:"
            for p in path:
                print p
            print "Time: ", time
            print "Max Queue Size: ", queueSize
            print "Max Visited Size: ", visitedSize

    def parseInput(self,file_name_string):
        with open(file_name_string, 'r') as f:
            data = f.read()

        dataArr = data.split("\n")

        if not dataArr[0].lower() == "pancakes":
            print "Unknown config file"
            return (None, None)

        pancakes = make_tuple(dataArr[1])
        completePancakes = make_tuple(dataArr[2])

        return (pancakes,completePancakes)


    def getSucessorStates(self,start):
        l = []
        for index in range(1, len(start) + 1):
            sublist_front = list(start[:index])
            sublist_back = list(start[index:])

            sublist_front.reverse()

            sublist_flipped = []
            for elem in sublist_front:
                elem = elem * -1
                sublist_flipped.append(elem)
            successor_list = sublist_flipped + sublist_back
            successor_tuple = tuple(successor_list)
            l.append(successor_tuple)
        return l

    def bfs(self,start, end):
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        queueSizes = []
        visitedSizes = []
        graph = set()
        time = 0

        while queue:
            node, path = queue.pop()
            if node not in visited:
                visited.add(node)
                visitedSizes.append(len(visited))
                graph.add(node)
                if node == end:
                    ret = (path,time, max(queueSizes), max(visitedSizes))
                    return ret
                neighbors = self.getSucessorStates(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.appendleft((neighbor, path + [neighbor]))
                        queueSizes.append(len(queue))
                time = len(graph)
        return "No Solution"

    def id_dfs(self,start, end):
        import itertools

        def dfs(start, end,depth ):
            visited = set()
            stack = [(start, [start])]
            stackSizes = []
            visitedSizes = []

            time = 0

            while stack:
                node, path = stack.pop()
                time +=1
                if node not in visited:
                    visited.add(node)
                    visitedSizes.append(len(visited))
                    if len(visited) > depth:
                        return
                    if node == end:
                        # print visited
                        # print "Nodes Created: ", len(graph.edges.keys())
                        # print "Frontier Max Size: ", max(stackSizes)
                        # print "Visited Max Size: ", max(visitedSizes)
                        # ret = (len(graph.edges.keys()), (visited,len(graph.edges.keys()), max(stackSizes),max(visitedSizes) ))
                        # return ret
                        ret = (path,time, max(stackSizes),max(visitedSizes))

                        return ret
                    neighbors = self.getSucessorStates(node)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append((neighbor, path + [neighbor]))
                            stackSizes.append(len(stack))
            return "No Solution"

        for depth in itertools.count():
            route = dfs(start, end, depth)
            if route:
                return (depth,route)

    def ucs(self,v, end):
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

        while not q.empty():             # while the queue is nonempty
            #print q.queue
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))
            time+=1

            #print path

            if current_node == end:
                # return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
                return (path,time,max(queueSizes), max(visitedSizes), f )
            else:
                neighbors = self.getSucessorStates(current_node)
                for edge in neighbors:
                    if edge not in visited:
                        q.put(( 1, edge, path + [edge]))
                        queueSizes.append(len(q.queue))

    def greedy(self,v, end):
        visited = set()                  # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

        while not q.empty():             # while the queue is nonempty
            #print q.queue
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))
            time+=1

            #print path

            if current_node == end:
                # return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
                return (path,time,max(queueSizes), max(visitedSizes),f )
            else:
                neighbors = self.getSucessorStates(current_node)
                for edge in neighbors:
                    if edge not in visited:
                        h = self.gapHeuristic(edge)
                        q.put((h, edge, path + [edge]))
                        queueSizes.append(len(q.queue))
    def astar(self,v, end):
        visited = set()                 # set of visited nodes
        q = PriorityQueue()        # we store vertices in the (priority) queue as tuples
                                         # (f, n, path), with
                                         # f: the cumulative cost,
                                         # n: the current node,
                                         # path: the path that led to the expansion of the current node
        q.put((0, v, [v]))               # add the starting node, this has zero *cumulative* cost
                                         # and it's path contains only itself.

        queueSizes = []
        visitedSizes = []
        time = 0

        while not q.empty():             # while the queue is nonempty
            #print q.queue
            f, current_node, path = q.get()
            visited.add(current_node)    # mark node visited on expansion,
                                         # only now we know we are on the cheapest path to
                                         # the current node.
            visitedSizes.append(len(visited))
            time+=1

            #print path

            if current_node == end:
                # return (finalCost,(path,finalCost,max(queueSizes), max(visitedSizes), time))
                return (path,time,max(queueSizes), max(visitedSizes), f)
            else:
                neighbors = self.getSucessorStates(current_node)
                for edge in neighbors:
                    if edge not in visited:
                        h = self.gapHeuristic(edge)
                        q.put((h + 1, edge, path + [edge]))
                        queueSizes.append(len(q.queue))

    def gapHeuristic(self,successor_state):
        heuristic = 0
        for index in range(0, len(successor_state) - 1):
            if (successor_state[index] - successor_state[index + 1]) > 1:
                heuristic += 1
            elif successor_state[index] * successor_state[index + 1] < 0:
                heuristic += 1
            else:
                continue

        return heuristic
import sys
# from problem2 import Problem2
# from problem1 import Problem1
# from problem3 import Problem3


def main():
    if len(sys.argv) != 3:
        print "Invalid Commandline Args"
        return

    with open(sys.argv[1], 'r') as f:
        data = f.read()

    dataArr = data.split("\n")

    if not dataArr[0].lower() == "aggregation" and not dataArr[0].lower() == "monitor" and not dataArr[0].lower() == "pancakes":
        print "Unknown config file"
        return

    if dataArr[0].lower() == "aggregation":
        p = Problem2()
    elif dataArr[0].lower() == "pancakes":
        p = Problem3()
    else:
        p = Problem1()

    if sys.argv[2] == "all":
        algos = ["bfs","iddfs", "unicost", "greedy", "astar"]
        for algo in algos:
            p.solveProblem(sys.argv[1], algo)
        return

    p.solveProblem(sys.argv[1], sys.argv[2])

main()
# 1166 Lines, because that makes debugging really easy