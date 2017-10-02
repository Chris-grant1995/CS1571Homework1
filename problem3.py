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
