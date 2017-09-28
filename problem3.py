from ast import literal_eval as make_tuple
from collections import deque



def parseInput(file_name_string):
    with open(file_name_string, 'r') as f:
        data = f.read()

    dataArr = data.split("\n")

    if not dataArr[0].lower() == "pancakes":
        print "Unknown config file"
        return

    pancakes = make_tuple(dataArr[1])
    completePancakes = make_tuple(dataArr[2])

    return (pancakes,completePancakes)


def getSucessorStates(start):
    l = []
    for index in range(1, len(start) + 1):
        # start by grabbing the tuple indices from 0->index
        sublist_front = list(start[:index])
        sublist_back = list(start[index:])

        # then reverse the front sublist we've just grabbed,
        # since we are flipping it, end-over-end
        sublist_front.reverse()

        sublist_flipped = []
        # now, change the sign of every element in sublist front
        for elem in sublist_front:
            elem = elem * -1
            sublist_flipped.append(elem)
        # concat our flipped front with old back
        successor_list = sublist_flipped + sublist_back
        # cast back to a tuple
        successor_tuple = tuple(successor_list)
        # add it to the successor states
        # print len(successor_tuple)
        l.append(successor_tuple)
        # and keep on keepin on.
    return l

def bfs(start, end):
    visited = []
    queue = deque()
    queue.append((start, [start]))
    queueSizes = []
    visitedSizes = []
    graph = set()
    time = 0

    while queue:
        node, path = queue.pop()
        if node not in visited:
            visited.append(node)
            visitedSizes.append(len(visited))
            graph.add(node)
            if node == end:
                ret = (path,time, max(queueSizes), max(visitedSizes))
                return ret
            neighbors = getSucessorStates(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.appendleft((neighbor, path + [neighbor]))
                    queueSizes.append(len(queue))
            time = len(graph)
    return "No Solution"

def id_dfs(start, end):
    import itertools

    def dfs(start, end,depth ):
        visited = []
        stack = [(start, [start])]
        stackSizes = []
        visitedSizes = []

        while stack:
            node, path = stack.pop()
            if node not in visited:
                visited.append(node)
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
                    return path
                neighbors = getSucessorStates(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
                        stackSizes.append(len(stack))
        return "No Solution"

    for depth in itertools.count():
        route = dfs(start, end, depth)
        if route:
            return route

pancakes, completedPancakes = parseInput("test_pancakes3.config")
print id_dfs(pancakes,completedPancakes)
