from ast import literal_eval as make_tuple
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
    g = Graph()
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

def bfs(start, goal):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop()
        if node not in visited:
            visited.add(node)

            if node == goal:
                print "TESTING"
                return
            neighbors = getSucessorStates(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)


pancakes, completedPancakes = parseInput("test_pancakes3.config")
print getSucessorStates(pancakes)
bfs(pancakes,completedPancakes)
