import sys
from problem2 import Problem2


def main():
    if len(sys.argv) != 3:
        print "Invalid Commandline Args"
        return

    with open(sys.argv[1], 'r') as f:
        data = f.read()

    dataArr = data.split("\n")

    if not dataArr[0] == "aggregation" and not dataArr[0] == "monitor":
        print "Unknown config file"
        return

    if dataArr[0] == "aggregation":
        p = Problem2()
    else:
        #p = Problem1()
        p = None
        print "Problem 1 Coming Soon"

    p.solveProblem(sys.argv[1], sys.argv[2])

main()