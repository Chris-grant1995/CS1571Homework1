import sys
from problem2 import Problem2
from problem1 import Problem1
from problem3 import  Problem3


def main():
    if len(sys.argv) != 3:
        print "Invalid Commandline Args"
        return

    with open(sys.argv[1], 'r') as f:
        data = f.read()

    dataArr = data.split("\n")

    if not dataArr[0].lower() == "aggregation" and not dataArr[0].lower() == "monitor" and not dataArr[0].lower() == "pancakes" :
        print "Unknown config file"
        return

    if dataArr[0].lower() == "aggregation":
        p = Problem2()
    elif dataArr[0].lower() == "pancakes":
        p = Problem3()
    else:
        p = Problem1()

    p.solveProblem(sys.argv[1], sys.argv[2])

main()