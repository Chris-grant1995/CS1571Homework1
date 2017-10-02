CS 1571 Homework 1 Readme

I wrote this homework assignment 2.7.10 and should work on any version of python 2.

The libraries that I used were the Queue, ast, collections and math libraries.
All of these are included in the python 2 standard library, and therefore no additional installations should be required.

I used this webpage as a starting point for the uninformed search strategies, and from there modified them to suit the problems.
http://cyluun.github.io/blog/uninformed-search-algorithms-in-python

I spoke with Ryan Yoder and Paul Davis about the assignment. For the most part, we talked about implementation details and worked
together to debug issues that we were having.

One thing I noticed during my testing using the samples provided is that if there is an extra newline at the end of the file, the script will crash as parsing the input will fail. If you experience this, please make sure that there is no newline at the end of the file. I didn't have time to fix this and as the problem description says nothing about handling an extra newline at the end of the file, it should not be there in the first place. 

If you want to run all algorithms on a certain file you can run:
	python puzzlesolver.py file.config all
That is what I used to generate the files in the outputs folder, with algorithms that loop infinitely on a problem removed. 