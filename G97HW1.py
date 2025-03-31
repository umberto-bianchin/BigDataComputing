from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import numpy as np
import math
import sys
import os

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(sum((point1 - point2) ** 2))

# Function to find the closest centroid for a given point
def closest_centroid(point, centroids):
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances)

# Function to compute the standard objective function Δ(𝑈,𝐶)
def MRComputeStandardObjective(inputPoints, C):
    distance = 0
    for point in inputPoints.collect():
         distance += euclidean_distance(point[0], C[closest_centroid(point[0], C)])

    obj_funct = (1/inputPoints.count()) * distance
    return obj_funct

# Function to compute the fairness objective function Φ(𝐴,𝐵,𝐶)
def MRComputeFairObjective(inputPoints, C):
    A = inputPoints.filter(lambda x: x[-1] == 'A')
    B = inputPoints.filter(lambda x: x[-1] == 'B')

    sum_A = 0
    for point in A.collect():
        sum_A += euclidean_distance(point[0], C[closest_centroid(point[0],C)])
    
    sum_B = 0
    for point in B.collect():
        sum_B += euclidean_distance(point[0], C[closest_centroid(point[0],C)])

    obj_funct = max(1/A.count()*sum_A,1/B.count()*sum_B)

    return obj_funct

# Function to print statistics (centroid, number of points in A and B)
def MRPrintStatistics(inputPoints, C):
    A = inputPoints.filter(lambda x: x[-1] == 'A')
    B = inputPoints.filter(lambda x: x[-1] == 'B')

    centroids_A = np.zeros(len(C))
    centroids_B = np.zeros(len(C))
    
    for point in A.collect():
        centroids_A[closest_centroid(point[0],C)] += 1
        
    for point in B.collect():
        centroids_B[closest_centroid(point[0],C)] += 1

    for i, centroid in enumerate(C):
        print(f"i = {i}, center = ({centroid}), NA{i} = {int(centroids_A[i])}, NB{i} = {int(centroids_B[i])}")

# Function to parse each line of the input file into a tuple (point, group)
def parse_line(line):
    values = line.split(',')
    point = tuple(map(float, values[:-1]))
    group = values[-1]

    return (point, group)


def main():

    #1 - Prints the command-line arguments and stores L,K,M into suitable variables.
    assert len(sys.argv) == 5, "Usage: python G97HW1.py <file_name> <L> <K> <M>"
    
    data_path = sys.argv[1]
    L = sys.argv[2]
    K = sys.argv[3]
    M = sys.argv[4]
    
    assert os.path.isfile(data_path), "File or folder not found"
    assert L.isdigit(), "L must be an integer"
    assert K.isdigit(), "K must be an integer"
    assert M.isdigit(), "M must be an integer"
       
    L = int(L)
    K = int(K)
    M = int(M)

    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    
    #2 - Reads the input points into an RDD of (point,group) pairs subdivided into 𝐿 partitions.
    conf = SparkConf().setAppName('G97HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    inputPoints = sc.textFile(data_path).map(lambda line: parse_line(line)).repartition(numPartitions=L).cache()

    #3 - Prints the number 𝑁 of points, the number 𝑁𝐴 of points of group A, and the number 𝑁𝐵 of points of group B
    print(f"N = {inputPoints.count()}, NA = {inputPoints.filter(lambda x: x[-1] == 'A').count()}, NB = {inputPoints.filter(lambda x: x[-1] == 'B').count()}")

    #4 - Computes a set 𝐶 of 𝐾 centroids by using the Spark implementation of the standard Lloyd's algorithm for the input points using 𝑀 as number of iterations.
    data = inputPoints.map(lambda x: np.array(x[0]))
    model = KMeans.train(data, K, M)
    
    #5 - Prints the values of the two objective functions Δ(𝑈,𝐶) and Φ(𝐴,𝐵,𝐶)
    print(f"Delta(U,C) = {MRComputeStandardObjective(inputPoints, model.centers)}")
    print(f"Phi(A,B,C) = {MRComputeFairObjective(inputPoints, model.centers)}")

    #6 - Runs MRPrintStatistics
    MRPrintStatistics(inputPoints, model.centers)

if __name__ == "__main__":
	main()