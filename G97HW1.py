from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import numpy as np
import math
import sys
import os

# ======================================================
#                  UTILITY FUNCTIONS
# ======================================================

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    return sum((p1 - p2) ** 2)

# Function to find the closest centroid for a given point
def closest_centroid(point, centroids):
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances)

# Function to parse each line of the input file into a tuple (point, group)
def parse_line(line):
    values = line.strip().split(',')
    point = tuple(map(float, values[:-1]))
    group = values[-1]

    return (point, group)

# ======================================================
#             DISTANCE MAPPING FUNCTIONS
# ======================================================

# For MRComputeStandardObjective: point to distance
def point_distance(point_group, bc_centroids):
        point = point_group[0]
        centroids = bc_centroids.value
        closest = closest_centroid(point, centroids)
        return euclidean_distance(point, centroids[closest])

# For MRComputeFairObjective: point to (group, (distance, 1))
def group_point_distance(point_group, bc_centroids):
        point, group = point_group
        centroids = bc_centroids.value
        closest = closest_centroid(point, centroids)
        dist = euclidean_distance(point, centroids[closest])
        return (group, (dist, 1))

# For MRPrintStatistics: point to ((centroid_idx, group), 1)
def map_to_centroid_group(point_group, bc_centroids):
        point, group = point_group
        centroids = bc_centroids.value
        idx = closest_centroid(point, centroids)
        return ((idx, group), 1)

# ======================================================
#               OBJECTIVE FUNCTIONS
# ======================================================

# Function to compute the standard objective function Δ(𝑈,𝐶)
def MRComputeStandardObjective(inputPoints, C):
    bc_centroids = inputPoints.context.broadcast(C)

    total_distance = inputPoints.map(lambda point_group: point_distance(point_group, bc_centroids)).sum()

    obj_funct = total_distance / inputPoints.count()
    return obj_funct

# Function to compute the fairness objective function Φ(𝐴,𝐵,𝐶)
def MRComputeFairObjective(inputPoints, C):
    bc_centroids = inputPoints.context.broadcast(C)

    group_distances = inputPoints.map(lambda point_group: group_point_distance(point_group, bc_centroids))
    reduced = (group_distances.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).collectAsMap())
    print(reduced)
    avg_A = reduced.get('A', (0, 1))
    avg_B = reduced.get('B', (0, 1))

    obj_funct = max(avg_A[0] / avg_A[1], avg_B[0] / avg_B[1])

    return obj_funct

# ======================================================
#            STATISTICS REPORTING FUNCTION
# ======================================================

# Function to print statistics (centroid, number of points in A and B)
def MRPrintStatistics(inputPoints, C):
    bc_centroids = inputPoints.context.broadcast(C)

    counts = (inputPoints.map(lambda point_group: map_to_centroid_group(point_group, bc_centroids)).reduceByKey(lambda x, y: x + y)
              .map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(dict))
    
    counts_map = dict(counts.collect())
    for i, center in enumerate(C):
        counts_dict = counts_map.get(i, {})
        NA = counts_dict.get('A', 0)
        NB = counts_dict.get('B', 0)
        formatted_center = ", ".join(f"{coord:.6f}" for coord in center)
        print(f"i = {i}, center = ({formatted_center}), NA{i} = {NA}, NB{i} = {NB}")


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
    
    L, K, M = map(int, (L, K, M))

    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    
    #2 - Reads the input points into an RDD of (point,group) pairs subdivided into 𝐿 partitions.
    conf = SparkConf().setAppName('G97HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    inputPoints = sc.textFile(data_path).map(lambda line: parse_line(line)).repartition(numPartitions=L).cache()

    #3 - Prints the number 𝑁 of points, the number 𝑁𝐴 of points of group A, and the number 𝑁𝐵 of points of group B
    NA = inputPoints.filter(lambda x: x[-1] == 'A').count()
    NB = inputPoints.filter(lambda x: x[-1] == 'B').count()
    N = NA + NB

    print(f"N = {N}, NA = {NA}, NB = {NB}")

    #4 - Computes a set 𝐶 of 𝐾 centroids by using the Spark implementation of the standard Lloyd's algorithm for the input points using 𝑀 as number of iterations.
    data = inputPoints.map(lambda x: np.array(x[0]))
    model = KMeans.train(data, K, M)
    
    #5 - Prints the values of the two objective functions Δ(𝑈,𝐶) and Φ(𝐴,𝐵,𝐶)
    print(f"Delta(U,C) = {MRComputeStandardObjective(inputPoints, model.centers)}")
    print(f"Phi(A,B,C) = {MRComputeFairObjective(inputPoints, model.centers)}")

    #6 - Runs MRPrintStatistics
    MRPrintStatistics(inputPoints, model.centers)

    sc.stop()

if __name__ == "__main__":
	main()