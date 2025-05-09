from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import numpy as np
import time
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

# Function to map a point to its closest centroid, collecting separate stats for groups A and B
def map_stats(point_group, centroids):
    pt, g = np.array(point_group[0]), point_group[1]
    idx = closest_centroid(pt, centroids)
    # If group A: return (ptA, sumSqA, countA, zeroB, zeroSqB, zeroCountB)
    if g == 'A':
        return idx, (pt, np.dot(pt, pt), 1, np.zeros_like(pt), 0.0, 0)
    # If group B: return (zeroA, zeroSqA, zeroCountA, ptB, sumSqB, countB)
    else:
        return idx, (np.zeros_like(pt), 0.0, 0, pt, np.dot(pt, pt), 1)

# Function to reduce two tuples by element-wise summation
def reduce_stats(a, b):
    return tuple(a[i] + b[i] for i in range(6))


# ======================================================
#            VECTOR AND CENTROID COMPUTATION
# ======================================================

# Given stats for each cluster i: stats[i] = (sumA, sumSqA, cntA, sumB, sumSqB, cntB)
# Compute all the needed vectors
def compute_vectors(stats, NA, NB, C, K):
    alpha = [0.0] * K
    beta = [0.0] * K
    muA = [None] * K
    muB = [None] * K
    ell = [0.0] * K
    deltaA = 0.0
    deltaB = 0.0

    for i in range(K):
        # Retrieve sum and count for group A and B, default zeros
        sumA, sqA, cntA, sumB, sqB, cntB = stats.get(i, (np.zeros_like(C[0]), 0.0, 0, np.zeros_like(C[0]), 0.0, 0))

        alpha_i = cntA / NA if NA > 0 else 0.0
        betha_i= cntB / NB if NB > 0 else 0.0

        mu_A_i = sumA / cntA if cntA > 0 else C[i]
        mu_B_i = sumB / cntB if cntB > 0 else C[i]

        alpha[i] = alpha_i
        beta[i] = betha_i
        
        muA[i] = mu_A_i
        muB[i] = mu_B_i

        deltaA += sqA - cntA * np.dot(mu_A_i, mu_A_i)
        deltaB += sqB - cntB * np.dot(mu_B_i, mu_B_i)
        
        ell[i] = float(np.linalg.norm(mu_A_i - mu_B_i))

    return alpha, beta, muA, muB, ell, deltaA, deltaB

def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
    gamma = 0.5
    x_dist = [0.0] * k
    power = 0.5
    t_max = 10

    for _ in range(t_max):
        f_a = fixed_a
        f_b = fixed_b
        power /= 2

        for i in range(k):
            temp = (1 - gamma) * beta[i] * ell[i] / (gamma * alpha[i] + (1 - gamma) * beta[i])
            x_dist[i] = temp
            f_a += alpha[i] * temp * temp
            temp = ell[i] - temp
            f_b += beta[i] * temp * temp

        if f_a == f_b:
            break

        gamma = gamma + power if f_a > f_b else gamma - power

    return x_dist

# Function to compute new fair centroids for all clusters
def centroid_selection(stats, NA, NB, C, K):
    alpha, betha, mu_A, mu_B, ell, deltaA, deltaB = compute_vectors(stats, NA, NB, C, K)

    fixedA = deltaA / NA
    fixedB = deltaB / NB

    x = computeVectorX(fixedA, fixedB, alpha, betha, ell, K)
    
    newC = []
    for i in range(K):
        if ell[i] > 0:
            ci = ((ell[i] - x[i]) * mu_A[i] + x[i] * mu_B[i]) / ell[i]
        else:
            ci = C[i]

        newC.append(ci)

    return newC
    

# ======================================================
#             DISTANCE MAPPING FUNCTIONS
# ======================================================

# For MRComputeFairObjective: point to (group, (distance, 1))
def group_point_distance(point_group, bc_centroids):
        point, group = point_group
        centroids = bc_centroids.value
        closest = closest_centroid(point, centroids)
        dist = euclidean_distance(point, centroids[closest])
        return (group, (dist, 1))


# ======================================================
#               OBJECTIVE FUNCTIONS
# ======================================================

# Function to compute the fairness objective function Φ(𝐴,𝐵,𝐶)
def MRComputeFairObjective(inputPoints, C):
    bc_centroids = inputPoints.context.broadcast(C)

    group_distances = inputPoints.map(lambda point_group: group_point_distance(point_group, bc_centroids))
    reduced = (group_distances.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).collectAsMap())

    avg_A = reduced.get('A', (0, 1))
    avg_B = reduced.get('B', (0, 1))

    obj_funct = max(avg_A[0] / avg_A[1], avg_B[0] / avg_B[1])

    return obj_funct
    
# Function to compute the fair Lloyd's algorithm
def MRFairLloyd(inputPoints, K, M):
    # Initialize centroids via kmeans (0 iterations)
    data = inputPoints.map(lambda x: np.array(x[0]))
    model = KMeans.train(data, K, 0)
    C = model.centers

    # |A| and |B|
    NA = inputPoints.filter(lambda x: x[-1] == 'A').count()
    NB = inputPoints.filter(lambda x: x[-1] == 'B').count()

    for _ in range(M):
        # Broadcast current C
        bc_cent = inputPoints.context.broadcast(C)

        #2.1 - Partition U into k clusters U1,U2,...,Uk where Ui consists of the points of U whose closest current centroid is ci
        #stats: dict mapping cluster_idx -> (sumA, sumSqA, cntA, sumB, sumSqB, cntB)
        stats = (
            inputPoints
            .map(lambda x: map_stats(x, bc_cent.value))
            .reduceByKey(reduce_stats)
            .collectAsMap()
        )
        bc_cent.destroy()

        #2.2 - Compute a new set {c1, ..., ck} of K centroids
        C = centroid_selection(stats, NA, NB, C, K)

    return [tuple(c.tolist()) for c in C]

def main():

    #1 - Prints the command-line arguments and stores L,K,M into suitable variables.
    assert len(sys.argv) == 5, "Usage: python G97HW2.py <file_name> <L> <K> <M>"
    
    data_path = sys.argv[1]
    L = sys.argv[2]
    K = sys.argv[3]
    M = sys.argv[4]
    
    #assert os.path.isfile(data_path), "File or folder not found"
    assert L.isdigit(), "L must be an integer"
    assert K.isdigit(), "K must be an integer"
    assert M.isdigit(), "M must be an integer"
    
    L, K, M = map(int, (L, K, M))

    #2 - Reads the input points into an RDD of (point,group) pairs subdivided into 𝐿 partitions.
    conf = SparkConf().setAppName('G97HW2')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    inputPoints = sc.textFile(data_path).map(lambda line: parse_line(line)).repartition(numPartitions=L).cache()
    inputPoints.count() # Used to load the document before the execution to measure the times

    #3 - Prints the number 𝑁 of points, the number 𝑁𝐴 of points of group A, and the number 𝑁𝐵 of points of group B
    NA = inputPoints.filter(lambda x: x[-1] == 'A').count()
    NB = inputPoints.filter(lambda x: x[-1] == 'B').count()
    N = NA + NB

    #4 - Computes a set 𝐶stand of 𝐾 centroids for the input points, by running the Spark implementation of the standard Lloyd's algorithm, with 𝑀 iterations, disregarding the demographic groups.
    data = inputPoints.map(lambda x: np.array(x[0]))

    start = time.perf_counter()
    model = KMeans.train(data, K, M)
    end = time.perf_counter()
    time_C_stand = f"{(end - start)*1000:.0f}"
    
    #5 - Computes a set 𝐶fair of 𝐾 centroids by running MRFairLloyd(inputPoints,K,M).
    start = time.perf_counter()
    C = MRFairLloyd(inputPoints, K, M)
    end = time.perf_counter()
    time_C_fair = f"{(end - start)*1000:.0f}"

    #5 - Computes and prints Φ(𝐴,𝐵,𝐶stand) and Φ(𝐴,𝐵,𝐶fair)
    start = time.perf_counter()
    phiStand = MRComputeFairObjective(inputPoints, model.centers)
    end = time.perf_counter()
    time_Phi_stand = f"{(end - start)*1000:.0f}"

    start = time.perf_counter()
    phiFair = MRComputeFairObjective(inputPoints, C)
    end = time.perf_counter()
    time_Phi_fair = f"{(end - start)*1000:.0f}"

    formatted_phi_stand = f"{phiStand:.4f}"
    formatted_phi_fair = f"{phiFair:.4f}"

    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {NA}, NB = {NB}")

    print(f"Fair Objective with Standard Centers = {formatted_phi_stand}")
    print(f"Fair Objective with Fair Centers = {formatted_phi_fair}")    

    #6 - Prints separately the times, in seconds, spent to compute : 𝐶stand, 𝐶fair, Φ(𝐴,𝐵,𝐶stand) and Φ(𝐴,𝐵,𝐶fair).
    print(f"Time to compute standard centers = {time_C_stand} ms")    
    print(f"Time to compute fair centers = {time_C_fair} ms")    
    print(f"Time to compute objective with standard centers = {time_Phi_stand} ms")    
    print(f"Time to compute objective with fair centers = {time_Phi_fair} ms")    
    
    sc.stop()

if __name__ == "__main__":
	main()