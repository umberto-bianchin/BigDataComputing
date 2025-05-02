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

def compute_vectors(stats, NA, NB, C, K):
    alpha, betha = [], []
    mu_A, mu_B, l = [], [], []

    for i in range(K):
        sumA, cntA = stats.get((i, 'A'), (np.zeros_like(C[0]), 0))
        sumB, cntB = stats.get((i, 'B'), (np.zeros_like(C[0]), 0))

        alpha_i = cntA / NA if NA > 0 else 0.0
        betha_i= cntB / NB if NB > 0 else 0.0

        mu_A_i = sumA / cntA if cntA > 0 else np.zeros(C[0])
        mu_B_i = sumB / cntB if cntB > 0 else np.zeros_like(C[0])

        alpha.append(alpha_i)
        betha.append(betha_i)
        
        mu_A.append(mu_A_i)
        mu_B.append(mu_B_i)
        
        l.append(float(np.linalg.norm(mu_A_i - mu_B_i)))
    return alpha, betha, mu_A, mu_B, l

def computeVectorX(fixedA, fixedB, alpha, betha, l, K, T=10):
    gamma = 0.5
    x = [0.0 for i in range(K)]
    
    for t in range(T):
        f_A = fixedA
        f_B = fixedB
        for i in range(K):
            denom = gamma * alpha[i] + (1 - gamma) * betha[i]
            if(denom > 0):
                x[i] = ((1 - gamma) * betha[i] * l[i]) / denom
            else:
                x[i] = 0.0

            f_A += alpha[i] * x[i] **2
            f_B += betha[i] * (l[i] - x[i]) **2
            
        if f_A == f_B:
             break
        else:
            if f_A > f_B:
                gamma += (0.5) **(t+1)
            else:
                gamma -= (0.5) **(t+1)

    return x

def centroid_selection(inputPoints, stats, NA, NB, C, K):
    alpha, betha, mu_A, mu_B, l = compute_vectors(stats, NA, NB, C,K)


    bcMuA = inputPoints.context.broadcast({i:mu_A[i] for i in range(K)})
    bcMuB = inputPoints.context.broadcast({i:mu_B[i] for i in range(K)})

    #(gruppo, (distanza ^2, 1))
    def obj_pair(pg):
        pt, g = pg
        i = closest_centroid(pt, C)
        mu = bcMuA.value[i] if g=='A' else bcMuB.value[i]
        d2 = float(((np.array(pt) - mu)**2).sum())
        return (g, (d2, 1))

    reduced = (inputPoints
                .map(obj_pair)
                .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
                .collectAsMap())

    fixedA = (reduced.get('A', (0.0,1))[0] / NA) if NA > 0 else 0.0
    fixedB = (reduced.get('B', (0.0,1))[0] / NB) if NB > 0 else 0.0
    
    bcMuA.unpersist()
    bcMuB.unpersist()

    x = computeVectorX(fixedA, fixedB, alpha, betha, l, K)
    
    newC = []
    for i in range(K):
        
        if l[i] > 0:
            ci = ((l[i]-x[i])*mu_A[i]+x[i]*mu_B[i])/l[i]
        else:
            ci = mu_A[i] # mu_A[i] == mu_B[i]

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

# Function to compute the fairness Lloyd's algorithm
def MRFairLloyd(inputPoints, K, M):
    data = inputPoints.map(lambda x: np.array(x[0]))
    model = KMeans.train(data, K, 0)
    C = model.centers

    # |A| and |B|
    NA = inputPoints.filter(lambda x: x[-1] == 'A').count()
    NB = inputPoints.filter(lambda x: x[-1] == 'B').count()

    for _ in range(M):
        # Broadcast current C
        bcC = inputPoints.context.broadcast(C)

        #2.1 - Partition U into k clusters U1,U2,...,Uk where Ui consists of the points of U whose closest current centroid is ci
        stats = (
            inputPoints
            .map(lambda x: (
                (closest_centroid(x[0], bcC.value), x[1]),  #key = (closest_centroid, group)
                (np.array(x[0]), 1)                         #value = (point, 1)
            ))
            .aggregateByKey(
                (np.zeros_like(C[0]), 0),
                lambda acc, val: (acc[0] + val[0], acc[1] + val[1]),    #for each key sum the vector and increment the counter
                lambda a, b: (a[0] + b[0], a[1] + b[1])                 #out togheter results from different partitions
            )
            .collectAsMap()
        )
        bcC.unpersist()

        #2.2 - Compute a new set {c1, ..., ck} of K centroids
        C = centroid_selection(inputPoints, stats, NA, NB, C, K)

    return [tuple(c.tolist()) for c in C]

def main():

    #1 - Prints the command-line arguments and stores L,K,M into suitable variables.
    assert len(sys.argv) == 5, "Usage: python G97HW2.py <file_name> <L> <K> <M>"
    
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
    conf = SparkConf().setAppName('G97HW2')
    #conf.set("spark.driver.bindAddress", "127.0.0.1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    inputPoints = sc.textFile(data_path).map(lambda line: parse_line(line)).repartition(numPartitions=L).cache()
    inputPoints.count() # Used to load the document before the execution to measure the times

    #3 - Prints the number 𝑁 of points, the number 𝑁𝐴 of points of group A, and the number 𝑁𝐵 of points of group B
    NA = inputPoints.filter(lambda x: x[-1] == 'A').count()
    NB = inputPoints.filter(lambda x: x[-1] == 'B').count()
    N = NA + NB

    print(f"N = {N}, NA = {NA}, NB = {NB}")

    #4 - Computes a set 𝐶stand of 𝐾 centroids for the input points, by running the Spark implementation of the standard Lloyd's algorithm, with 𝑀 iterations, disregarding the demographic groups.
    data = inputPoints.map(lambda x: np.array(x[0]))

    start = time.perf_counter()
    model = KMeans.train(data, K, M)
    end = time.perf_counter()
    time_C_stand = f"{end - start:.4f}"
    
    #5 - Computes a set 𝐶fair of 𝐾 centroids by running MRFairLloyd(inputPoints,K,M).
    start = time.perf_counter()
    C = MRFairLloyd(inputPoints, K, M)
    end = time.perf_counter()
    time_C_fair = f"{end- start:.4f}"

    #5 - Computes and prints Φ(𝐴,𝐵,𝐶stand) and Φ(𝐴,𝐵,𝐶fair)
    start = time.perf_counter()
    phiStand = MRComputeFairObjective(inputPoints, model.centers)
    end = time.perf_counter()
    time_Phi_stand = f"{end - start:.4f}"

    start = time.perf_counter()
    phiFair = MRComputeFairObjective(inputPoints, C)
    end = time.perf_counter()
    time_Phi_fair = f"{end- start:.4f}"

    formatted_phi_stand = f"{phiStand:.6f}"
    formatted_phi_fair = f"{phiFair:.6f}"

    print(f"Phi(A,B,Cstand) = {formatted_phi_stand}")
    print(f"Phi(A,B,Cfair) = {formatted_phi_fair}")    

    #6 - Prints separately the times, in seconds, spent to compute : 𝐶stand, 𝐶fair, Φ(𝐴,𝐵,𝐶stand) and Φ(𝐴,𝐵,𝐶fair).
    print(f"Time C stand   = {time_C_stand} s")    
    print(f"Time C fair    = {time_C_fair} s")    
    print(f"Time Phi stand = {time_Phi_stand} s")    
    print(f"Time Phi fair  = {time_Phi_fair} s")    
    
    sc.stop()

if __name__ == "__main__":
	main()