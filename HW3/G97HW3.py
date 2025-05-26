from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random as rn
import numpy as np
from collections import defaultdict
import statistics

SEED = 20   # Seed for reproducibility of hash functions
p = 8191    # param p for hash functions

# ======================================================
#                STREAM BATCH PROCESSING
# ======================================================
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram
    batch_size = batch.count()

    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0]>=T:
        return
    
    streamLength[0] += batch_size
    # Extract the distinct items from the batch
    batch_items = batch.map(lambda s: (int(s), 1)).reduceByKey(lambda i1, i2: 1).collectAsMap()

    # Update the streaming state
    for key in batch_items:
        histogram[key] += 1
    
    for key in batch_items:
        conservative_count_min_sketch(key)
        count_sketch(key)
    
    #if batch_size > 0:
    #   print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= T:
        stopping_condition.set()


# ======================================================
#           COUNT-MIN & COUNT SKETCH ESTIMATORS
# ======================================================
def conservative_count_min_sketch(key):
    item_map = {}
    columns = {}
    for i, row in enumerate(countMin):
        column = hash_compute(hashList[i], W, key)
        columns[i] = column
        item_map[i] = row[column]

    min_value = min(item_map.values())
    for map_key, value in item_map.items():
        if value == min_value:
            countMin[map_key][columns[map_key]] += 1


def count_sketch(key):
    for i in range(D):
        column = hash_compute(hashList2D[i][0], W, key)
        countSketch[i][column] += g_compute(hashList2D[i][1], W, key)

def estimate_frequency(u):
    item_map = {}

    for i in range(D):
        column = hash_compute(hashList[i], W, u)
        item_map[i] = countMin[i][column]
        
    return min(item_map.values())

def estimate_frequency_sketch(u):
    frequencies = []

    for i in range(D):
        column = hash_compute(hashList2D[i][0], W, u)
        freq = g_compute(hashList2D[i][1], W, u) * countSketch[i][column]
        frequencies.append(freq)
    
    ordered_frequencies = sorted(frequencies)
    return statistics.median(ordered_frequencies)

# ======================================================
#               HASH FUNCTION UTILITIES
# ======================================================
def hash_compute(h, C, x):
    a = h[0]
    b = h[1]
    return ((a * x + b) % p) % C

def g_compute(h, C, x):
    a = h[0]
    b = h[1]
    return 1 if ((a * x + b) % p) % C % 2 == 0 else -1

def hash_function():
    a = rn.randint(1, p - 1)
    b = rn.randint(0, p - 1)
    return (a, b)

# ======================================================
#               HEAVY HITTER DETECTION
# ======================================================
def heavy_hitters():
    histogram_sorted = sorted(histogram.values(), reverse=True)
    phiK = histogram_sorted[K - 1]  # index starts from zero
    top_k = []

    for key, freq in histogram.items():
        if freq >= phiK:
            top_k.append(key)

    return top_k


if __name__ == "__main__":
    assert len(sys.argv) == 6, "Usage: python G97HW3.py <portExp> <T> <D> <W> <K>"
    
    portExp = sys.argv[1]   # port number
    T = sys.argv[2]         # target number of items to process
    D = sys.argv[3]         # number of rows of each sketch
    W = sys.argv[4]         # number of columns of each sketch
    K = sys.argv[5]         # number of top frequent items of interest
    
    assert portExp.isdigit(), "portExp must be an integer"
    assert T.isdigit(), "T must be an integer"
    assert D.isdigit(), "D must be an integer"
    assert W.isdigit(), "W must be an integer"
    assert K.isdigit(), "K must be an integer"
    
    portExp, T, D, W, K = map(int, (portExp, T, D, W, K))

    conf = SparkConf().setMaster("local[*]").setAppName("G97HW3")
    #conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event()

    rn.seed(SEED)   # setting the seed for reproducibility

    streamLength = [0]
    histogram = defaultdict(int)
    countMin = np.zeros((D, W), dtype=int)
    countSketch = np.zeros((D, W), dtype=int)
    hashList = [hash_function() for _ in range(D)]
    hashList2D = [[hash_function(), hash_function()] for _ in range(D)]

    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    #print("Starting streaming engine")
    ssc.start()
    #print("Waiting for shutdown condition")
    stopping_condition.wait()
    #print("Stopping the streaming engine")

    ssc.stop(False, False)
    #print("Streaming engine stopped")

    # COMPUTE AND PRINT FINAL STATISTICS
    topK = heavy_hitters()
    relative_errors_min = 0
    relative_errors_sketch = 0

    for num in topK:
        estimated_frequency_min = estimate_frequency(num)
        estimated_frequency_sketch = estimate_frequency_sketch(num)
        real_frequency = histogram[num]

        relative_error_min = abs(real_frequency - estimated_frequency_min) / real_frequency
        relative_error_sketch = abs(real_frequency - estimated_frequency_sketch) / real_frequency
        relative_errors_min += relative_error_min
        relative_errors_sketch += relative_error_sketch

    avg_relative_error_min = relative_errors_min / len(topK)
    avg_relative_error_sketch = relative_errors_sketch / len(topK)

    # PRINTINGS
    print(f"portExp = {portExp}, T = {T}, D = {D}, W = {W}, K = {K}")
    print("Number of distinct items =", len(histogram))
    print("Average Relative Error CM =", avg_relative_error_min)
    print("Average Relative Error CS =", avg_relative_error_sketch)
