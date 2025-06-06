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
def process_batch(batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram
    batch_size = batch.count()

    # If the batch is empty or we already have enough points (> THRESHOLD), skip this batch.
    if batch_size == 0 or streamLength[0]>=T:
        return

    streamLength[0] += batch_size

    batch_items = batch.map(int).countByValue()   # dict {key: count}
    for key, cnt in batch_items.items():
        histogram[key] += cnt
        for _ in range(cnt):
            count_min_sketch(key)
            count_sketch(key)

    if streamLength[0] >= T:
        stopping_condition.set()


# ======================================================
#           COUNT-MIN & COUNT SKETCH ESTIMATORS
# ======================================================
def count_min_sketch(key):
    for i in range(D):
        column = hash_compute(hashList[i], W, key)
        countMin[i][column] += 1


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
    stream.foreachRDD(lambda batch: process_batch(batch))
    
    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)

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
    print(f"Port = {portExp} T = {T} D = {D} W = {W} K = {K}")
    print("Number of processed items =", streamLength[0])
    print("Number of distinct items =", len(histogram))
    print("Number of Top-K Heavy Hitters = ", len(topK))
    print("Avg Relative Error for Top-K Heavy Hitters with CM =", avg_relative_error_min)
    print("Avg Relative Error for Top-K Heavy Hitters with CS =", avg_relative_error_sketch)

    if K <= 10:
        print("Top-K Heavy Hitters:")
        for num in sorted(topK):
            estimated_frequency_min = estimate_frequency(num)
            true_frequency = histogram[num]
            print(f"Item {num} True frequency = {true_frequency} Estimated frequency with CM = {estimated_frequency_min}")
