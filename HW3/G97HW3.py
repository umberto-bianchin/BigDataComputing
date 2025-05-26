from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random as rn
import numpy as np
from collections import defaultdict

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
    
    conservative_count_min_sketch(batch_items)
    # If we wanted, here we could run some additional code on the global histogram
    #if batch_size > 0:
    #   print("Batch size at time [{0}] is: {1}".format(time, batch_size))
    

    if streamLength[0] >= T:
        stopping_condition.set()


def conservative_count_min_sketch(batch_items):
    for key in batch_items:
        item_map = {}
        columns = {}
        for i, row in enumerate(count_min):
            column = hash_compute(hash_list[i], W, key)
            columns[i] = column
            item_map[i] = row[column]

        min_value = min(item_map.values())
        for map_key, value in item_map.items():
            if value == min_value:
                count_min[map_key, columns[map_key]] += 1

def estimate_frequency(u):
    item_map = {}
    columns = {}
    for i, row in enumerate(count_min):
        column = hash_compute(hash_list[i], W, u)
        columns[i] = column
        item_map[i] = row[column]
    
    return min(item_map.values())

def hash_compute(h, C, x):
    p = 8191
    
    a = h[0]
    b = h[1]

    return ((a * x + b) % p) % C

def hash_function():
    p = 8191

    a = rn.randint(1, p - 1)
    b = rn.randint(0, p - 1)

    return (a, b)

def heavy_hitters():
    histogram_sorted = sorted(histogram.values(), reverse=True)
    phiK = histogram_sorted[K - 1]  # index starts from zero
    top_k = []

    for key, freq in histogram.items():
        if freq >= phiK:
            top_k.append(key)

    return top_k

if __name__ == "__main__":

    #1 - Prints the command-line arguments and stores L,K,M into suitable variables.
    assert len(sys.argv) == 6, "Usage: python G97HW3.py <portExp> <T> <D> <W> <K>"
    
    portExp = sys.argv[1]  # port number
    T = sys.argv[2]         # target number of items to process
    D = sys.argv[3]         # number of rows of each sketch
    W = sys.argv[4]         # number of columns of each sketch
    K = sys.argv[5]         # number of top frequent items of interest
    
    #assert os.path.isfile(data_path), "File or folder not found"
    assert portExp.isdigit(), "portExp must be an integer"
    assert T.isdigit(), "T must be an integer"
    assert D.isdigit(), "D must be an integer"
    assert W.isdigit(), "W must be an integer"
    assert K.isdigit(), "K must be an integer"
    
    portExp, T, D, W, K = map(int, (portExp, T, D, W, K))

    print(f"portExp = {portExp}, T = {T}, D = {D}, W = {W}, K = {K}")

    conf = SparkConf().setMaster("local[*]").setAppName("G97HW3")
    #conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event()

    streamLength = [0]
    histogram = defaultdict(int)
    count_min = np.zeros((D, W), dtype=int)
    hash_list = [hash_function() for _ in range(D)]

    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    

    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")

    ssc.stop(False, False)
    print("Streaming engine stopped")

    # COMPUTE AND PRINT FINAL STATISTICS

    topK = heavy_hitters()
    relative_errors = 0

    for num in topK:
        estimated_frequency = estimate_frequency(num)
        real_frequency = histogram[num]

        relative_error = abs(real_frequency - estimated_frequency) / real_frequency
        relative_errors += relative_error

    avg_relative_error = relative_errors / len(topK)

    print("Number of distinct items =", len(histogram))
    print("Average Relative Error CM =", avg_relative_error)