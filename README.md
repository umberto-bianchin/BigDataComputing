# Big Data Computing Projects
This repository contains the solutions to three homework assignments developed for the **Big Data Computing** course. The implementations make use of **Apache Spark** for distributed data processing and are written in **Python** using the PySpark API.

## Authors

* [@umberto-bianchin](https://www.github.com/umberto-bianchin)
* [@Massimo-Vettorello](https://github.com/Massimo-Vettorello)

## 🧠 Overview

### 🔹 HW1 – Fair Assignment and Statistics with MapReduce
Implements a two-phase MapReduce pipeline in Spark to:
- Assign data points to the closest centroid using Lloyd's algorithm.
- Compute group-based statistics (counts per group per centroid) to evaluate fairness.
- Analyze memory usage and communication overhead in the distributed setting.

Relevant files:
- `G97HW1.py` – main logic
- `G97HW1analysis.docx` – theoretical memory cost analysis

---

### 🔹 HW2 – Fair K-Means with Synthetic Data
Implements and compares the **standard Lloyd’s algorithm** and its **fair variant**, using Spark's parallelism. The fair variant ensures demographic balance between groups A and B within each cluster.

Additional features:
- Scalability testing with different numbers of executors
- Timing of different pipeline stages
- Support for synthetic datasets with controlled group imbalance

Relevant files:
- `G97HW2.py` – clustering logic
- `G97GEN.py` – synthetic dataset generator
- `G97HW2form.docx` – scalability test results and explanations

---

### 🔹 HW3 – Heavy Hitters with Count-Min and Count Sketch
Processes a data stream using Spark Streaming to:
- Track item frequencies using **Count-Min Sketch** and **Count Sketch**.
- Identify the **Top-K heavy hitters** in the stream.
- Evaluate estimation accuracy for both sketches based on average relative error.

Features:
- Real-time processing with custom stopping criteria
- Accuracy experiments for different values of K and sketch width W

Relevant files:
- `G97HW3.py` – streaming logic
- `G97HW3Form.docx` – experiment results and analysis

---

## 🔧 Technologies Used
- Python 3
- Apache Spark (PySpark)
- Spark Streaming
- HDFS (for input data)
- NumPy, statistics, collections