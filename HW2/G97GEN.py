# G97GEN.py
import sys
import math
import random
import csv
import os

'''
The script G97GEN.py generates a synthetic 2D dataset of N points with demographic labels (A or B).
The dataset is structured to highlight the quality gap between standard Lloyd's algorithm and its fair variant.
It creates K-1 compact clusters for the majority group A, and a single wide-spread cluster for the minority group B.
Points are printed and saved in CSV format (datasets/G97GEN.csv).
'''

def gen_dataset(N, K):
    output_file = os.path.join("datasets", "G97GEN.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sigma_A = 1.0    # spread for A clusters
    sigma_B = 15.0   # spread for single B cluster

    N_B = max(1, int(0.1 * N)) #betha = 0.1
    N_A = N - N_B
    A_per_cluster, remainder = divmod(N_A, K - 1)

    dataset = []

    # Generate B points (minority group, one loose cluster)
    dataset.extend(
        [[f"{random.gauss(0, sigma_B):.4f}", f"{random.gauss(0, sigma_B):.4f}", "B"]
         for _ in range(N_B)]
    )


    # Generate (K-1) clusters for A
    for i, angle in enumerate(math.tau * j / (K - 1) for j in range(K - 1)):
        cx, cy = 50 * math.cos(angle), 50 * math.sin(angle) #radius = 50
        num_points = A_per_cluster + (i < remainder)
        dataset.extend(
            [ [f"{random.gauss(cx, sigma_A):.4f}", f"{random.gauss(cy, sigma_A):.4f}", "A"] 
              for _ in range(num_points) ]
        )


    # Shuffle and write to file and console
    random.shuffle(dataset)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point in dataset:
            writer.writerow(point)
            print(point)

    print(f"Saved {len(dataset)} points to {output_file}")

def main():

    if len(sys.argv) != 3:
        print("Usage: python G97GEN.py N K")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
        K = int(sys.argv[2])
        if K < 2:
            raise ValueError
    except ValueError:
        print("N must be an integer, and K must be an integer ≥ 2.")
        sys.exit(1)


    gen_dataset(N, K)

if __name__ == "__main__":
    main()
