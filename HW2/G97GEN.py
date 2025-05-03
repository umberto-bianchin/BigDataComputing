# G97GEN.py
import sys
import random
import os
import csv
'''
The script G97GEN.py generates a synthetic dataset of NN two-dimensional points with demographic labels (A or B), designed to highlight the difference in clustering quality between standard Lloyd's algorithm and its fair variant. The generator creates KK spatial clusters within the approximate NYC area. Each cluster is biased toward one demographic group, alternating between A-majority and B-majority clusters (with a 90%-10% distribution). This deliberate imbalance ensures that standard clustering may group similar points but fail to respect demographic fairness, while the fair variant should yield more balanced groupings. The output is saved in CSV format and can be directly used as input for clustering algorithms.
'''
def generate_cluster(center_lat, center_lon, radius, num_points, group_ratio):
    points = []
    for _ in range(num_points):
        lat = round(random.uniform(center_lat - radius, center_lat + radius), 6)
        lon = round(random.uniform(center_lon - radius, center_lon + radius), 6)
        label = random.choices(['A', 'B'], weights=group_ratio)[0]
        points.append((lat, lon, label))
    return points

def main():
    if len(sys.argv) != 3:
        print("Usage: python G97GEN.py N K")
        sys.exit(1)

    N = int(sys.argv[1])
    K = int(sys.argv[2])
    points_per_cluster = N // K
    remaining = N % K

    # NYC-area cluster centers
    cluster_centers = [
        (40.75, -73.99),
        (40.72, -74.00),
        (40.78, -73.97),
        (40.69, -74.01),
        (40.80, -73.95)
    ]
    random.shuffle(cluster_centers)

    while len(cluster_centers) < K:
        base = random.choice(cluster_centers)
        perturb = (random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005))
        cluster_centers.append((base[0] + perturb[0], base[1] + perturb[1]))

    dataset = []
    for i in range(K):
        center = cluster_centers[i]
        group_ratio = [0.98, 0.02] if i % 2 == 0 else [0.02, 0.98]
        count = points_per_cluster + (1 if i < remaining else 0)
        cluster_points = generate_cluster(center[0], center[1], radius=0.01, num_points=count, group_ratio=group_ratio)
        dataset.extend(cluster_points)

    # Shuffle and save to CSV
    random.shuffle(dataset)
    output_path = os.path.join("datasets", "G97GEN.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for lat, lon, label in dataset:
            writer.writerow([lat, lon, label])
            print([lat, lon, label])

    print(f"Saved {len(dataset)} points to {output_path}")

if __name__ == "__main__":
    main()