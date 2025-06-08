import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets import load_dataset

# Load partitions and labels
partition_file = "./data/partitions/noniid_partitions_80.pt"
partition_indices = torch.load(partition_file)
cifar = load_dataset("uoft-cs/cifar10")
labels = cifar["train"]["label"]

# Label map
label_map = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# Select 5 random clients
selected_clients = random.sample(range(len(partition_indices)), 10)

# Prepare DataFrame of label proportions per partition
data = []
for client_id in selected_clients:
    total = len(partition_indices[client_id])
    class_counts = [0] * 10
    for idx in partition_indices[client_id]:
        class_counts[labels[idx]] += 1
    for label, count in enumerate(class_counts):
        data.append({
            "Partition ID": client_id,
            "Label": label_map[label],
            "Percent": (count / total) * 100
        })

df = pd.DataFrame(data)

# Plot stacked bar chart
pivot_df = df.pivot(index="Partition ID", columns="Label", values="Percent")
pivot_df = pivot_df.fillna(0)

plt.figure(figsize=(12, 6))
bottom = np.zeros(len(pivot_df))
colors = sns.color_palette("Spectral", n_colors=len(pivot_df.columns))

for idx, label in enumerate(pivot_df.columns):
    plt.bar(pivot_df.index, pivot_df[label], bottom=bottom, label=label, color=colors[idx])
    bottom += pivot_df[label].values

plt.title("Per Partition Labels Distribution")
plt.ylabel("Percent %")
plt.xlabel("Partition ID")
plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
