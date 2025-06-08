import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter

# Load partition and labels
partition_file = "./data/partitions/noniid_partitions_80.pt"
partition_indices = torch.load(partition_file)
labels = load_dataset("uoft-cs/cifar10")["train"]["label"]

# Label map
label_map = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# Analyze 10 clients
selected_clients = list(range(10))

# Create data matrix: [class x client]
data = []
for label in range(10):
    row = []
    for cid in selected_clients:
        count = sum(1 for i in partition_indices[cid] if labels[i] == label)
        row.append(count)
    data.append(row)

df = pd.DataFrame(data, index=[label_map[i] for i in range(10)], columns=[f"{i}" for i in selected_clients])

# Plot annotated heatmap
plt.figure(figsize=(12, 4))
sns.heatmap(df, annot=True, fmt="d", cmap="Greens", cbar_kws={"label": "Count"})
plt.title("Per Partition Labels Distribution")
plt.xlabel("Partition ID")
plt.ylabel("")
plt.tight_layout()
plt.show()
