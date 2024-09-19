import yaml
import numpy as np
from matplotlib import pyplot as plt
from mesh_utils.mesh import MeshObj
import matplotlib

# Set the global font to Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

# Load data from JSON file
with open('list_heatmaps_results_sept19.json', 'r') as json_file:
    data = yaml.safe_load(json_file)

fig = plt.figure()
ax = fig.add_subplot(111)

# Load and prepare target data
targets = MeshObj(adf_num=0, stl_num=1, body_index=1)
targets.points *= -10

best_case = data[-1]
best_heatmap = best_case["best_heatmap"]

# Plot the scatter plot
ax.scatter(targets.points[:, 1], targets.points[:, 0], c=best_heatmap, cmap='plasma')

# Remove all non-data entities
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_aspect('equal')

plt.show()