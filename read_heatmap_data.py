import yaml
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import pyplot as plt
from mesh_utils.mesh import MeshObj
import matplotlib

# Set the global font to Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.axisbelow'] = True

with open('list_heatmaps_results_sept19.json', 'r') as json_file:
    data = yaml.safe_load(json_file)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel('Sagittal Axis (cm)', fontsize=30, labelpad = 5, fontname='Times New Roman')
ax.set_xlabel('Frontal Axis (cm)', fontsize=30, labelpad = 5, fontname='Times New Roman')
# ax.set_zlabel('Height (cm)', fontsize=15, labelpad = 5, fontname='Times New Roman')
ax.set_title('Palate Pose Reachability', fontsize=35, fontname='Arial', fontweight='bold', y=1.05)

# plot average RCM reachability for all tested points
"""
rcm_points = np.empty((0,3))
averages = np.empty(len(data)-1)
for i in range(len(data)-1):
    RCM = data[i]
    rcm_points = np.vstack((rcm_points,np.array(RCM["RCM"]).reshape(1,3)))
    averages[i] = RCM["average_ratio"]
ax.scatter(rcm_points[:, 0], rcm_points[:, 1], rcm_points[:, 2], c=averages, cmap='plasma')
"""

targets = MeshObj(adf_num=0, stl_num=1, body_index=1)
targets.points *= -10


best_case = data[len(data)-1]
best_heatmap = best_case["best_heatmap"]
target_scatter = ax.scatter(targets.points[:, 1], targets.points[:, 0], c=best_heatmap, cmap='plasma')
cbar = fig.colorbar(target_scatter, pad=0, aspect=50)  # shrink scales the colorbar size
cbar.ax.tick_params(labelsize=30)
cbar.set_ticks(np.around(np.linspace(min(best_heatmap), max(best_heatmap), num = 10), decimals=2))
cbar.set_label('Reachable Pose Ratio', fontsize=30, labelpad=8)

ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)

plt.axis('scaled')

plt.grid()

plt.show()