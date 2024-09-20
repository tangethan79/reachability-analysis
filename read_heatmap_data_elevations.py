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


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_ylabel('Sagittal Axis (cm)', fontsize=20, labelpad = 5, fontname='Times New Roman')
ax1.set_xlabel('Frontal Axis (cm)', fontsize=20, labelpad = 5, fontname='Times New Roman')
# ax.set_zlabel('Height (cm)', fontsize=15, labelpad = 5, fontname='Times New Roman')
ax1.set_title('13cm Elevation', fontsize=25, fontname='Arial', fontweight='bold', y=1.05)

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

ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', which='minor', labelsize=20)

ax1.grid()


# best at 16 cm elevation
best_16 = 0
for case in data:
    if "RCM" in case:
        if 1.5 < case["RCM"][2] < 1.7:
            if case["average_ratio"] > best_16:
                best_16 = case["average_ratio"]
                c16 = case["heatmap"]

ax2.set_xlabel('Frontal Axis (cm)', fontsize=20, labelpad = 5, fontname='Times New Roman')
# ax.set_zlabel('Height (cm)', fontsize=15, labelpad = 5, fontname='Times New Roman')
ax2.set_title('16cm Elevation', fontsize=25, fontname='Arial', fontweight='bold', y=1.05)
target_scatter = ax2.scatter(targets.points[:, 1], targets.points[:, 0], c=c16, cmap='plasma')

ax2.set_yticklabels([])  # Remove y-axis tick labels
ax2.tick_params(axis='y', which='both', length=0)
ax2.tick_params(axis='x', which='both', labelsize=20)
ax2.grid(True)


# best at 19 cm elevation
best_19 = 0
for case in data:
    if "RCM" in case:
        if 1.8 < case["RCM"][2] < 2.0:
            if case["average_ratio"] > best_19:
                best_19 = case["average_ratio"]
                c19 = case["heatmap"]

ax3.set_xlabel('Frontal Axis (cm)', fontsize=20, labelpad = 5, fontname='Times New Roman')
# ax.set_zlabel('Height (cm)', fontsize=15, labelpad = 5, fontname='Times New Roman')
ax3.set_title('19cm Elevation', fontsize=25, fontname='Arial', fontweight='bold', y=1.05)
target_scatter = ax3.scatter(targets.points[:, 1], targets.points[:, 0], c=c19, cmap='plasma')

ax3.set_yticklabels([])  # Remove y-axis tick labels
ax3.tick_params(axis='y', which='both', length=0)
ax3.tick_params(axis='x', which='both', labelsize=20)
ax3.grid(True)


best_case = data[len(data)-1]
best_heatmap = best_case["best_heatmap"]
target_scatter = ax1.scatter(targets.points[:, 1], targets.points[:, 0], c=best_heatmap, cmap='plasma')


cbar = fig.colorbar(target_scatter, ax = ax3, pad=0.05, aspect=50)  # shrink scales the colorbar size
cbar.ax.tick_params(labelsize=20)
cbar.set_ticks(np.around(np.linspace(min(best_heatmap), max(best_heatmap), num = 10), decimals=2))
cbar.set_label('Reachable Pose Ratio', fontsize=20, labelpad=8)

ax1.axis('scaled')
ax2.axis('scaled')
ax3.axis('scaled')
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

plt.show()