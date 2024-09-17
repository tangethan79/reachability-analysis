import numpy as np
from scipy.spatial.transform import Rotation
from mesh_utils.mesh import MeshObj
from psm_utils.psm import PSM
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import json

def convert_np_to_list(data):
    if isinstance(data, dict):
        return {k: convert_np_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_to_list(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

if __name__ == '__main__':

    # figure setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=0, top=2)

    ax.set_aspect('equal', adjustable='box')

    cleft = MeshObj(adf_num=0, stl_num=0, body_index=0)
    #print(cleft.points)
    #cleft.plot_mesh(ax)
    targets = MeshObj(adf_num=0, stl_num=1, body_index=1)
    #print(targets.points)
    #targets.plot_mesh(ax)

    # set robot in 3d space
    orientation = Rotation.from_euler('XYZ', [0, 0, 0])
    translation = np.array([0.5, -0.5, 1.5]).reshape(3,1)
    transform = np.hstack((orientation.as_matrix(), translation))
    bottom = np.array([0, 0, 0, 1])
    transform = np.vstack((transform, bottom))
    psm = PSM(transform, col_mesh=cleft)

    # set up range of angles to iterate through
    angles = np.arange(0, 360, 72)
    r, p, y = np.meshgrid(angles, angles, angles, indexing='ij')  # Creates a grid
    rpy = np.vstack([r.ravel(), p.ravel(), y.ravel()]).T  # Flatten and combine into row vectors
    rpy = np.deg2rad(rpy)

    # set up list of RCM positions to iterate through
    xrange = np.arange(0.9,1.2,0.05)
    yrange = np.arange(-0.3,-0.1,0.05)
    zrange = np.arange(1.5,3,0.25)
    x,y,z = np.meshgrid(xrange,yrange,zrange, indexing='ij')
    xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    best_ratio = 0
    best_RCM = None
    best_targets = None

    num_orientations = rpy.shape[0]
    list_heatmaps = []


    with tqdm(total=xyz.shape[0]* len(targets.points)) as pbar:
        for rcm_pos in np.nditer(xyz, flags=['external_loop'], order='C'):

            ratios = np.empty(len(targets.points))
            avg = 0
            transform[0:3, 3] = rcm_pos.T
            for i in range(len(targets.points)):
                target_point = targets.points[i]
                col_free_accum = 0
                for orientation in np.nditer(rpy, flags=['external_loop'], order='C'):
                    psm.origin = transform
                    joints = psm.inverse_kinematics(target_point, rpy=orientation, global_frame=True)
                    if psm.col_check(joints):
                        col_free_accum +=1
                    #print(joints)
                    #psm.visualize_robot(ax, joint_inputs=joints)
                # check the ratio of collision free orientations for a given target point
                col_free_accum = col_free_accum/num_orientations
                ratios[i] = col_free_accum
                # add this ratio to the total
                avg += col_free_accum

            pbar.update(1)
            # average the collision free ratio across all target points
            avg = avg/len(targets.points)
            if avg > best_ratio:
                best_ratio = avg
                best_RCM = rcm_pos
                best_targets = ratios

            rcm_dict = {'RCM':rcm_pos,'average_ratio':avg,'heatmap':ratios}
            list_heatmaps.append(rcm_dict)

    best_dict = {'best_RCM':best_RCM,'best_ratio_average':best_ratio,'best_heatmap':best_targets}
    list_heatmaps.append(best_dict)

    list_heatmaps_serializable = convert_np_to_list(list_heatmaps)

    # Save to JSON file
    json_file_path = "list_heatmaps_results.json"
    with open(json_file_path, "w") as json_file:
        json.dump(list_heatmaps_serializable, json_file, indent=4)

    print(f"Saved list_heatmaps to {json_file_path}")

    targets.plot_mesh(ax,scatter_color=True,colors=best_targets)
    print(best_RCM)

    plt.show()

    """
    rcm_pos = np.array([1,0,2])
    transform[0:3, 3] = rcm_pos.T

    ratios = np.empty(len(targets.points))
    with tqdm(total=len(targets.points)) as pbar:
        for i in range(len(targets.points)):
            target_point = targets.points[i]
            col_free_accum = 0
            for orientation in np.nditer(rpy, flags=['external_loop'], order='C'):
                psm.origin = transform
                joints = psm.inverse_kinematics(target_point, rpy=orientation, global_frame=True)
                if psm.col_check(joints):
                    col_free_accum += 1
                # print(joints)
                # psm.visualize_robot(ax, joint_inputs=joints)
            # check the ratio of collision free orientations for a given target point
            col_free_accum = col_free_accum / num_orientations
            ratios[i] = col_free_accum
            pbar.update(1)


    targets.plot_mesh(ax,scatter_color=True,colors=ratios)
    print(ratios)

    plt.show()
    """
    # now need to call psm.linear_inteprolation(joints), build and prep mesh, and query tree
    # threshold query points based on radius of each region