import numpy as np
from tqdm import tqdm
import time
from scipy.spatial.transform import Rotation
from mesh_utils.mesh import MeshObj
from psm_utils.psm import PSM
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import pyplot as plt


if __name__ == '__main__':

    # set robot in 3d space
    orientation = Rotation.from_euler('XYZ', [0, 0, 0])
    translation = np.array([0, 0, 0.5]).reshape(3,1)
    transform = np.hstack((orientation.as_matrix(), translation))
    bottom = np.array([0, 0, 0, 1])
    transform = np.vstack((transform, bottom))
    psm = PSM(transform)
    #visualize_robot(psm)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)

    ax.axes.set_xlim3d(left=-0.2, right=0.2)
    ax.axes.set_ylim3d(bottom=-0.2, top=0.2)
    ax.axes.set_zlim3d(bottom=0.4, top=0.7)

    angles = np.arange(0, 360, 72)
    r, p, y = np.meshgrid(angles, angles, angles, indexing='ij')  # Creates a grid
    xyz = np.vstack([r.ravel(), p.ravel(), y.ravel()]).T  # Flatten and combine into row vectors

    xyz = np.deg2rad(xyz)
    #print(xyz)
    with tqdm(total=xyz.shape[0]) as pbar:
        for row in np.nditer(xyz, flags=['external_loop'], order='C'):
            #print(row)
            joints = psm.inverse_kinematics([0.05, 0.05, 0.45], row.tolist(), global_frame=True)
            #print(joints)
            psm.visualize_robot(ax, joint_inputs=joints)
            pbar.update(1)

    ax.set_aspect('equal', adjustable='box')
    plt.show()