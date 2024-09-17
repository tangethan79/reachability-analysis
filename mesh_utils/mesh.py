#mesh and model libraries
import numpy as np
from stl import mesh
from scipy import spatial
from scipy.spatial.transform import Rotation as R
from matplotlib import cm
import yaml
import os

#plotting libraries
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


class MeshObj:
    def __init__(self, adf_num=None, body_index=None, stl_num=None):
        yaml_list = ['reachability_config.yaml']
        stl_list = ['Complete_remeshed_20_dec.STL', 'target_points.STL']

        if adf_num:
            self.adf_str = yaml_list[adf_num]
        else:
            # default stl string, change as needed
            self.adf_str = yaml_list[0]
        self.adf_str = os.path.join(os.path.dirname(__file__), self.adf_str)

        if body_index:
            self.body_ind = body_index
        else:
            self.body_ind = 0

        if stl_num:
            self.stl_str = stl_list[stl_num]
        else:
            self.stl_str = stl_list[adf_num]
        self.stl_str = os.path.join(os.path.dirname(__file__), self.stl_str)

        self.get_stl()
        self.load_tree()

    def load_tree(self):
        self.tree = spatial.KDTree(self.points)
        return self.tree

    def plot_mesh(self, axes, scatter_color = False, colors = None):
        if scatter_color:
            axes.scatter(self.points[:,0],self.points[:,1],self.points[:,2],c=colors, cmap = 'plasma')
        else:
            # plotting test
            if self.body_ind == 1:
                axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mesh.vectors, facecolors='red'))
            else:
                axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mesh.vectors))
            # automatically scale to the mesh size
        return

    def get_stl(self):
        # load stl from file and get transfrom from adf
        mouth = mesh.Mesh.from_file(self.stl_str)
        self.get_pose_homog()
        mouth.transform(self.homog)
        self.mesh = mouth
        self.points = np.around(np.unique(mouth.vectors.reshape([int(mouth.vectors.size / 3), 3]), axis=0), 5)
        return self.points

    def get_pose_homog(self):

        # extract the correct transform information for the desired body
        with open(self.adf_str, 'r') as stream:
            mouth_attrib = yaml.safe_load(stream)
        bodies = mouth_attrib['bodies']

        body_string = bodies[self.body_ind]
        transform = mouth_attrib[body_string]['location']

        # take rpy values from yaml file and put them in a list for scipy spatial method
        # assume that this is in the xyz Euler angle form
        xyz_eul = [transform['orientation']['r'], transform['orientation']['p'], transform['orientation']['y']]
        r_mat = R.from_euler('xyz', xyz_eul)
        r_mat = r_mat.as_matrix()

        # extract position information and concatenate to get homogenous transform
        pos = np.array([[transform['position']['x']],
                        [transform['position']['y']],
                        [transform['position']['z']]])
        h_mat = np.hstack((r_mat, pos))
        buffer = np.array([0, 0, 0, 1])
        h_mat = np.vstack((h_mat, buffer))

        self.homog = h_mat
        # print(h_mat)
        return self.homog

    def query_tree(self, query_points):
        return self.tree.query(query_points)