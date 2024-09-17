import numpy as np
from typing import List
from .joint import Joint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation
import math


def generate_frame(position: List[float], rpy: List[float]) -> np.ndarray:
    end_orientation = Rotation.from_euler('XYZ', rpy, degrees=False)
    end_R = end_orientation.as_matrix()

    pad = np.array([0,0,0,1], ndmin=2)
    pos = np.array(position)
    pos = pos.reshape(3,1)
    H = np.hstack((end_R,pos))
    H = np.vstack((H,pad))
    return H


def invert_frame(H: np.ndarray) -> np.ndarray:
    R = H[0:3,0:3]
    d = H[0:3,3].reshape(-1,1)

    R = np.linalg.inv(R)
    d = np.matmul(-R, d)

    pad = np.array([0, 0, 0, 1],ndmin=2)
    H_new = np.hstack((R,d))
    H_new = np.vstack((H_new, pad))
    return H_new


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)  # Calculate the magnitude (Euclidean norm)
    if norm == 0:  # To avoid division by zero
        return vector
    return vector / norm  # Normalize the vector

def get_angle(vec_a, vec_b, up_vector=None):
    vec_a = normalize(vec_a)
    vec_b = normalize(vec_b)

    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()

    cross_ab = np.cross(vec_a, vec_b)

    vdot = np.dot(vec_a, vec_b)
    vdot = vdot.astype(float)
    # Check if the vectors are in the same direction
    if 1.0 - vdot < 0.000001:
        angle = 0.0
        # Or in the opposite direction
    elif 1.0 + vdot < 0.000001:
        angle = np.pi
    else:
        angle = math.acos(vdot)

    if up_vector is not None:
        same_dir = np.sign(np.dot(cross_ab, up_vector.flatten()))
        if same_dir < 0.0:
            angle = -angle

    return angle



# THIS WAS GENERATED WITH CHATGPT ASSISTANCE
class PSM:
    def __init__(self, origin: np.ndarray):
        """
        Initialize the PSM with a list of joints and an initial origin (in world coordinates).

        :param joints: List of Joint objects that define the manipulator.
        :param origin: Initial 4x4 homogeneous transform of the base in world coordinates.
        """

        # this is in units of metres
        self.L_rcc = 0.4736 # from RCM to prismatic joint start
        self.L_tool = 0.361 # from tool roll to tool pitch
        self.L_pitch2yaw = 0.0038
        self.L_yaw2ctrlpnt = 0.0025
        self.L_tool2rcm_offset = self.L_rcc - self.L_tool


        self.joints = []  # List of Joint objects
        # note that it is convention for the first joint to have y-axis facing away from the collision geometry

        # DH parameters (alpha, a, theta, d)
        # setting up the PSM with known parameters, these match the HSC 3mm instruments
        self.joints.append(Joint('revolute', np.array([np.pi/2, 0, np.pi/2, 0]), 1.5, convention_change=True))
        self.joints.append(Joint('revolute', np.array([-np.pi/2, 0, -np.pi/2, 0]), 1.5, convention_change=True))
        self.joints.append(Joint('prismatic', np.array([np.pi/2, 0, 0, -self.L_rcc]), 1.5, convention_change=True))
        self.joints.append(Joint('revolute', np.array([0, 0, 0, self.L_tool]), 1.5, convention_change=True))
        self.joints.append(Joint('revolute', np.array([-np.pi/2, 0, -np.pi/2, 0]), 1.5, convention_change=True))
        self.joints.append(Joint('revolute', np.array([-np.pi/2, self.L_pitch2yaw, -np.pi/2, 0]), 1.5, convention_change=True))
        self.joints.append(Joint('revolute', np.array([-np.pi/2, 0, np.pi/2, self.L_yaw2ctrlpnt]), 1.5, convention_change=True))


        #self.origin = np.matmul(origin,np.array([[0, 0, -1, 0],[0, 1, 0, 0],[1, 0, 0, 0],[0, 0, 0,1]]))  # Initial homogeneous transform matrix of the base (world coordinates)
        self.origin = origin

        self.a = 2 # the ratio between free space and overlap space when calculating overlap points

        self.int_distance = []
        for i in range(len(self.joints)):
            self.int_distance.append(self.find_int_distance(i))

    def visualize_robot(self, ax, joint_inputs = None) -> None:
        if joint_inputs is not None:
            transforms = self.forward_kinematics(joint_inputs)
        else:
            joint_inputs = [0]*len(self.joints)
            transforms = self.forward_kinematics(joint_inputs)  # assume that all inputs are zero
        prev_line = np.empty(shape=[3, 0])
        for i in range(len(transforms)):
            transform = transforms[i]

            # joint origin
            if i in [0,1,2]:
                # this is for the first 2 joints (plus origin)
                ax.plot(transform[0, 3], transform[1, 3], transform[2, 3], '.r', markersize=5)
            elif i in [3,4]:
                # remaining ones are in green
                ax.plot(transform[0, 3], transform[1, 3], transform[2, 3], '.g', markersize=5)
            elif i in [5,6]:
                ax.plot(transform[0, 3], transform[1, 3], transform[2, 3], '.b', markersize=5)
            else:
                ax.plot(transform[0, 3], transform[1, 3], transform[2, 3], '.b', marker='x', markersize=10)

            prev_line = np.hstack((prev_line, transform[0:3, 3].reshape((3, 1)))).copy()
        segments = [prev_line[:, i:i + 2].T for i in range(prev_line.shape[1] - 1)]
        lc = Line3DCollection(segments, linewidths=0.5, colors='b')
        ax.add_collection3d(lc)
        return


    #function to find the distance between two interpolated points
    def find_int_distance(self, joint: int) -> float:
        r = self.joints[joint].radius
        coefficients = [(1-self.a)/12, 0, self.a * r**2, self.a * -4/3* r**3]
        roots = np.roots(coefficients)
        roots = roots.tolist()

        distance = next((x for x in roots if 0 <= x <= 2*r), None)

        return distance


    def forward_kinematics(self, joint_inputs: List[float]) -> List[np.ndarray]:
        """
        Computes the forward kinematics of the manipulator by updating the DH parameters
        of each joint based on the joint inputs and generating new transforms.

        :param joint_inputs: A list of inputs relative to baseline, one for each joint.
        :return: A list of 4x4 numpy arrays representing the transformation matrices.
        """
        transforms = [self.origin] # add origin to transforms list
        current_transform = self.origin  # Start with the initial origin


        for i, joint in enumerate(self.joints):
            # Generate the new frame for the joint after updating DH parameters
            joint.frame = joint.generate_frame(joint_inputs[i])

            # Update the current transform (multiply the previous transforms)
            current_transform = np.dot(current_transform, joint.frame)

            # Store the transform for the current joint
            transforms.append(current_transform)
        return transforms

    def inverse_kinematics(self, position: List[float], rpy: List[float] = None, global_frame: bool = False) -> List[float]:
        # this function takes the same strategy as the one in the surgical robotics challenge
        # if global_frame is True, need to convert to base robot frame first

        # rpy is in radians
        if rpy is None:
            rpy = [0,0,0]

        # this is the frame of the desired position expressed in base frame
        end_frame = generate_frame(position, rpy)

        # tranfsorm to robot frame if necessary
        if global_frame is True:
            end_frame = np.matmul(invert_frame(self.origin), end_frame)
            #print(end_frame)

        # convert local yaw frame into base one through matrix multiplication
        # note that number indicates frame in which coordinates are expressed
        yaw_7 = generate_frame([0,0,-self.L_yaw2ctrlpnt], [0,0,0])
        yaw_0 = np.matmul(end_frame, yaw_7)

        yaw_local = invert_frame(yaw_0)

        palm_normal = yaw_local[0:3,3].reshape(3,1)
        palm_normal[0,0] = 0
        palm_normal = normalize(palm_normal)

        # palm joint in global frame
        palm_from_yaw = generate_frame(self.L_pitch2yaw *  palm_normal, [0,0,0])
        palm_0 = np.matmul(yaw_0,palm_from_yaw)

        palm_0_p = palm_0[0:3,3]
        insertion_depth = np.linalg.norm(palm_0_p)
        xz_diagonal = math.sqrt(palm_0_p[0]**2 + palm_0_p[2]**2)
        j1 = math.atan2(palm_0_p[0], -palm_0_p[2])
        j2 = -math.atan2(palm_0_p[1], xz_diagonal)
        j3 = insertion_depth + self.L_tool2rcm_offset

        end_mat = Rotation.from_euler('XYZ', rpy)
        end_mat = end_mat.as_matrix()
        cross_palm_x7_0 = np.cross(end_mat[0:3,0].flatten(),(yaw_0[0:3,3]-palm_0[0:3,3]).flatten())

        # extract the 3rd frame (recall that the first frame returned is just the global origin of the robot)
        _,_,_,T_3_0,_,_,_,_ = self.forward_kinematics([j1,j2,j3,0,0,0,0])
        j4 = get_angle(cross_palm_x7_0, T_3_0[0:3,1], up_vector=-T_3_0[0:3,2])

        T_4_3 = self.joints[3].generate_frame(j4)
        T_4_0 = np.matmul(T_3_0, T_4_3)
        j5 = get_angle((yaw_0[0:3,3]-palm_0[0:3,3]), T_4_0[0:3,2], up_vector=-T_4_0[0:3,1])

        T_5_4 = self.joints[4].generate_frame(j5)
        T_5_0 = np.matmul(T_4_0, T_5_4)

        j6 = get_angle(end_frame[0:3,2],T_5_0[0:3,0], up_vector=-T_5_0[0:3,1])

        # note that last joint is just a fixed extension from yaw, has zero as input
        return [j1,j2,j3,j4,j5,j6,0]

    def linear_interpolation(self, joint_inputs) -> List[np.ndarray]:
        frames = self.forward_kinematics(joint_inputs)
        points = []

        for i in range(len(frames) - 1):
            start_frame = frames[i]
            end_frame = frames[i + 1]

            # Compute distance between joints
            start_pos = start_frame[:3, 3]
            end_pos = end_frame[:3, 3]
            distance_between_joints = np.linalg.norm(end_pos - start_pos)

            # Determine the number of interpolation points based on self.int_distance
            desired_distance = self.int_distance[i]
            if desired_distance is not None:
                num_interpolations = max(1, int(distance_between_joints / desired_distance))
                interpolation_points = np.linspace(start_pos, end_pos, num_interpolations + 1)
                points.append(interpolation_points)

        return points