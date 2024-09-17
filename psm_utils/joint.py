import numpy as np
from typing import Literal


# THIS WAS GENERATED WITH CHATGPT ASSISTANCE
class Joint:
    def __init__(self, joint_type: Literal['revolute', 'prismatic'], DH: np.ndarray, radius: float, convention_change: bool = False, joint_limits: tuple[float, float] = None):
        self.joint_limits = joint_limits
        self.convention_change = True
        self.type = joint_type  # type is either 'revolute' or 'prismatic'
        self.DH = DH            # DH parameters (alpha, a, theta, d)
        self.radius = radius    # Collision checking radius
        self.frame = self.generate_frame()  # Automatically generate frame during initialization

    def generate_frame(self, input = None) -> np.ndarray:
        """
        Takes the DH parameters and converts them into the homogenous transformation matrix.
        DH array is expected to be of the form [theta, d, a, alpha].
        """
        alpha, a, theta, d = self.DH

        # provide option of adding input to DH table for when PSM moves
        if input is not None:
            if self.type =='revolute':
                theta = theta + input
            else:
                d = d + input

        if self.convention_change is False:
            # Create the transformation matrix using DH parameters
            T = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0,              np.sin(alpha),                np.cos(alpha),              d],
                [0,              0,                            0,                          1]
            ])

        else:
            # surgical_robotics_challenge uses a MODIFIED DH CONVENTION!!!!
            # this is not the standard homogenous transform, it is custom defined for this problem
            # this was chosen in order to mesh with their inverse kinematics code

            T = np.array([
                [np.cos(theta), -np.sin(theta), 0, a],
                [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
                [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
                [0, 0, 0, 1]])


        return T