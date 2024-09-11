import numpy as np
from scipy.spatial.transform import Rotation

# a class for helping with transforms, assume initialized with units in radians and millimetres

class Frame:
    def __init__(self, rpy: list[float] = None, pos: list[float] = None):
        if rpy is None:
            rpy = [0, 0, 0]

        if pos is None:
            pos = [0, 0, 0]

        self.R = Rotation.from_euler('XYZ', rpy)
        self.p = np.ndarray(pos)
        self.p = self.p.reshape(3, 1)

        self.H = None

        # create r and position matrices then combine
        self.construct_H()

    def construct_H(self):
        self.H = np.hstack(self.R, self.p)
        fill = np.ndarray([0, 0, 0, 1])
        self.H = np.vstack(self.H, fill)

    def invert(self):
        self.R = self.R.inv()
        self.p = np.matmul(self.R,self.p)
        self.construct_H()


class PSM:
    def __init__(self, origin: list[float] = None):
        print('hi')


if __name__ == '__main__':
    print()