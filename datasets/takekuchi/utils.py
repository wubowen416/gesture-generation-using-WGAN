from scipy.spatial.transform import Rotation as R
import numpy as np


def zxy_to_yxz(degs):
    r = R.from_euler('ZXY', degs, degrees=True)
    return r.as_euler('YXZ', degrees=True)

def transform(vector):
    num_joint = int(vector.shape[1] / 3)
    lines = []
    for frame in vector:
        line = []
        for i in range(num_joint):
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            y, x, z = zxy_to_yxz([z_deg, x_deg, y_deg])
            line.extend([x, y, z])
        lines.append(line)
    return np.array(lines)

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)