import numpy as np

def vectorize_bvh_to_rotation(gesture_filename):

    with open(gesture_filename, 'r') as f:
        org = f.readlines()

    frametime = org[310].split()

    del org[0:311]

    bvh_len = len(org)

    for idx, line in enumerate(org):
        org[idx] = [float(x) for x in line.split()]

    for i in range(0, bvh_len):
        for j in range(0, int(306 / 3)):
            st = j * 3
            del org[i][st:st + 3]

    # if data is 100fps, cut it to 20 fps (every fifth line)
    # if data is approx 24fps, cut it to 20 fps (del every sixth line)
    if float(frametime[2]) == 0.0416667:
        del org[::6]
    elif float(frametime[2]) == 0.010000:
        org = org[::5]
    else:
        print("smth wrong with fps of " + gesture_filename)

    org = np.array(org)

    # Deal with x rot of joint 'spine', rot_vec[:, 2]
    # Since there are positive degree and negative degree that represents save direction
    # In the distribution learning, model will interpolate between causing error results
    for frame_idx, rot_vec_frame in enumerate(org):

        if rot_vec_frame[2] < 0:

            org[frame_idx, 2] += 360

        if rot_vec_frame[2] > 359:

            org[frame_idx, 2] -= 360

    org = np.concatenate([org[:, :27], org[:, 72:84]], axis=1)

    # print(org[:, 0], org[:, 1], org[:, 2])
    # assert 0

    # excluding fingers and hip and thigh
    # org = np.concatenate([org[:, 3:27], org[:, 72:84]], axis=1)

    # Clip rotation value to [0, 360)
    # while np.any(org < 0):
    #     org[org < 0] = org[org < 0] + 360
    # while np.any(org >= 360) :
    #     org[org >= 360] = org[org >= 360] - 360
    

    return org