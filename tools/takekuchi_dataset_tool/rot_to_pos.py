import numpy as np
import pyquaternion as pyq


def remove_hand_in_rot(motion):
    T = motion.shape[0]
    motion = motion.reshape(-1, 13, 3)
    motion = np.concatenate(
        [motion[:, :8], motion[:, 9:12]], axis=1)
    return motion.reshape(T, -1)


def create_hierarchy_nodes():
    """
    Create hierarchy nodes: an array of markers used in the motion capture
    Args:
        hierarchy: bvh file read in a structure
    Returns:
        nodes: array of markers to be used in motion processing
    """

    with open("tools/takekuchi_dataset_tool/hierarchy.txt", 'r') as f:
        hierarchy = f.readlines()


    joint_offsets = []
    joint_names = []

    for idx, line in enumerate(hierarchy):
        hierarchy[idx] = hierarchy[idx].split()
        if not len(hierarchy[idx]) == 0:
            line_type = hierarchy[idx][0]
            if line_type == 'OFFSET':
                offset = np.array([float(hierarchy[idx][1]), float(
                    hierarchy[idx][2]), float(hierarchy[idx][3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(hierarchy[idx][1])
            elif line_type == 'End':
                joint_names.append('End Site')

    nodes = []
    for idx, name in enumerate(joint_names):
        if idx == 0:
            parent = None
        elif idx in [3, 6, 10]:  # spine1->shoulders
            parent = 2
        else:
            parent = idx - 1

        if name == 'End Site':
            children = None
        elif idx == 0:  # hips
            children = [1]
        elif idx == 2:  # spine1
            children = [3, 6, 10]
        elif idx == 9:  # lefthand
            children = [10]
        # elif idx == 33:  # righthand
        #     children = [34, 38, 42, 46, 50]
        else:
            children = [idx + 1]

        node = dict([('name', name), ('parent', parent), ('children', children), ('offset', joint_offsets[idx]),
                     ('rel_degs', None), ('abs_qt', None), ('rel_pos', None), ('abs_pos', None)])
        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)

    return nodes


def rot_vec_to_abs_pos_vec(frames, nodes):
    """
    Transform vectors of the human motion from the joint angles to the absolute positions
    Args:
        frames: human motion in the join angles space
        nodes:  set of markers used in motion caption
    Returns:
        output_vectors : 3d coordinates of this human motion
    """
    output_lines = []

    for frame in frames:
        node_idx = 0
        for i in range(11):  # changed from 51
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            if nodes[node_idx]['name'] == 'End Site':
                node_idx = node_idx + 1
            nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
            current_node = nodes[node_idx]

            node_idx = node_idx + 1

        for start_node in nodes:
            abs_pos = np.array([0, 60, 0])
            current_node = start_node
            # = if not start_node['name'] = 'end site'
            if start_node['children'] is not None:
                for child_idx in start_node['children']:
                    child_node = nodes[child_idx]

                    child_offset = np.array(child_node['offset'])
                    qz = pyq.Quaternion(
                        axis=[0, 0, 1], degrees=start_node['rel_degs'][0])
                    qx = pyq.Quaternion(
                        axis=[1, 0, 0], degrees=start_node['rel_degs'][1])
                    qy = pyq.Quaternion(
                        axis=[0, 1, 0], degrees=start_node['rel_degs'][2])
                    qrot = qz * qx * qy
                    offset_rotated = qrot.rotate(child_offset)
                    child_node['rel_pos'] = start_node['abs_qt'].rotate(
                        offset_rotated)

                    child_node['abs_qt'] = start_node['abs_qt'] * qrot

            while current_node['parent'] is not None:

                abs_pos = abs_pos + current_node['rel_pos']
                current_node = nodes[current_node['parent']]
            start_node['abs_pos'] = abs_pos

        line = []
        for node in nodes:
            line.append(node['abs_pos'])
        output_lines.append(line)

    out = []
    for idx, line in enumerate(output_lines):
        ln = []
        for jn, _ in enumerate(line):
            ln.append(output_lines[idx][jn])
        out.append(ln)

    output_array = np.asarray(out)
    output_vectors = np.empty([len(output_array), 42])
    for idx, line in enumerate(output_array):
        output_vectors[idx] = line.flatten()
    return output_vectors


def remove_extra_joints_in_pos(motion):
    # Remove Neck to back before shoulder
    T = motion.shape[0]
    motion = motion.reshape(-1, 14, 3)
    motion = np.concatenate([
        motion[:, 0:6],
        motion[:, 7:10],
        motion[:, 11:]], axis=1)
    return motion.reshape(T, -1)


def rot2pos(motion):
    '''
    2D to 2D
    '''
    nodes = create_hierarchy_nodes()
    motion = remove_hand_in_rot(motion)
    pos = rot_vec_to_abs_pos_vec(motion, nodes)
    pos = remove_extra_joints_in_pos(pos)
    return pos