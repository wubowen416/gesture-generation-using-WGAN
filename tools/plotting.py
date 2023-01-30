"""
python visualize.py --npy_path=motion.npy --save_path=motion --mode=mp4
"""


import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np

matplotlib.use('Agg')
sns.set()

lines = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [3, 6], [6, 7], [7, 8], [8, 9],
        [3, 10], [10, 11], [11, 12], [12, 13]
    ]

def plot_position_3D(motion, path):

    print(f'Plotting with length {motion.shape[0]}')
    
    def update_graph(num):
        x, y, z = motion[num][::3], -motion[num][2::3], motion[num][1::3]
        graph._offsets3d = (x, y, z)
        title.set_text('3D Test, step={}'.format(num))

        for g_l, l in zip(graph_lines, lines):
            g_l[0].set_data(x[l], y[l])
            g_l[0].set_3d_properties(z[l])
        
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    graph = ax.scatter([], [], [], c='b', s=5, alpha=1)
    graph_lines = [ax.plot([], [], [], c='b') for i, _ in enumerate(lines)]

    # ax.view_init(0, 0)
    ax.set_ylim(-50, 50)
    ax.set_zlim(0, 100)
    ax.set_xlim(-50, 50)

    anim = FuncAnimation(fig, update_graph, range(motion.shape[0]), interval=50)
    anim.save(path)

    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, help='Motion npy file')
    parser.add_argument('--save_path', type=str, default='motion.gif', help='with extension')
    args = parser.parse_args()

    motion = np.load(args.npy_path)

    plot_position_3D(motion.reshape(-1, 36), args.save_path)