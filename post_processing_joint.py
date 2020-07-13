import numpy as np
from IPython.display import Video
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits import mplot3d



def find_extrema(axis, margins, data_dict_1, data_dict_2):
    return [min(min(data_dict_1[axis + "_min"]), min(data_dict_2[axis + "_min"])) - margins, 
            max(max(data_dict_1[axis + "_max"]), max(data_dict_2[axis + "_max"])) + margins]

def plot_video_3D(plot_params_1: dict, plot_params_2: dict, video_name="video.mp4", fps=15, margins=0.2):  

    """
    Creates 3d graph video of simulation 
    """

    position_rod1 = np.array(plot_params_1["position"])
    position_rod2 = np.array(plot_params_2["position"])
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8))

    x_lim = find_extrema("x", margins, plot_params_1, plot_params_2)
    y_lim = find_extrema("y", margins, plot_params_1, plot_params_2)
    z_lim = find_extrema("z", margins, plot_params_1, plot_params_2)

    with writer.saving(fig, video_name, dpi=100):
        for time in range(1, len(plot_params_1["time"])): 
            x_1 = position_rod1[time][0]
            y_1 = position_rod1[time][1]
            z_1 = position_rod1[time][2]
            x_2 = position_rod2[time][0]
            y_2 = position_rod2[time][1]
            z_2 = position_rod2[time][2]
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            plt.plot(x_1, y_1, z_1, "-", linewidth=3, color='green')
            plt.plot(x_2, y_2, z_2, "-", linewidth=3, color='blue')

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.view_init(elev=20, azim=-80)
            writer.grab_frame()
    plt.close(fig)