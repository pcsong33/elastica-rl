import numpy as np
from IPython.display import Video
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits import mplot3d



def find_extrema(axis, margins, data_dict):
    return [min(data_dict[axis + "_min"]) - margins, max(data_dict[axis + "_max"]) + margins]

def plot_video_3D(plot_params: dict, video_name="video.mp4", fps=15, margins=0.2):  

    """
    Creates 3d graph video of simulation 
    """

    positions_over_time = np.array(plot_params["position"])
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8))

    x_lim = find_extrema("x", margins, plot_params)
    y_lim = find_extrema("y", margins, plot_params)
    z_lim = find_extrema("z", margins, plot_params)

    with writer.saving(fig, video_name, dpi=100):
        for time in range(1, len(plot_params["time"])): 
            x = positions_over_time[time][0]
            y = positions_over_time[time][1]
            z = positions_over_time[time][2]
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            plt.plot(x, y, z, "-", linewidth=3)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.view_init(elev=20, azim=-80)
            writer.grab_frame()
    plt.close(fig)