import os
import subprocess

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import PIL


class Figure:
    def __init__(self, fig_size=540, ratio=2, dpi=300, subplots=(1, 1),
                 width_ratios=None, height_ratios=None, 
                 hspace=None, wspace=None,
                 ts=2, pad=0.2, sw=0.2,
                 minor_ticks=True,
                 theme='dark', color=None, ax_color=None,
                 grid=True):

        fig_width, fig_height = fig_size * ratio / dpi, fig_size / dpi
        fs = np.sqrt(fig_width * fig_height)

        self.fs = fs
        self.fig_size = fig_size
        self.ratio = ratio
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.subplots = subplots
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.hspace = hspace
        self.wspace = wspace
        self.ts = ts
        self.sw = sw
        self.pad = pad
        self.minor_ticks = minor_ticks
        self.grid = grid

        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        self.dpi = dpi

        # theme can only be dark or default, make a raiserror
        if not theme in ['dark', 'default']:
            raise ValueError('Theme must be "dark" or "default".')

        self.theme = theme  # This is set but not used in your provided code
        if theme == "dark":
            self.color = '#222222'
            self.ax_color = 'w'
            self.fig.patch.set_facecolor(self.color)
            plt.rcParams.update({"text.color": "white"})
        else:
            self.color = 'w'
            self.ax_color = 'k'

        if color is not None:
            self.color = color
        if ax_color is not None:
            self.ax_color = ax_color

        # GridSpec setup
        gs = mpl.gridspec.GridSpec(
            nrows=subplots[0], ncols=subplots[1], figure=self.fig,
            width_ratios=width_ratios or [1] * subplots[1],
            height_ratios=height_ratios or [1] * subplots[0],
            hspace=hspace, wspace=wspace
        )

        # Creating subplots
        self.axes = []
        for i in range(subplots[0]):
            row_axes = []
            for j in range(subplots[1]):
                ax = self.fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            self.axes.append(row_axes)

        for i in range(subplots[0]):
            for j in range(subplots[1]):
                ax = self.axes[i][j]

                ax.set_facecolor(self.color)

                for spine in ax.spines.values():
                    spine.set_linewidth(fs * sw)
                    spine.set_color(self.ax_color)

                if grid:

                    ax.grid(
                        which="major",
                        linewidth=fs * sw*0.5,
                        color=self.ax_color,
                        alpha=0.25
                    )

                ax.tick_params(
                    axis="both",
                    which="major",
                    labelsize=ts * fs,
                    size=fs * sw*5,
                    width=fs * sw*0.9,
                    pad= pad * fs,
                    top=True,
                    right=True,
                    labelbottom=True,
                    labeltop=False,
                    direction='inout',
                    color=self.ax_color,
                    labelcolor=self.ax_color
                )

                if minor_ticks == True:
                    ax.minorticks_on()

                    ax.tick_params(axis='both', which="minor", 
                    direction='inout',
                    top=True,
                    right=True,
                    size=fs * sw*2.5, 
                    width=fs * sw*0.8,
                    color=self.ax_color)

        self.axes_flat = [ax for row in self.axes for ax in row]


    def customize_axes(self, ax, ylabel_pos='left', 
                       xlabel_pos='bottom',):

        if ylabel_pos == 'left':
            labelright_bool = False
            labelleft_bool = True
        elif ylabel_pos == 'right':
            labelright_bool = True
            labelleft_bool = False

        if xlabel_pos == 'bottom':
            labeltop_bool = False
            labelbottom_bool = True
        elif xlabel_pos == 'top':
            labeltop_bool = True
            labelbottom_bool = False
        

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.ts * self.fs,
            size=self.fs * self.sw*5,
            width=self.fs * self.sw*0.9,
            pad= self.pad * self.fs,
            top=True,
            right=True,
            labelbottom=labelbottom_bool,
            labeltop=labeltop_bool,
            labelright=labelright_bool,
            labelleft=labelleft_bool,
            direction='inout',
            color=self.ax_color,
            labelcolor=self.ax_color
        )

        if self.minor_ticks == True:
            ax.minorticks_on()

            ax.tick_params(axis='both', which="minor", 
            direction='inout',
            top=True,
            right=True,
            size=self.fs * self.sw*2.5, 
            width=self.fs * self.sw*0.8,
            color=self.ax_color)

        ax.set_facecolor(self.color)

        for spine in ax.spines.values():
            spine.set_linewidth(self.fs * self.sw)
            spine.set_color(self.ax_color)

        if self.grid:

            ax.grid(
                which="major",
                linewidth=self.fs * self.sw*0.5,
                color=self.ax_color,
                alpha=0.25
            )

        return ax


    def save(self, path, bbox_inches=None, pad_inches=None):

        self.fig.savefig(path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=None)

        self.path = path

    def check_saved_image(self):

        if not hasattr(self, 'path'):
            raise ValueError('Figure has not been saved yet.')


        with Image.open(self.path) as img:
            print(img.size)
            return
        
    def show_image(self):

        if not hasattr(self, 'path'):
            raise ValueError('Figure has not been saved yet.')
        
        with Image.open(self.path) as img:
            img.show()
            return


def png_to_gif(fold, title='video', outfold=None, fps=24, 
               digit_format='04d', quality=500, max_colors=256, extension='.jpg'):

    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort()

    name = os.path.splitext(files[0])[0]
    basename = name.split('_')[0]

    ffmpeg_path = 'ffmpeg'  
    framerate = fps

    if outfold is None:
        abs_path = os.path.abspath(fold)
        parent_folder = os.path.dirname(abs_path)+'\\'
    else:
        parent_folder = outfold
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

    output_file = parent_folder + "{}.gif".format(title)

    # Create a palette with limited colors for better file size
    palette_file = parent_folder + "palette.png"
    palette_command = f'{ffmpeg_path} -i {fold}{basename}_%{digit_format}{extension} -vf "fps={framerate},scale={quality}:-1:flags=lanczos,palettegen=max_colors={max_colors}" -y {palette_file}'
    subprocess.run(palette_command, shell=True)

    # set paletteuse
    paletteuse = 'paletteuse=dither=bayer:bayer_scale=5'

    # Use the optimized palette to create the GIF
    gif_command = f'{ffmpeg_path} -r {framerate} -i {fold}{basename}_%04d{extension} -i {palette_file} -lavfi "fps={framerate},scale={quality}:-1:flags=lanczos [x]; [x][1:v] {paletteuse}" -y {output_file}'
    subprocess.run(gif_command, shell=True)


def png_to_mp4(fold, title='video', fps=36, digit_format='04d', res=None, resize_factor=1, custom_bitrate=None, extension='.jpg'):

    # Get a list of all .png files in the directory
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not files:
        raise ValueError("No PNG files found in the specified folder.")

    im = PIL.Image.open(os.path.join(fold, files[0]))
    resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resize_factor * resx)
        resy = int(resize_factor * resy)
        resx += resx % 2  # Ensuring even dimensions
        resy += resy % 2

    basename = os.path.splitext(files[0])[0].split('_')[0]

    ffmpeg_path = 'ffmpeg'
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")
    
    crf = 10  # Lower for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf scale={resx}:{resy} {output_file}'


    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)

    
def initialize_3d(fig_size=540, ratio=1, subplots=(1, 1), facecolor='#F8F9E6',
                  elev=10, azim=30, lim=1, dpi=300):

    fig_w, fig_h = fig_size * ratio, fig_size
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
        
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout='none',
    )
    fig.patch.set_facecolor(facecolor)
    fig.patch.set_facecolor('grey')

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig)
    axs = [[None] * subplots[1] for _ in range(subplots[0])]

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            axs[i][j] = fig.add_subplot(gs[i, j], projection='3d')
            ax = axs[i][j]

            ax.set_facecolor(facecolor)


            ax.view_init(elev=elev, azim=azim)

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.grid(False)
            for k, axis in enumerate([ax.xaxis, ax.yaxis, ax.zaxis]):
                axis.line.set_linewidth(0 * fs)
                axis.set_visible(False)
                axis.pane.fill = False
                axis.pane.set_edgecolor('none')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            dz = 2*lim
            dx = 2*lim
            dy = 2*lim

            ax.set_box_aspect([1,dy/dx,dz/dx])

    return fig, axs, fs