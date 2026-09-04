"""# matplotlib_tools.
A set of tools for formatting and saving matplotlib figures.

Functions
---------
- save_active_fig: Saves the currently active figure to the specified filepath.
- get_color_list: Returns a predefined list of color hex codes for plotting.
- get_line_list: Returns a predefined list of line styles for plotting.
- get_color: Returns a color hex code from the predefined list based on the given index.
- get_line_style: Returns a line style based on the given index.
- unique_format: Returns a unique combination of color and line style based on the given index.
- format_plot: Formats the given matplotlib figure and axes for better visualization.
- plot_circle: Plots a circle on the given axes.
- plot_debug: Plots a simple debug plot of y vs x.

Classes
-------
- PlotSaveClass: Class to manage saving matplotlib figures with optional customization.
"""
from os import mkdir
from os.path import exists
from typing import overload
from warnings import warn

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import cos as np_cos
from numpy import linspace, ndarray
from numpy import pi as PI
from numpy import sin as np_sin


class PlotSaveClass:
    """Class to manage saving matplotlib figures with optional customization."""
    def __init__(self, fig: Figure, ax: Axes | ndarray, file_name: str | None = None) -> None:
        self.fig = fig
        self.ax = ax

        if file_name is not None:
            self.file_name = file_name
        else:
            if self.fig.get_suptitle() is not None and self.fig.get_suptitle() != "":
                file_name = self.fig.get_suptitle()
            elif isinstance(self.ax, Axes) and self.ax.get_title() != "":
                file_name = self.ax.get_title()
            else:
                file_name = "Figure"
            file_name = file_name.replace(" ", "_").replace("\n", "_").replace("$", "")
            self.file_name = file_name

    def save(self, ouput_folder: str,
             transparent: bool = False,
             close_fig: bool = True,
             overwrite: bool = False) -> None:
        """Saves the figure to the specified output folder.

        Parameters
        ----------
        ouput_folder: str
            The folder where the figure will be saved.
        transparent: bool, optional
            Whether to save the figure with a transparent background. Default is False.
        close_fig: bool, optional
            Whether to close the figure after saving. Default is True.
        overwrite: bool, optional
            Whether to overwrite an existing file with the same name. Default is False.
        """
        if len(ouput_folder.split(".")) > 1:
            warn("Output folder appears to be a file path, not a folder path." \
                 f"Ignoring the figure's FileName and saving to the provided path. {ouput_folder}",
                 stacklevel=2)
            file_name = ouput_folder.split("/")[-1].split("\\")[-1]
            ouput_folder = "/".join(
                ouput_folder.split("fileName")[:-1]) # Remove the file name from the path
        else:
            file_name = self.file_name
        if not ouput_folder.endswith("/") and not ouput_folder.endswith("\\"):
            ouput_folder += "/"
        if not exists(ouput_folder):
            mkdir(ouput_folder)
        if not overwrite:
            if exists(f"{ouput_folder}{file_name}.png"):
                print(f"File {file_name}.png already exists in {ouput_folder}")
                for i in range(0, 10):
                    input_str = input(f"Do you want to overwrite it? (y/n) Attempt {i+1}/10: ")
                    if input_str.lower() == "y":
                        break
                    elif input_str.lower() == "n":
                        print("File not overwritten. Exiting save function.")
                        return
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                else:
                    print("Maximum attempts reached. File not overwritten. Exiting save function.")
                    return
        save_active_fig(self.fig, f"{ouput_folder}{file_name}", transparent=transparent)
        if close_fig:
            try:
                self.fig.clf()
            except Exception as e: #pylint: disable=W0718
                print(f"Error occurred while trying to clear the figure: {e}")

def save_active_fig(fig: Figure, filepath: str, transparent: bool = False) -> None:
    """Saves the currently active figure to the specified filepath.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        filepath (str): The path where the figure will be saved.
    """
    fig.savefig(filepath, bbox_inches='tight', transparent=transparent)

def get_color_list(skiplist: None | list[int] = None) -> list[str]:
    """Returns a predefined list of color hex codes for plotting.
    Args:
        skiplist (None | list[int]): List of indices to skip from the default color list.
    Returns:
        list[str]: A list of color hex codes.
    """
    defualt_list = ['#0000FF',
                    '#FF0000',
                    '#00FF00',
                    '#FF9900',
                    '#800080',
                    '#222222',
                    '#00FFFF',
                    '#FFA500',
                    '#A52A2A',
                    '#808000']
    if skiplist is None:
        return defualt_list
    else:
        return [color for i, color in enumerate(defualt_list) if i not in skiplist]

def get_line_list() -> list[tuple[float, tuple[int, ...]]]:
    """Returns a predefined list of line styles for plotting.

    Returns
    -------
    line_styles:
        A list of line styles, with the linestyles formatted for use in matplotlib's plot function.
    """
    linestyle_tuple = [
        (0, ()), # Solid line
        (0, (1, 5)), # Dotted
        (0, (5, 5)), # Dashed
        (0, (3, 5, 1, 5)), # Dashdotted
        (0, (3, 5, 1, 5, 1, 5)), # Dashdotdotted

        (5, (10, 3)), # Long dash with offset
        (0, (1, 10)), # Loosely dotted
        (0, (5, 10)), # Loosely dashed
        (0, (3, 10, 1, 10)), # Loosely dashdotted
        (0, (3, 10, 1, 10, 1, 10)), # Loosely dashdotdotted

        (0, (1, 1)), # Densely dotted
        (0, (5, 1)), # Densely dashed
        (0, (3, 1, 1, 1)), # Densely dashdotted
        (0, (3, 1, 1, 1, 1, 1)) # Densely dashdotdotted
        ]
    return linestyle_tuple

def get_color(index: int, skiplist: None | list[int] = None) -> str:
    """Returns a color hex code from the predefined list based on the given index.

    Args:
        index (int): The index of the desired color.
        skiplist (None | list[int]): List of indices to skip from the default color list.
    Returns:
        str: The color hex code.
    """
    color_list = get_color_list(skiplist=skiplist)
    return color_list[index % len(color_list)]

def get_line_style(index: int) -> tuple[float, tuple[int, ...]]:
    """Returns a line style based on the given index.

    Args:
        index (int): The index of the desired line style.
    Returns:
        tuple[float, tuple[int, ...]]: The line style.
    """

    linestyle_tuple = get_line_list()
    return linestyle_tuple[index % len(linestyle_tuple)]

def unique_format(index: int) -> dict:
    """Returns a unique combination of color and line style based on the given index.

    Args:
        index (int): Thse index for which to get the unique format.
    Returns:
        dict: A dictionary containing 'color' and 'linestyle'.
    """
    n_color = len(get_color_list())
    n_line_style = len(get_line_list())
    index = index % (n_color * n_line_style)
    # Iterate through colors first, then linestyles so you have an i * j combination of both
    color_index = index % n_line_style
    linestyle_index = index // n_line_style

    color = get_color(color_index)
    linestyle = get_line_style(linestyle_index)
    return {'color': color, 'linestyle': linestyle}

@overload
def format_plot(fig: Figure,
                ax: Axes,
                aspect_ratio: float = ...,
                dims: tuple[float, float] | str = ...
                ) -> tuple[Figure, Axes]: ...
@overload
def format_plot(fig: Figure,
                ax: ndarray,
                aspect_ratio: float = ...,
                dims: tuple[float, float] | str = ...
                ) -> tuple[Figure, ndarray]: ...
def format_plot(fig: Figure,
                ax: Axes | ndarray,
                aspect_ratio: float | None = None,
                dims: tuple[float, float] | str = "default"
                ) -> tuple[Figure, Axes | ndarray]:
    """Formats the given matplotlib figure and axes for better visualization.

    Parameters
    ----------
        fig: [Figure]
            The figure to format.
        ax: [Axes | ndarray[Axes]]
            The axes to format.
        aspect_ratio: [1]
            The aspect ratio for the figure, defined as height/width. Defaults to 9/16


    Returns:
        tuple: A tuple containing the formatted figure and axes.
    """
    default_dpi = 300
    default_width = 12.8

    def format_axis(ax: Axes, n_elements: int) -> None:
        assert isinstance(n_elements, int), "n_elements must be an integer"
        assert n_elements > 0, "n_elements must be greater than 0"
        label_font_size = 24
        title_size_scalar = 1.25

        label_size = label_font_size / n_elements if n_elements > 1 else label_font_size
        title_size = label_font_size * title_size_scalar / n_elements

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(which='major',
                       length=7 / n_elements,
                       width=1.5/n_elements,
                       labelsize=label_size * 0.8)
        ax.tick_params(which='minor',
                       length=4 / n_elements,
                       width=1 / n_elements,
                       labelsize=label_size * 0.8)

        ax.set_title(ax.get_title(), fontsize=title_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)

    # Check dims is compatile with ar
    if (isinstance(dims, str | tuple) and (aspect_ratio is not None)) and (dims != "default"):
        raise ValueError("dims and AR are both specified. Please only specify one or the other.")
    if isinstance(dims, str):
        dims = dims.lower().strip()

    if dims == "default":
        aspect_ratio = 9/16
    elif dims == "4k":
        dims = (3180, 2160)
        default_dpi = 200
    elif dims == "1080p":
        dims = (1920, 1080)
        default_dpi = 200

    if aspect_ratio is not None:
        fig.set_dpi(default_dpi)
        fig.set_figwidth(default_width)
        fig.set_figheight(default_width * aspect_ratio)
    elif isinstance(dims, tuple):
        fig.set_dpi(default_dpi)
        fig.set_figwidth(dims[0] / default_dpi)
        fig.set_figheight(dims[1] / default_dpi)
    elif dims == "default":
        pass
    else:
        raise ValueError("dims must be a tuple of (width, height) or 'default'")

    fig.set_facecolor('#FFFFFFFF')

    if isinstance(ax, ndarray):
        shape = ax.shape
        if len(shape) == 2:
            nrows, ncols = shape
        elif len(shape) == 1:
            nrows, ncols = shape[0], 1
        else:
            raise ValueError("ax must be a 1D or 2D array of Axes objects")
        for sub_ax in ax.flatten():
            format_axis(sub_ax, n_elements=nrows * ncols)
    elif isinstance(ax, Axes):
        format_axis(ax, n_elements=1)
    else:
        raise TypeError(
            "ax must be either a matplotlib.axes.Axes object or a numpy.ndarray of Axes objects")

    fig.tight_layout()
    fig.show()
    return fig, ax

def plot_circle(ax: Axes,
                center: tuple[float, float] | ndarray,
                radius: float,
                theta: ndarray | tuple[float, float] = (0, 2*PI),
                **kwargs) -> None:
    """Plots a circle on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot the circle.
        center (tuple[float, float]): The (x, y) coordinates of the circle's center.
        radius (float): The radius of the circle.
        **kwargs: Additional keyword arguments to pass to the plot function.
    """
    theta = linspace(theta[0], theta[1], 100)
    x = center[0] + radius * np_cos(theta)
    y = center[1] + radius * np_sin(theta)
    ax.plot(x, y, **kwargs)

def plot_debug(x: float | ndarray,
               y: float | ndarray,
               fig: Figure | None = None,
               ax: Axes | None = None,
               x_label: str = "x",
               y_label: str = "y",
               title: str = "Debug Plot") -> None | tuple[Figure, Axes]:
    """Plots a simple debug plot of y vs x.

    Parameters
    ----------
    x: [varies]
        The x values to plot.
    y: [varies]
        The y values to plot.
    fig: [Figure]
        An optional figure to plot on. If None, a new figure will be created.
    ax: [Axes]
        An optional axes to plot on. If None, new axes will be created.

    Returns
    -------
    fig: [Figure]
        The figure containing the plot, or None if an error occurred.
    ax: [Axis]
        The axes containing the plot, or None if an error occurred.
    """
    if fig is None or ax is None:
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        else:
            raise NotImplementedError("If fig or ax is provided, both must be provided. " \
                                      "Otherwise, set both to None to create a new set.")

    try:
        ax.plot(x, y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        fig.tight_layout()
        fig, ax = format_plot(fig, ax)
        fig.show()
    except Exception as e: #pylint: disable=W0718
        print(f"An error occurred while plotting: {e}")
        return None
    return fig, ax
