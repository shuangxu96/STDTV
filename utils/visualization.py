# -*- coding: utf-8 -*-

import scienceplots

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np

from matplotlib_inline import backend_inline

# https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py
def use_svg_display(): 
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_font_family(family='Arial'):
    plt.rc('font',family=family) 

def set_axes(ax, title=None, 
             xlabel=None, ylabel=None, 
             xlim=None, ylim=None, 
             xscale='linear', yscale='linear', # ['asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'mercator', 'symlog']
             xticks=None, yticks=None,
             xticklabel=None,yticklabel=None,
             xtickminor=False,ytickminor=False,
             legend=None, grid=False):
    """Set the axes for matplotlib."""
    # title and axes label
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel) 
    if ylabel:
        ax.set_ylabel(ylabel)
    # scale
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    # lim
    if xlim:
        ax.set_xlim(xlim) 
    if ylim:
        ax.set_ylim(ylim)
    # tick
    if xticks:
        if xticklabel:
            ax.set_xticks(xticks,labels=xticklabel,minor=xtickminor)
        else:
            ax.set_xticks(xticks,minor=xtickminor)
    if yticks:
        if yticklabel:
            ax.set_yticks(yticks,labels=yticklabel,minor=ytickminor)
        else:
            ax.set_yticks(yticks,minor=ytickminor)
  
    if legend:
        ax.legend(legend)
    
    if grid:
        ax.grid()
    
# https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py



# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
# fig, ax = plt.subplots()
# im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
#                    cmap="YlGn", cbarlabel="harvest [t/year]")
# texts = annotate_heatmap(im, valfmt="{x:.1f} t")
# fig.tight_layout()
# plt.show()

def vanilla_heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts





from matplotlib.colors import LinearSegmentedColormap

def custom_cmap(name = 'sci16', N = 100):
    colors = get_color(name)
    return LinearSegmentedColormap.from_list(name, colors, N=N)

def get_color(name = 'sci16'):
    # colors are from https://zhuanlan.zhihu.com/p/593320758
    if name == 'sci1':
        colors = [[0,0,0],
               [19,33,60],
               [252,163,17],
               [229,229,229]]
    elif name == 'sci2':
        colors = [[224,241,222],
               [223,122,94],
               [60,64,91],
               [130,178,154],
               [242,204,142]]
    elif name == 'sci3':
        colors = [[38,70,83],
               [42,157,142],
               [233,196,107],
               [243,162,97],
               [230,111,81]]
    elif name == 'sci4':
        colors = [[246,111,105],
               [254,179,174],
               [255,244,242],
               [21,151,165],
               [14,96,107],
               [255,194,75]]
    elif name == 'sci5':
        colors = [[144,201,231],
               [33,158,188],
               [19,103,131],
               [2,48,74],
               [254,183,5],
               [255,158,2],
               [250,134,0],]
    elif name == 'sci6':
        colors = [[115,186,214],
               [13,76,109],
               [3,50,80],
               [2,38,62],
               [239,65,67],
               [191,30,46],
               [196,50,63],]        
    elif name == 'sci7':
        colors = [[231,56,71],
               [240,250,239],
               [168,218,219],
               [69,123,157],
               [29,53,87]]    
    elif name == 'sci8':
        colors = [[183,181,160],
               [68,117,122],
               [69,42,61],
               [212,76,60],
               [221,108,76],
               [229,133,93],
               [238,213,183],
               ]   
    elif name == 'sci9':
        colors = [[219,49,36],
               [252,140,90],
               [255,223,146],
               [230,241,243],
               [144,190,224],
               [75,116,178]
               ]   
    elif name == 'sci16':
        colors = [[0.14901961, 0.2745098 , 0.3254902 ],
               [0.15686275, 0.44705882, 0.44313725],
               [0.16470588, 0.61568627, 0.54901961],
               [0.54117647, 0.69019608, 0.49019608],
               [0.91372549, 0.76862745, 0.41960784],
               [0.95294118, 0.63529412, 0.38039216],
               [0.90196078, 0.43529412, 0.31764706]]
    colors = np.array(colors)
    if colors.max()>1:
        colors /= 255
    return colors
