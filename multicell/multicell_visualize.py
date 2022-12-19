import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from matplotlib.patches import Rectangle

from singlecell.singlecell_functions import single_memory_projection
from multicell.graph_adjacency import lattice_square_int_to_loc
"""
COMMENTS:
    -radius seems to extend 85% of r, to intersect middle of line seg
        -eg. radius 10 means cell takes up almost 20 x slots
INPUT:
   1) n
   2) list of lists, of size n x n, containing labels (corresponding to colour)
OUTPUT: rectangular lattice with labels coloured appropriately
"""


# Constants
# =================================================
axis_buffer = 20.0
axis_length = 100.0
axis_tick_length = int(axis_length + axis_buffer)
memory_keys = [5,24]
memory_colour_dict = {5: 'blue', 24: 'red'}
fast_flag = False  # True - fast / simple plotting
#nutrient_text_flag = False  # True - plot nutrient quantity at each grid location (slow)  TODO: plot scalar at each location?


# TODO inefficient to loop over lattice each time?
def get_graph_lattice_uniproj(multicell, step, mu, use_proj=True):
    """ Builds nn x nn array of projections onto memory mu (uses multicell.data_dict)
    multicell - Multicell class object
    mu - memory to get the projection for
    use_proj - if False, use overlap instead of projection
    """
    assert multicell.graph_style == 'lattice_square'
    nn = multicell.graph_kwargs['sidelength']

    if use_proj:
        datakey = 'memory_proj_arr'
    else:
        datakey = 'memory_overlap_arr'

    # used to plot the lattice according to projection on one memory
    lattice_of_proj_vals = np.zeros((nn, nn))
    for a in range(multicell.num_cells):
        i, j = lattice_square_int_to_loc(a, nn)
        lattice_of_proj_vals[i, j] = multicell.data_dict[datakey][a, mu, step]
    return lattice_of_proj_vals


# TODO inefficient to loop over lattice each time?
def get_graph_lattice_state_ints(multicell, step):
    """ Builds nn x nn array of state integers (uses multicell.data_dict)
    multicell - Multicell class object
    """
    assert multicell.flag_state_int
    assert multicell.graph_style == 'lattice_square'
    nn = multicell.graph_kwargs['sidelength']
    # used to annotate the lattice according to unique integer rep of each node
    lattice_of_state_ints = np.zeros((nn, nn), dtype=int)
    for a in range(multicell.num_cells):
        i, j = lattice_square_int_to_loc(a, nn)
        lattice_of_state_ints[i, j] = multicell.data_dict['cell_state_int'][a, step]
    return lattice_of_state_ints


# TODO inefficient to loop over lattice each time?
def get_graph_lattice_overlaps(multicell, step, ref_node=0):
    """ Builds nn x nn array of overlaps between all cells and a reference cell
    multicell - Multicell class object
    """
    assert multicell.graph_style == 'lattice_square'
    nn = multicell.graph_kwargs['sidelength']
    # used to annotate the lattice according to unique integer rep of each node
    lattice_of_overlaps = np.zeros((nn, nn))
    for b in range(multicell.num_cells):
        i, j = lattice_square_int_to_loc(b, nn)
        lattice_of_overlaps[i, j] = multicell.cell_cell_overlap(ref_node, b, step)
    return lattice_of_overlaps


def graph_lattice_uniplotter(multicell, step, n, lattice_plot_dir, mu, use_proj=True):
    """ Plots nn x nn array of projections onto memory mu (uses multicell.data_dict)
    mu - singlecell memory to produce the plot for
    use_proj - if False, use overlap instead of projection
    """
    if use_proj:
        datatitle = 'projection'
    else:
        datatitle = 'overlap'

    simsetup = multicell.simsetup
    # generate figure data
    proj_vals = get_graph_lattice_uniproj(multicell, step, mu, use_proj=use_proj)
    # plot projection
    #colourmap = plt.get_cmap('PiYG')
    colourmap = mpl.cm.get_cmap('PiYG')  # 'PiYG' or 'Spectral'

    plt.imshow(proj_vals, cmap=colourmap, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Lattice site-wise %s onto memory %d (%s) (Step=%d)' %
              (datatitle, mu, simsetup['CELLTYPE_LABELS'][mu], step))
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # save figure
    plt.savefig(os.path.join(lattice_plot_dir, '%s%d_lattice_step%d.png' % (datatitle, mu, step)),
                dpi=max(80.0, n / 2.0))
    plt.close()
    return


def graph_lattice_projection_composite(multicell, step, cmap_vary=False, use_proj=True, fpath=None):
    """ Creates grid of lattice projections onto each memory mu (grid is ~ sqrt(P) x sqrt(P))
    use_proj - if False, use overlap instead of projection
    """
    if use_proj:
        datatitle = 'projection'
    else:
        datatitle = 'overlap'

    simsetup = multicell.simsetup
    state_int = multicell.flag_state_int
    assert multicell.graph_style == 'lattice_square'
    nn = multicell.graph_kwargs['sidelength']

    psqrt = np.sqrt(simsetup['P'])
    intceil = int(np.ceil(psqrt))
    if intceil * (intceil - 1) >= simsetup['P']:
        ncol = intceil
        nrow = intceil - 1
    else:
        ncol = intceil
        nrow = intceil
    empty_subplots = []

    # prep figure
    fig, ax = plt.subplots(nrow, ncol, squeeze=False)
    fig.set_size_inches(16, 16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Lattice %s onto p=%d memories (Step=%d)' %
                 (datatitle, simsetup['P'], step), fontsize=20)

    def get_colourmap(mem_idx):
        if cmap_vary:
            assert simsetup['P'] == 2
            from matplotlib.colors import LinearSegmentedColormap
            c3 = {0: '#9dc3e6', 1: '#ffd966'}[mem_idx]
            colours = [(0.0, 'black'), (0.5, 'white'), (1.0, c3)]
            colourmap = LinearSegmentedColormap.from_list('customcmap', colours, N=1e4)
        else:
            #colourmap = plt.get_cmap('PiYG')  # 'PiYG' or 'Spectral'
            colourmap = mpl.cm.get_cmap('PiYG')  # 'PiYG' or 'Spectral'

        return colourmap

    mu = 0  # Note mu loop to handle empty mu plots (for composite grid)
    for row in range(nrow):
        for col in range(ncol):
            if mu < simsetup['P']:
                colourmap = get_colourmap(mu)
                subax = ax[row][col]
                # plot data
                proj_vals = get_graph_lattice_uniproj(multicell, step, mu, use_proj=use_proj)

                im = subax.imshow(proj_vals, cmap=colourmap, vmin=-1, vmax=1)
                if state_int:
                    state_ints = get_graph_lattice_state_ints(multicell, step)
                    for (j, i), label in np.ndenumerate(state_ints):
                        subax.text(i, j, label, color='black', ha='center', va='center')
                # hide axis nums
                subax.set_title('%d (%s)' % (mu, simsetup['CELLTYPE_LABELS'][mu][:24]), fontsize=8)
                labels = [item.get_text() for item in subax.get_xticklabels()]
                empty_string_labels = [''] * len(labels)
                subax.set_xticklabels(empty_string_labels)
                labels = [item.get_text() for item in subax.get_yticklabels()]
                empty_string_labels = [''] * len(labels)
                subax.set_yticklabels(empty_string_labels)
                # nice gridlines
                subax.set_xticks(np.arange(-.5, nn, 1))
                subax.set_yticks(np.arange(-.5, nn, 1))
                subax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
                # add cmap if cmap_vary
                if cmap_vary:
                    #cbar = fig.colorbar(im, ax=ax.ravel().tolist(), ticks=[-1, 0, 1],
                    #                    orientation='horizontal', fraction=0.046, pad=0.04)
                    cbar = fig.colorbar(im, ax=subax, ticks=[-1, 0, 1], orientation='horizontal',
                                        fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=16)
                mu += 1
            else:
                empty_subplots.append((row, col))

    # turn off empty boxes
    for pair in empty_subplots:
        ax[pair[0], pair[1]].axis('off')
    # plot colourbar
    if not cmap_vary:
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), ticks=[-1, 0, 1],
                            orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)
    # save figure
    # save figure
    if fpath is None:
        lattice_plot_dir = multicell.io_dict['latticedir']
        fpath = os.path.join(lattice_plot_dir, 'composite_%s_lattice_step%d.png' %
                             (datatitle, step))
    plt.savefig(fpath, dpi=max(120.0, nn / 2.0))
    plt.close()
    return


def graph_lattice_reference_overlap_plotter(multicell, step, ref_node=0, fpath=None):
    """
    - ref_node: the cell to which all other cells will be overlap compared
    """
    state_int = multicell.flag_state_int

    assert multicell.graph_style == 'lattice_square'
    nn = multicell.graph_kwargs['sidelength']
    ref_site = lattice_square_int_to_loc(ref_node, nn)

    # get lattice size array of overlaps
    overlaps = get_graph_lattice_overlaps(multicell, step, ref_node=ref_node)

    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # see https://matplotlib.org/examples/color/colormaps_reference.html... used 'PiYG',
    #colourmap = plt.get_cmap('Spectral')
    colourmap = mpl.cm.get_cmap('Spectral')  # 'PiYG' or 'Spectral'

    # TODO: normalize? also use this for other lattice plot fn...
    plt.imshow(overlaps, cmap=colourmap, vmin=-1,vmax=1)
    if state_int:
        state_ints = get_graph_lattice_state_ints(multicell, step)
        for (j, i), label in np.ndenumerate(state_ints):
            plt.gca().text(i, j, label, color='black', ha='center', va='center')
    plt.colorbar()
    plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' %
              (ref_site[0], ref_site[1], step))
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, nn, 1))
    ax.set_yticks(np.arange(-.5, nn, 1))
    # mark reference
    ax.plot(ref_site[0], ref_site[1], marker='*', c='gold')
    # save figure
    if fpath is None:
        lattice_plot_dir = multicell.io_dict['latticedir']
        overlapname = 'overlapRef_%d_%d' % (ref_site[0], ref_site[1])
        if not os.path.exists(os.path.join(lattice_plot_dir, overlapname)):
            os.makedirs(os.path.join(lattice_plot_dir, overlapname))
        fpath = os.path.join(lattice_plot_dir, overlapname, 'lattice_%s_step%d.png' %
                             (overlapname, step))
    plt.savefig(fpath, dpi=max(80.0, nn / 2.0))
    plt.close()
    return
