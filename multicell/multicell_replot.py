import proplot
#proplot.use_style('default')
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcdefaults()

import numpy as np
import os
import pickle

from multicell.multicell_lattice import read_grid_state_int
from multicell.multicell_visualize import graph_lattice_reference_overlap_plotter
from singlecell.singlecell_functions import label_to_state, state_to_label
from singlecell.singlecell_simsetup import singlecell_simsetup
from multicell.graph_helper import state_load
from multicell.graph_adjacency import lattice_square_loc_to_int, lattice_square_int_to_loc
from utils.file_io import run_subdir_setup, RUNS_FOLDER, INPUT_FOLDER


GLOBAL_DPI = 450


turquoise = [30, 223, 214]

white = [255,255,255]
soft_grey = [225, 220, 222]
soft_grey_alt1 = [206, 199, 182]
soft_grey_alt2 = [219, 219, 219]
beige = [250, 227, 199]

soft_blue = [148, 210, 226]
soft_blue_alt1 = [58, 128, 191]

soft_red = [192, 86, 64]
soft_red_alt1 = [240, 166, 144]
soft_red_alt2 = [255, 134, 113]

soft_yellow = [237, 209, 112]

soft_orange = [250, 173, 63]
soft_orange_alt1 = [248, 200, 140]

soft_green = [120, 194, 153]
sharp_green = [142, 200, 50]

soft_purple = [177, 156, 217]

soft_grey_norm = np.array(soft_grey) / 255.0

color_anchor_beige = np.array(beige) / 255.0
color_anchor_white = np.array(white) / 255.0
color_anchor = color_anchor_white


color_A_pos = np.array(soft_blue) / 255.0
#color_A_pos = np.array(soft_blue_alt1) / 255.0
color_A_neg = np.array(soft_orange) / 255.0

color_B_pos = np.array(soft_red) / 255.0
color_B_neg = np.array(soft_green) / 255.0

color_C_pos = np.array(soft_yellow) / 255.0
color_C_neg = np.array(soft_purple) / 255.0

color_AB = np.array(soft_purple) / 255.0
color_AC = np.array(soft_green) / 255.0
color_BC = np.array(soft_orange) / 255.0


# building library of 2**N random colours
def fixed_state_to_colour_map(N, show=True, shuffle=False):
    """
    # maps each state integer to a unique colour
    """
    #assert N == 9  # untested otherwise, inappropriate for large N > 14 or so
    num_states = 2 ** N

    def shift_cmap(cmap, frac):
        """Shifts a colormap by a certain fraction.

        Keyword arguments:
        cmap -- the colormap to be shifted. Can be a colormap name or a Colormap object
        frac -- the fraction of the colorbar by which to shift (must be between 0 and 1)
        """
        N = 512
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        n = cmap.name
        x = np.linspace(0, 1, N)
        out = np.roll(x, int(N * frac))
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f'{n}_s', cmap(out))
        return new_cmap


    def shuffle_state_space(seed=16):
        """
        Default seed is zero. Previously used seeds for 'multi-W' slides:
        - 3,
        """
        state_labels_shuffled = np.arange(num_states)
        np.random.seed(seed)
        np.random.shuffle(state_labels_shuffled)
        return state_labels_shuffled

    # choose a colourmap to slice (note lut = lookuptable size)
    #cmap_list = ['Spectral', 'hsv', 'nipy_spectral']
    #cmap_list = ['Spectral', 'PRGn', 'Spectral']
    cmap_list = ['YlGnBu']

    # Version 1: matplotlib
    #cmaps = [mpl.cm.get_cmap(cmap_list[i], lut=num_states) for i in range(num_cmaps)]

    # Version 2: proplot
    # docs: https://proplot.readthedocs.io/en/latest/api/proplot.constructor.Colormap.html
    #cmaps = [proplot.Colormap('Spectral', samples=num_states, shift=0, left=0.05, right=1)]
    """data1 = {
        'hue': [[0, 'red', 'red'], [1, 'blue', 'blue']],
        'saturation': [[0, 100, 100], [1, 100, 100]],
        'luminance': [[0, 100, 100], [1, 20, 20]],
    }
    #cmaps = [proplot.PerceptuallyUniformColormap(data1)]
    cmaps = [
        proplot.Colormap(
        {
            'hue': ['red', 'red-720'],
            'saturation': [80, 20],
            'luminance': [20, 100]
        },
        name='custom_cmap',
        space='hsl',
        samples=num_states,
    )
    ]"""

    # Version 3: manual proplot of one cmap shifted
    """
    deg = 33  # 66 for acton, 30 or 66 for spectral
    cstring_a = 'acton'  # Spectral acton
    cstring_b = 'Sunset'  # Spectral Sunset
    cmaps = [proplot.Colormap(cstring_a, samples=num_states, shift=0),
             proplot.Colormap(cstring_b, samples=num_states, shift=1*deg),
             proplot.Colormap(cstring_a, samples=num_states, shift=2*deg),
             proplot.Colormap(cstring_b, samples=num_states, shift=3*deg)]"""

    # Version 4: manual matplotlib to not import proplot
    deg = -24  # 66 for acton, 30 or 66 for spectral
    cstring_a = 'acton'  # Spectral acton
    cstring_b = 'Sunrise'  # Spectral Sunset
    cstring_c = 'Sunset'  # Spectral Sunset
    cmaps = [proplot.Colormap(cstring_a, samples=num_states, shift=0, left=0.10, right=1),
             proplot.Colormap(cstring_b, samples=num_states, shift=0, left=0.00, right=0.9),
             proplot.Colormap(cstring_c, samples=num_states, shift=0, left=0.00, right=0.9),
             proplot.Colormap(cstring_a, samples=num_states, shift=1*deg, left=0.10, right=1),
             proplot.Colormap(cstring_b, samples=num_states, shift=1*deg, left=0.00, right=0.9),
             proplot.Colormap(cstring_c, samples=num_states, shift=1*deg, left=0.00, right=0.9)]

    # build cmap, with each consecutive integer alternating amongst the num_cmaps
    colour_map = {}
    state_labels = list(range(num_states))
    if shuffle:
        state_labels = shuffle_state_space()
    num_cmaps = len(cmaps)
    for idx, label in enumerate(state_labels):
        cmap_choice = idx % num_cmaps
        colour_map[label] = cmaps[cmap_choice](idx)

    custom_mpl_cmap = mpl.colors.ListedColormap([colour_map[i] for i in np.arange(num_states)])
    if show:
        x = np.arange(num_states)
        y = np.arange(num_states)
        fig, ax = plt.subplots()
        sc = plt.scatter(x, y, c=y, cmap=custom_mpl_cmap)
        plt.title('fixed_state_to_colour_map() sample plot')
        plt.colorbar(ax=ax, mappable=sc)
        plt.show()
    return colour_map


def construct_cmap_from_ranked_states(state_data, N, show=True):
    """
    Converts list of state data arrays to "ranked_states"
    Ranked states is a list of integer state labels
    """
    assert N == 9

    def rank_states():
        # states and counts dict
        state_counts = {}

        for idx, X in enumerate(state_data):

            assert X.shape[0] == N
            X_as_statelabels = np.zeros(X.shape[1])
            for c in range(X.shape[1]):
                X_as_statelabels[c] = state_to_label(X[:, c])

            unique, unique_counts = np.unique(X_as_statelabels, return_counts=True)
            for idx, elem in enumerate(unique):
                if elem in state_counts.keys():
                    state_counts[elem] += unique_counts[idx]
                else:
                    state_counts[elem] = unique_counts[idx]

        # now need to create sorted list
        from operator import itemgetter
        i = 0
        ranked_states = [0] * len(list(state_counts.keys()))
        for key, value in sorted(state_counts.items(), key=itemgetter(1), reverse=True):
            ranked_states[i] = int(key)
            i += 1
        print(ranked_states)
        return ranked_states

    ranked_states = rank_states()
    num_states = len(ranked_states)

    # Version 1: manual proplot of one cmap shifted
    #cmaps = [proplot.Colormap('acton', samples=num_states, shift=0, left=0.00, right=1, reverse=True)]

    # Version 2: manual proplot of one cmap shifted
    """deg = 0  # 66 for acton, 30 or 66 for spectral
    cstring_a = 'acton'  # Spectral acton
    cstring_b = 'Sunset'  # Spectral Sunset
    cmaps = [proplot.Colormap(cstring_a, samples=num_states, shift=0),
             proplot.Colormap(cstring_b, samples=num_states, shift=1*deg),
             proplot.Colormap(cstring_a, samples=num_states, shift=2*deg),
             proplot.Colormap(cstring_b, samples=num_states, shift=3*deg)]"""

    # Version 3: manual matplotlib to not import proplot
    deg = 33  # 66 for acton, 30 or 66 for spectral
    cstring_a = 'acton'  # Spectral acton
    cstring_b = 'Sunset'  # Spectral Sunset
    cstring_c = 'Sunrise'  # Spectral Sunset
    cmaps = [proplot.Colormap(cstring_a, samples=num_states, shift=0, left=0.10, right=1),
             proplot.Colormap(cstring_b, samples=num_states, shift=0, left=0.00, right=0.9),
             proplot.Colormap(cstring_c, samples=num_states, shift=0, left=0.00, right=0.9),
             proplot.Colormap(cstring_a, samples=num_states, shift=1*deg, left=0.10, right=1),
             proplot.Colormap(cstring_b, samples=num_states, shift=1*deg, left=0.00, right=0.9),
             proplot.Colormap(cstring_c, samples=num_states, shift=1*deg, left=0.00, right=0.9)]

    # build cmap
    colour_map = {}
    num_cmaps = len(cmaps)
    for idx, label in enumerate(ranked_states):
        cmap_choice = idx % num_cmaps
        colour_map[label] = cmaps[cmap_choice](idx)
    custom_mpl_cmap = mpl.colors.ListedColormap([colour_map[ranked_states[i]]
                                                 for i in np.arange(num_states)])

    if show:
        x = np.arange(num_states)
        y = np.arange(num_states)
        fig, ax = plt.subplots()
        sc = plt.scatter(x, y, c=y, cmap=custom_mpl_cmap)
        plt.title('fixed_state_to_colour_map() sample plot')
        plt.colorbar(ax=ax, mappable=sc)
        plt.show()
    return colour_map


# fixed global colourmap for v3 of replot_modern
N = 9
FIXED_COLOURMAP = fixed_state_to_colour_map(N, show=False, shuffle=False)


def state_int_to_colour(state_int, simsetup, proj=True, noanti=True):

    def linear_interpolate(val, c2, c1=color_anchor):
        assert 0.0 <= val <= 1.0
        return (1 - val) * c1 + val * c2

    # gewt similarities i.e. proj or overlap with all cell types
    state = label_to_state(state_int, simsetup['N'], use_neg=True)
    similarities = np.dot(simsetup['XI'].T, state) / simsetup['N']
    if proj:
        similarities = np.dot(simsetup['A_INV'], similarities)

    # convert similarities to colours as rgb
    assert simsetup['P'] == 3
    if noanti:
        colour_a = linear_interpolate(max(0, similarities[0]), color_A_pos)
        colour_b = linear_interpolate(max(0, similarities[1]), color_B_pos)
        colour_c = linear_interpolate(max(0, similarities[2]), color_C_pos)
        idx_max = np.argmax(similarities)
    else:
        colour_a = linear_interpolate(np.abs(similarities[0]), [color_A_pos, color_A_neg][similarities[0] < 0])
        colour_b = linear_interpolate(np.abs(similarities[1]), [color_B_pos, color_B_neg][similarities[0] < 0])
        colour_c = linear_interpolate(np.abs(similarities[2]), [color_C_pos, color_C_neg][similarities[0] < 0])
        idx_max = np.argmax(np.abs(similarities))
    #rgb = color_a + colk  # TODO decide if want to avg the 3 colours in this fn or use all 3 with some alpha?
    #print colour_a, colour_b, colour_c
    #print proj, similarities, colour_a, colour_b, colour_c, idx_max

    sa = np
    colour_mix = [(max(0, similarities[0])*colour_a[i] + max(0, similarities[1])*colour_b[i] + max(0, similarities[2])*colour_c[i]) /(max(0, similarities[0])+max(0, similarities[1])+max(0, similarities[2])) for i in range(3)]
    return colour_mix
    #return colour_a, colour_b, colour_c, idx_max


def replot(filename, simsetup):
    grid_state_int = read_grid_state_int(filename)

    n = len(grid_state_int)
    imshowcolours_TOP = np.zeros((n, n, 3))
    imshowcolours_A = np.zeros((n, n, 3))
    imshowcolours_B = np.zeros((n, n, 3))
    imshowcolours_C = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            """
            c1, c2, c3, idx_max = state_int_to_colour(grid_state_int[i,j], simsetup)
            top_color = [c1,c2,c3][idx_max]
            imshowcolours_TOP[i,j] = top_color
            """

            imshowcolours_TOP[i, j] = state_int_to_colour(grid_state_int[i, j], simsetup)

            """
            imshowcolours_A[i, j] = c1
            imshowcolours_B[i, j] = c2
            imshowcolours_C[i, j] = c3
            """

    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    """
    plt.imshow(imshowcolours_A, alpha=0.65)
    plt.imshow(imshowcolours_B, alpha=0.65)
    plt.imshow(imshowcolours_C, alpha=0.65)
    """
    plt.imshow(imshowcolours_TOP)

    """
    if state_int:
        state_ints = get_graph_lattice_state_ints(lattice, n)
        for (j, i), label in np.ndenumerate(state_ints):
            plt.gca().text(i, j, label, color='black', ha='center', va='center')
    """
    #plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' % (ref_site[0], ref_site[1], time))
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # save figure
    plotname = os.path.dirname(filename) + os.sep + os.path.basename(filename)[:-4] + '.jpg'
    plt.savefig(plotname)
    plt.close()
    return


def replot_overlap(filename, simsetup, ref_site=(0,0), state_int=False):
    grid_state_int = read_grid_state_int(filename)
    ref_state = label_to_state(grid_state_int[0, 0], simsetup['N'], use_neg=True)
    print(grid_state_int)

    def site_site_overlap(loc):
        cellstate = label_to_state(grid_state_int[loc[0], loc[1]], simsetup['N'], use_neg=True)
        return np.dot(ref_state.T, cellstate) / float(simsetup['N'])

    # get lattice size array of overlaps
    n = grid_state_int.shape[0]
    overlaps = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            overlaps[i, j] = site_site_overlap([i, j])
    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    colourmap = plt.get_cmap('Spectral')  # see https://matplotlib.org/examples/color/colormaps_reference.html... used 'PiYG',
    plt.imshow(overlaps, cmap=colourmap, vmin=-1,vmax=1)  # TODO: normalize? also use this for other lattice plot fn...

    plt.colorbar()
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # mark reference
    #ax.plot(ref_site[0], ref_site[1], marker='*', c='gold')
    # save figure
    plotname = os.path.dirname(filename) + os.sep + os.path.basename(filename)[:-4] + '_ref00.jpg'
    plt.savefig(plotname)
    plt.close()
    return


def replot_modern(lattice_state, simsetup, sidelength, outpath, version='2', fmod='', state_int=False):
    """
    Works well for 3 celltypes, visualizing 'positive' lattice states
    v2: sum ocerlaps + anti-minima are set to white in this version
    v3: 512 unique colours -- fixed
    """

    def state_to_colour_modern_v0(state, proj=True, noanti=True):

        def linear_interpolate(val, c2, c1=color_anchor):
            eps = 1e-4
            assert 0.0 <= val <= 1.0 + eps
            return c1 + val * (c2 - c1)

        # gewt similarities i.e. proj or overlap with all cell types
        similarities = np.dot(simsetup['XI'].T, state) / simsetup['N']
        if proj:
            similarities = np.dot(simsetup['A_INV'], similarities)

        # convert similarities to colours as rgb
        assert simsetup['P'] == 3
        if noanti:
            colour_a = linear_interpolate(max(0, similarities[0]), color_A_pos)
            colour_b = linear_interpolate(max(0, similarities[1]), color_B_pos)
            colour_c = linear_interpolate(max(0, similarities[2]), color_C_pos)
            idx_max = np.argmax(similarities)
        else:
            colour_a = linear_interpolate(np.abs(similarities[0]),
                                          [color_A_pos, color_A_neg][similarities[0] < 0])
            colour_b = linear_interpolate(np.abs(similarities[1]),
                                          [color_B_pos, color_B_neg][similarities[0] < 0])
            colour_c = linear_interpolate(np.abs(similarities[2]),
                                          [color_C_pos, color_C_neg][similarities[0] < 0])
            idx_max = np.argmax(np.abs(similarities))
        # rgb = color_a + colk  # TODO decide if want to avg the 3 colours in this fn or use all 3 with some alpha?
        # print colour_a, colour_b, colour_c
        # print proj, similarities, colour_a, colour_b, colour_c, idx_max

        sa = np
        colour_mix = \
            [(max(0, similarities[0]) * colour_a[i] +
              max(0, similarities[1]) * colour_b[i] +
              max(0, similarities[2]) * colour_c[i])
             / (max(0, similarities[0]) + max(0, similarities[1]) + max(0,similarities[2]))
             for i in range(3)]
        return colour_mix
        # return colour_a, colour_b, colour_c, idx_max

    def state_to_colour_modern_v1(state, proj=True, noanti=False):
        # plot colour of max abs val for similarities

        def linear_interpolate(val, c2, c1=color_anchor):
            eps = 1e-4
            assert 0.0 <= val <= 1.0 + eps
            return c1 + val * (c2 - c1)

        # gewt similarities i.e. proj or overlap with all cell types
        similarities = np.dot(simsetup['XI'].T, state) / simsetup['N']
        if proj:
            similarities = np.dot(simsetup['A_INV'], similarities)

        # convert similarities to colours as rgb
        assert simsetup['P'] == 3

        if noanti:
            colour_a = linear_interpolate(max(0, similarities[0]), color_A_pos)
            colour_b = linear_interpolate(max(0, similarities[1]), color_B_pos)
            colour_c = linear_interpolate(max(0, similarities[2]), color_C_pos)
            idx_max = np.argmax(similarities)
        else:
            colour_a = linear_interpolate(
                np.abs(similarities[0]),
                [color_A_pos, color_A_neg][similarities[0] < 0])
            colour_b = linear_interpolate(
                np.abs(similarities[1]),
                [color_B_pos, color_B_neg][similarities[0] < 0])
            colour_c = linear_interpolate(
                np.abs(similarities[2]),
                [color_C_pos, color_C_neg][similarities[0] < 0])
            idx_max = np.argmax(np.abs(similarities))
        # rgb = color_a + colk  # TODO decide if want to avg the 3 colours in this fn or use all 3 with some alpha?
        # print colour_a, colour_b, colour_c
        # print proj, similarities, colour_a, colour_b, colour_c, idx_max

        colour_max = [colour_a, colour_b, colour_c][idx_max]

        colour_mix = colour_max
        return colour_mix
        # return colour_a, colour_b, colour_c, idx_max

    def state_to_colour_modern_v2(state, proj=True, noanti=True):
        # plot colour of max abs val for similarities -- preset colours if some are equal
        assert noanti  # TODO determine how to blend mixtures of + and - states

        def linear_interpolate(val, c2, c1=color_anchor):
            eps = 1e-4
            if not(0.0 <= val <= 1.0 + eps):
                print('WARNING')
                print(val)
            assert 0.0 <= val <= 1.0 + eps
            cout = c1 + val * (c2 - c1)
            return cout

        # gewt similarities i.e. proj or overlap with all cell types
        similarities = np.dot(simsetup['XI'].T, state) / simsetup['N']
        if proj:
            similarities = np.dot(simsetup['A_INV'], similarities)

        maxval = np.max(np.abs(similarities))

        #winners = np.argwhere(np.abs(similarities) == maxval).flatten()
        winners_check = np.isclose(np.abs(similarities), maxval).flatten()
        winners = [i for i,val in enumerate(winners_check) if val]
        #print(winners, winners2)

        #print('any1', similarities, maxval, winners)

        # convert similarities to colours as rgb
        assert simsetup['P'] == 3

        if noanti:
            colour_a = linear_interpolate(max(0, similarities[0]), color_A_pos)
            colour_b = linear_interpolate(max(0, similarities[1]), color_B_pos)
            colour_c = linear_interpolate(max(0, similarities[2]), color_C_pos)
        else:
            colour_a = linear_interpolate(
                np.abs(similarities[0]),
                [color_A_pos, color_A_neg][similarities[0] < 0])
            colour_b = linear_interpolate(
                np.abs(similarities[1]),
                [color_B_pos, color_B_neg][similarities[0] < 0])
            colour_c = linear_interpolate(
                np.abs(similarities[2]),
                [color_C_pos, color_C_neg][similarities[0] < 0])
        # rgb = color_a + colk  # TODO decide if want to avg the 3 colours in this fn or use all 3 with some alpha?
        # print colour_a, colour_b, colour_c
        # print proj, similarities, colour_a, colour_b, colour_c, idx_max

        colour_options = [colour_a, colour_b, colour_c]
        winners = np.sort(winners)
        winners_pruned = [r for r in winners if similarities[r] > 0]

        if len(winners_pruned) == 0:
            # in this case all the overlaps are negative -- set it to white
            colour_mix = color_anchor_white
        elif len(winners_pruned) == 1:
            colour_mix = colour_options[winners_pruned[0]]
        elif len(winners_pruned) == 2:
            #magnitude = 0.5
            magnitude = similarities[winners_pruned[0]]
            #magnitude = np.sqrt(similarities[winners[0]])
            if winners_pruned[0] == 0 and winners_pruned[1] == 1:
                colour_mix = linear_interpolate(magnitude, color_AB, c1=color_anchor_white)
            elif winners_pruned[0] == 0 and winners_pruned[1] == 2:
                colour_mix = linear_interpolate(magnitude, color_AC, c1=color_anchor_white)
            else:
                colour_mix = linear_interpolate(magnitude, color_BC, c1=color_anchor_white)
        else:
            assert len(winners_pruned) == 3
            val = similarities[winners_pruned[0]]
            morph_val = max(0, val)
            morph_val = morph_val ** (0.33)
            colour_mix = linear_interpolate(morph_val, soft_grey_norm, c1=color_anchor_white)

        #for idx, mu_idx in enumerate(winners):
        #    colour = colour_options[mu_idx]
        #    colour_mix += colour
        #colour_mix = colour_mix / num_winners

        return colour_mix
        # return colour_a, colour_b, colour_c, idx_max

    def state_to_colour_modern_v3(state, proj=None, noanti=None):
        # assign a unique colour to each state based on a colourmap
        label = state_to_label(cellstate)
        unique_colour = FIXED_COLOURMAP[label]
        return unique_colour[0:3]

    if version == '0':
        state_to_colour_modern = state_to_colour_modern_v0
        fmod += '_v0'
    elif version == '1':
        state_to_colour_modern = state_to_colour_modern_v1
        fmod += '_v1'
    elif version == '2':
        state_to_colour_modern = state_to_colour_modern_v2
        fmod += '_v2'
    else:
        assert version == '3'
        fmod += '_v3'
        state_to_colour_modern = state_to_colour_modern_v3

    n = sidelength
    imshowcolours_TOP = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            grid_loc_to_idx = lattice_square_loc_to_int((i,j), sidelength)
            cellstate = lattice_state[:, grid_loc_to_idx]
            imshowcolours_TOP[i, j] = state_to_colour_modern(cellstate, simsetup)
    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.imshow(imshowcolours_TOP, alpha=1.0)

    if state_int:
        num_cells = lattice_state.shape[1]
        for k in range(num_cells):
            cellstate = lattice_state[:, k]
            label = state_to_label(cellstate)
            i, j = lattice_square_int_to_loc(k, n)
            plt.gca().text(j, i, label, color='black', ha='center', va='center')

    #plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' % (ref_site[0], ref_site[1], time))
    # draw gridlines
    ax = plt.gca()
    #plt.axis('off')  @ no grid can look nice
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    #ax.set_xticks([], [])
    #ax.set_yticks([], [])
    xticks = np.arange(-.5, n, 1)
    yticks = np.arange(-.5, n, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_ticklabels(['' for _ in xticks])
    ax.yaxis.set_ticklabels(['' for _ in yticks])

    # save figure
    outpath = outpath + fmod + '.jpg'
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    return


def replot_graph_lattice_reference_overlap_plotter(X, sidelength, outpath, fmod='', ref_node=0,
                                                   state_int=False):
    """
    - ref_node: the cell to which all other cells will be overlap compared
    """
    nn = sidelength

    num_cells = X.shape[1]
    assert nn**2 == num_cells
    ref_site = lattice_square_int_to_loc(ref_node, nn)

    # get lattice size array of overlaps
    ref_state = X[:, ref_node]
    ref_overlaps = np.dot(X.T, ref_state) / X.shape[0]
    overlaps_grid = np.zeros((nn,nn))
    for i in range(nn):
        for j in range(nn):
            node_idx = lattice_square_loc_to_int((i,j), nn)
            overlaps_grid[i,j] = ref_overlaps[node_idx]

    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # see https://matplotlib.org/examples/color/colormaps_reference.html... used 'PiYG',
    cmap = 'Spectral'  # Spectral RdYlBu
    fmod += '_%s' % cmap
    colourmap = plt.get_cmap(cmap)

    #plt.colorbar()
    #plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' %
    #          (ref_site[0], ref_site[1], step))

    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    xticks = np.arange(-.5, nn, 1)
    yticks = np.arange(-.5, nn, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_ticklabels(['' for _ in xticks])
    ax.yaxis.set_ticklabels(['' for _ in yticks])

    # mark reference
    ax.plot(ref_site[0], ref_site[1], marker='*', c='gold', markersize=28)
    plt.imshow(overlaps_grid, cmap=colourmap, vmin=-1,vmax=1)

    # mark reference
    ax.plot(ref_site[0], ref_site[1], marker='*', c='gold')

    if state_int:
        num_cells = X.shape[1]
        for k in range(num_cells):
            cellstate = X[:, k]
            label = state_to_label(cellstate)
            i, j = lattice_square_int_to_loc(k, nn)
            plt.gca().text(j, i, label, color='black', ha='center', va='center')

    # save figure
    outpath = outpath + fmod + '.jpg'
    plt.savefig(outpath, dpi=max(80.0, nn / 2.0), bbox_inches='tight')
    plt.close()
    return


def replot_scatter_circle(lattice_state, simsetup, sidelength, outpath, fmod='', state_int=False):
    """
    Full info morphology plot with grid of 9 genes as cilia on circle
    """

    def state_to_colour_and_morphology(state, simsetup):
        """
        # assign a unique colour to each state based on a colourmap
        cellstate_01 = ((cellstate + 1) / 2).astype(int)
        cellstate_brief = str(cellstate_01[2]) + str(cellstate_01[5]) + str(cellstate_01[8])

        # eight handpicked colours based on combinations of encoded celltypes
        color_dict_brief = {
            '000': soft_grey_norm,        # grey (all off)
            '100': color_A_pos,           # type A - blue
            '010': color_B_pos,           # type B - red
            '001': color_C_pos,           # type C - yellow
            '101': color_AC,              # type A+C - green
            '011': color_BC,              # type B+C - orange
            '110': color_AB,              # type A+B - purple
            '111': color_anchor_white,    # white (all on)
        }

        unique_colour = color_dict_brief[cellstate_brief]
        """
        genes = [0,1,2, 3,4,5, 6,7,8]
        cellstate_brief = state[genes]
        label = state_to_label(cellstate_brief)
        unique_colour = FIXED_COLOURMAP[label]
        return unique_colour[0:3]

    n = sidelength
    assert n == 20  # redo params for n 10 visualization
    x = np.zeros(n ** 2)
    y = np.zeros(n ** 2)
    colors = np.zeros((n**2, 3))
    markers = [0] * (n**2)
    for i in range(n):
        for j in range(n):
            grid_loc_to_idx = lattice_square_loc_to_int((i,j), sidelength)
            cellstate = lattice_state[:, grid_loc_to_idx]
            colors[grid_loc_to_idx, :] = state_to_colour_and_morphology(cellstate, simsetup)
            x[grid_loc_to_idx] = j
            y[grid_loc_to_idx] = n - i

    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def get_cell_mask(gene_idx):
        mask = lattice_state[gene_idx, :] == 1
        return mask

    eps = 0.3
    lw = 2.5
    boxsize = 850   # 600, 750, 850
    mainsize = 350  # 350
    trisize = 225   # 225
    appendage_style = 2  # 1
    appendage_z = 2
    t_series = [0] * 9
    angles = [40, 0, -40, -80, -120, -160, -200, -240, -280]
    for idx in range(9):
        t_mod = mpl.markers.MarkerStyle(marker=appendage_style)
        t_mod._transform = t_mod.get_transform().rotate_deg(angles[idx])
        t_series[idx] = t_mod

    # center circle each cell
    plt.scatter(x, y, marker='o', c=colors, alpha=1.0, s=mainsize, ec='k', zorder=5)
    # outer square with alpha
    plt.scatter(x, y, marker='s', c=colors, alpha=0.4, s=boxsize, ec='k', zorder=1)
    # gene 0, 1 mask for celltype A: up/down appendage
    for g in range(simsetup['N']):
        maskg = get_cell_mask(g)
        plt.scatter(x[maskg], y[maskg], marker=t_series[g], c=colors[maskg], alpha=1.0,
                    s=trisize, ec='k', zorder=appendage_z, linewidths=lw)

    if state_int:
        assert 1==2
        num_cells = lattice_state.shape[1]
        for k in range(num_cells):
            cellstate = lattice_state[:, k]
            label = state_to_label(cellstate)
            i, j = lattice_square_int_to_loc(k, n)
            plt.gca().text(j, i, label, color='black', ha='center', va='center')

    # draw gridlines
    ax = plt.gca()
    plt.axis('off')  # no grid can look nice
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    #ax.set_xticks([], [])
    #ax.set_yticks([], [])
    xticks = np.arange(-.5, n, 1)
    yticks = np.arange(-.5, n, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_ticklabels(['' for _ in xticks])
    ax.yaxis.set_ticklabels(['' for _ in yticks])

    # save figure
    plt.savefig(outpath + fmod + '.jpg', bbox_inches='tight')
    plt.savefig(outpath + fmod + '.pdf', bbox_inches='tight')
    plt.close()
    return


def replot_scatter_tri(lattice_state, simsetup, sidelength, outpath, fmod='', state_int=False):
    """
Full info morphology plot with 9 genes as triangle cilia
    """

    def state_to_colour_and_morphology(state, simsetup):
        """
        # assign a unique colour to each state based on a colourmap
        cellstate_01 = ((cellstate + 1) / 2).astype(int)
        cellstate_brief = str(cellstate_01[2]) + str(cellstate_01[5]) + str(cellstate_01[8])

        # eight handpicked colours based on combinations of encoded celltypes
        color_dict_brief = {
            '000': soft_grey_norm,        # grey (all off)
            '100': color_A_pos,           # type A - blue
            '010': color_B_pos,           # type B - red
            '001': color_C_pos,           # type C - yellow
            '101': color_AC,              # type A+C - green
            '011': color_BC,              # type B+C - orange
            '110': color_AB,              # type A+B - purple
            '111': color_anchor_white,    # white (all on)
        }

        unique_colour = color_dict_brief[cellstate_brief]
        """
        genes = [0,1,2, 3,4,5, 6,7,8]
        cellstate_brief = state[genes]
        label = state_to_label(cellstate_brief)
        unique_colour = FIXED_COLOURMAP[label]
        return unique_colour[0:3]

    n = sidelength
    assert n == 20  # redo params for n 10 visualization
    x = np.zeros(n ** 2)
    y = np.zeros(n ** 2)
    colors = np.zeros((n**2, 3))
    markers = [0] * (n**2)
    for i in range(n):
        for j in range(n):
            grid_loc_to_idx = lattice_square_loc_to_int((i,j), sidelength)
            cellstate = lattice_state[:, grid_loc_to_idx]
            colors[grid_loc_to_idx, :] = state_to_colour_and_morphology(cellstate, simsetup)
            x[grid_loc_to_idx] = j
            y[grid_loc_to_idx] = n - i

    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    """unique_markers = np.unique(markers)
    for um in unique_markers:
        mask = markers == um
        plt.scatter(x[mask], y[mask], marker=um, c=colors[mask], alpha=1.0, s=10)"""

    def get_cell_mask(gene_idx):
        mask = lattice_state[gene_idx, :] == 1
        return mask

    eps = 0.1
    lw = 2.5
    boxsize = 990   # 600, 750, 850, at 990 it forms grey grid
    mainsize = 300  # 350
    trisize = 100   # 225
    appendage_style = 2  # 1
    appendage_z = 2
    t_series = [0] * 9
    angles = [60, 60, 60, -60, -60, -60, -180, -180, -180]
    for idx in range(9):
        t_mod = mpl.markers.MarkerStyle(marker=appendage_style)
        t_mod._transform = t_mod.get_transform().rotate_deg(angles[idx])
        t_series[idx] = t_mod


    # center circle each cell
    plt.scatter(x, y, marker='^', c=colors, alpha=1.0, s=mainsize, ec='k', zorder=5)
    # outer square with alpha
    plt.scatter(x, y, marker='s', c=colors, alpha=0.4, s=boxsize, ec='k', zorder=1)
    # gene 0, 1 mask for celltype A: up/down appendage
    mask0 = get_cell_mask(0)
    mask1 = get_cell_mask(1)
    mask2 = get_cell_mask(2)
    x0, x1, x2 = -0.15, -0.08, 0
    y0, y1, y2 = -0.2, -0.05, +0.1
    plt.scatter(x[mask0]+x0, y[mask0]+y0, marker=t_series[0], c=colors[mask0], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    plt.scatter(x[mask1]+x1, y[mask1]+y1, marker=t_series[1], c=colors[mask1], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    plt.scatter(x[mask2]+x2, y[mask2]+y2, marker=t_series[2], c=colors[mask2], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    # gene 3, 4 mask for celltype B: left/right appendage
    mask3 = get_cell_mask(3)
    mask4 = get_cell_mask(4)
    mask5 = get_cell_mask(5)
    x3, x4, x5 = -x2, -x1, -x0
    y3, y4, y5 = y2, y1, y0
    plt.scatter(x[mask3]+x3, y[mask3]+y3, marker=t_series[3], c=colors[mask3], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    plt.scatter(x[mask4]+x4, y[mask4]+y4, marker=t_series[4], c=colors[mask4], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    plt.scatter(x[mask5]+x5, y[mask5]+y5, marker=t_series[5], c=colors[mask5], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    # gene 6, 7 mask for celltype C: membrane/circle interior
    mask6 = get_cell_mask(6)
    mask7 = get_cell_mask(7)
    mask8 = get_cell_mask(8)
    x6, x7, x8 = +0.16, -0, -0.16
    y6, y7, y8 = -0.2, -0.2, -0.2
    plt.scatter(x[mask6]+x6, y[mask6]+y6, marker=t_series[6], c=colors[mask6], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    plt.scatter(x[mask7]+x7, y[mask7]+y7, marker=t_series[7], c=colors[mask7], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)
    plt.scatter(x[mask8]+x8, y[mask8]+y8, marker=t_series[8], c=colors[mask8], alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw)

    if state_int:
        assert 1==2
        num_cells = lattice_state.shape[1]
        for k in range(num_cells):
            cellstate = lattice_state[:, k]
            label = state_to_label(cellstate)
            i, j = lattice_square_int_to_loc(k, n)
            plt.gca().text(j, i, label, color='black', ha='center', va='center')

    #plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' % (ref_site[0], ref_site[1], time))
    # draw gridlines
    ax = plt.gca()
    plt.axis('off')  # no grid can look nice
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    #ax.set_xticks([], [])
    #ax.set_yticks([], [])
    xticks = np.arange(-.5, n, 1)
    yticks = np.arange(-.5, n, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_ticklabels(['' for _ in xticks])
    ax.yaxis.set_ticklabels(['' for _ in yticks])

    # save figure
    plt.savefig(outpath + fmod + '.jpg', bbox_inches='tight')
    plt.savefig(outpath + fmod + '.pdf', bbox_inches='tight')
    plt.close()
    return


def replot_scatter_dots(lattice_state, sidelength, outpath,
                        fmod='', state_int=False, cmap=None, title=None,
                        ext=['.jpg', '.svg'], rasterized=True):
    """
    Full info morphology plot with grid of 9 genes as dots
    Assumes lattice state is 2D array of num_genes (per cell) x num_cells, e.g. 9 x 100 not 900
    """

    if cmap is None:
        cmap = FIXED_COLOURMAP

    def state_to_colour_and_morphology(state):
        """
        # assign a unique colour to each state based on a colourmap
        cellstate_01 = ((cellstate + 1) / 2).astype(int)
        cellstate_brief = str(cellstate_01[2]) + str(cellstate_01[5]) + str(cellstate_01[8])

        # eight handpicked colours based on combinations of encoded celltypes
        color_dict_brief = {
            '000': soft_grey_norm,        # grey (all off)
            '100': color_A_pos,           # type A - blue
            '010': color_B_pos,           # type B - red
            '001': color_C_pos,           # type C - yellow
            '101': color_AC,              # type A+C - green
            '011': color_BC,              # type B+C - orange
            '110': color_AB,              # type A+B - purple
            '111': color_anchor_white,    # white (all on)
        }

        unique_colour = color_dict_brief[cellstate_brief]
        """
        genes = [0,1,2, 3,4,5, 6,7,8]
        cellstate_brief = state[genes]
        #cellstate_brief = [1,1,1, -1, -1, -1, -1, -1, -1]
        #cellstate_brief = [-1,-1,-1, 1, 1, 1, -1, -1, -1]
        #cellstate_brief = [-1,-1,-1, -1, -1, -1, 1, 1, 1]

        label = state_to_label(cellstate_brief)
        unique_colour = cmap[label]
        return unique_colour[0:3]

    n = sidelength
    x = np.zeros(n ** 2)
    y = np.zeros(n ** 2)
    colors = np.zeros((n**2, 3))
    for i in range(n):
        for j in range(n):
            grid_loc_to_idx = lattice_square_loc_to_int((i,j), sidelength)
            cellstate = lattice_state[:, grid_loc_to_idx]
            colors[grid_loc_to_idx, :] = state_to_colour_and_morphology(cellstate)
            x[grid_loc_to_idx] = j
            y[grid_loc_to_idx] = n - i

    # plot
    #fig = plt.figure(figsize=(12, 12))
    #fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    #ax = fig.add_axes([0, 0, 1, 1])  # position: left, bottom, width, height
    #ax.set_axis_off()
    fig, ax = plt.subplots(figsize=(12, 12), dpi=GLOBAL_DPI)
    ax.set_axis_off()

    def get_cell_mask(gene_idx):
        mask = lattice_state[gene_idx, :] == 1
        return mask

    # plot - detailed settings
    assert n in [2, 10, 20]
    if n == 10:
        box_lw = 2*1.5
        eps = 0.25
        lw = 2*2
        boxsize = 4*1800  # 600, 750, 850, at 990 it forms grey grid
        trisize = 4*50  # 225
        lw_eps = 0.05
        fontsize = 24

    elif n == 2:
        box_lw = 4*1.5
        eps = 0.3
        lw = 4*2
        boxsize = 95*1800  # 600, 750, 850, at 990 it forms grey grid
        trisize = 30*50  # 225
        lw_eps = 0.05
        fontsize = 24

    else:
        assert n == 20
        box_lw = 1.5
        eps = 0.25
        lw = 2
        boxsize = 1800  # 600, 750, 850, at 990 it forms grey grid
        trisize = 50  # 225
        lw_eps = 0.05
        fontsize = 24

    # create gene markers (the three celltype block functionality is no longer used, now 9 dots)
    appendage_style = 'o'  # 1
    appendage_z = 2
    t_series = [0] * 9
    for idx in range(9):
        t_mod = mpl.markers.MarkerStyle(marker=appendage_style)
        t_series[idx] = t_mod

    # outer square with alpha (orig 0.4 alpha)
    plt.scatter(x, y, marker='s', c=colors, alpha=1.0, s=boxsize,
                ec='k', zorder=1, lw=box_lw, rasterized=rasterized)
    # gene 0, 1 mask for celltype A: originally up/down appendage
    mask0 = get_cell_mask(0)
    mask1 = get_cell_mask(1)
    mask2 = get_cell_mask(2)
    x0, x1, x2 = -eps, 0, +eps
    y0, y1, y2 = +eps, +eps, +eps
    plt.scatter(x[mask0]+x0, y[mask0]+y0, marker=t_series[0], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    plt.scatter(x[mask1]+x1, y[mask1]+y1, marker=t_series[1], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    plt.scatter(x[mask2]+x2, y[mask2]+y2, marker=t_series[2], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    # gene 3, 4 mask for celltype B: originally left/right appendage
    mask3 = get_cell_mask(3)
    mask4 = get_cell_mask(4)
    mask5 = get_cell_mask(5)
    x3, x4, x5 = x0, x1, x2
    y3, y4, y5 = 0, 0, 0
    plt.scatter(x[mask3]+x3, y[mask3]+y3, marker=t_series[3], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    plt.scatter(x[mask4]+x4, y[mask4]+y4, marker=t_series[4], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    plt.scatter(x[mask5]+x5, y[mask5]+y5, marker=t_series[5], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    # gene 6, 7 mask for celltype C: originally membrane/circle interior
    mask6 = get_cell_mask(6)
    mask7 = get_cell_mask(7)
    mask8 = get_cell_mask(8)
    x6, x7, x8 = x0, x1, x2
    y6, y7, y8 = -eps, -eps, -eps
    plt.scatter(x[mask6]+x6, y[mask6]+y6, marker=t_series[6], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    plt.scatter(x[mask7]+x7, y[mask7]+y7, marker=t_series[7], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)
    plt.scatter(x[mask8]+x8, y[mask8]+y8, marker=t_series[8], c='white', alpha=1.0,
                s=trisize, ec='k', zorder=appendage_z, linewidths=lw, rasterized=rasterized)

    if state_int:
        num_cells = lattice_state.shape[1]
        for k in range(num_cells):
            cellstate = lattice_state[:, k]
            label = state_to_label(cellstate)
            i, j = lattice_square_int_to_loc(k, n)
            xc = j
            yc = n - i
            #plt.gca().text(j, i, label, color='black', ha='center', va='center')  # bugfix below
            plt.gca().text(xc, yc, '%s' % label,
                           color='black', ha='center', va='center')

    if title is not None:
        plt.title(title, fontsize=fontsize)
    # draw gridlines
    ax = plt.gca()
    plt.axis('off')  # no grid can look nice
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    #ax.set_xticks([], [])
    #ax.set_yticks([], [])
    xticks = np.arange(-.5, n, 1)
    yticks = np.arange(-.5, n, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_ticklabels(['' for _ in xticks])
    ax.yaxis.set_ticklabels(['' for _ in yticks])

    # this crops border
    plt.xlim(-0.5 - lw_eps, n - 0.5 + lw_eps)
    plt.ylim( 0.5 - lw_eps, n + 0.5 + lw_eps)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # unsure
    plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=1.0)                            # maybe remove line
    # save figure
    if title is None:
        bbox_inches = None
    else:
        bbox_inches = 'tight'
    if isinstance(ext, list):
        for ext_str in ext:
            assert ext_str[0] == '.'
            plt.savefig(outpath + fmod + ext_str, bbox_inches=bbox_inches, dpi=GLOBAL_DPI)
    else:
        plt.savefig(outpath + fmod + ext, bbox_inches=bbox_inches, dpi=GLOBAL_DPI)
    plt.close()
    return


def translate_lattice_state(X, sidelength, down=0, right=0):
    nn = sidelength
    num_cells = X.shape[1]
    X_translated = np.zeros_like(X)
    for k in range(num_cells):
        loc_orig = lattice_square_int_to_loc(k, nn)
        loc_new = [(loc_orig[0] + down) % nn,
                   (loc_orig[1] + right) % nn]
        k_new = lattice_square_loc_to_int(loc_new, nn)
        X_translated[:, k_new] = X[:, k]

    return X_translated


def plot_tissue_given_agg_idx(
        data_subdict, agg_index, fmod, outdir,
        state_int=False, smod_last=True, title=None):
    """
    Used and designed within explore_aligned.ipynb
    Args:
    - data_subdict: data_subdict (stores metadata, embedding) from parent notebook dict termed 'embedded_datasets'
    - settings_alignment: dict of settings for the notebook,
    - agg_index: selects the tissue state (int in range(0, num_runs))
    """
    manyruns_path = data_subdict['path']
    multicell_template = data_subdict['multicell_template']

    # Load and replace W in the multicell_template
    """
    agg_datadir = manyruns_path + os.sep + 's%d' % agg_index
    W_LOAD = np.loadtxt(agg_datadir + os.sep + 'simsetup' + os.sep + 'matrix_W.txt', delimiter=',')
    #print(W_LOAD)  # compare vs template W ?
    multicell.matrix_W = W_LOAD
    multicell.simsetup['FIELD_SEND'] = W_LOAD"""

    # constants
    num_cells = multicell_template.num_cells
    num_genes = multicell_template.num_genes
    simsetup = multicell_template.simsetup
    sidelength = int(np.sqrt(num_cells)); assert sidelength ** 2 == num_cells

    # switchable settings
    if smod_last:
        smod = '_last' # oldstyle
    else:
        smod = ''      # newstyle
    #if step is not None:
    #    smod = '_%d' % step

    # load and reshape desired tissue state
    X_state = data_subdict['data'][agg_index, :].copy()
    X_state = X_state.reshape(num_cells, num_genes)

    #outpath_ref = outdir + os.sep + 'agg%d_ref0' % agg_index
    #replot_graph_lattice_reference_overlap_plotter(
    #    X_state.T, sidelength, outpath_ref, fmod=fmod, ref_node=0)

    outpath = outdir + os.sep + 'agg%d_modern' % agg_index
    replot_modern(X_state.T, simsetup, sidelength, outpath,
                  version='3', fmod=fmod, state_int=state_int)

    # plot option 3) using replot_scatter_dots
    outpath = outdir + os.sep + 'agg%d_scatter' % agg_index
    replot_scatter_dots(X_state.T, sidelength, outpath,
                        fmod=fmod, state_int=state_int, title=title)
    return


if __name__ == '__main__':

    label = 'slide6'  # 'slide4', 'slide5', 'slide6', 'specific'

    version = '2'
    state_int = False
    fmod = '_int%d' % state_int
    # fmod = '_beige'

    sidelength = 20  #2, 10, or 20
    num_cells = sidelength ** 2
    curated = True
    random_mem = False  # TODO incorporate seed in random XI in simsetup/curated
    random_W = False  # TODO incorporate seed in random W in simsetup/curated
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_maze.txt'
    simsetup_main = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    if W_override_path is not None:
        print('Note: in main, overriding W from file...')
        explicit_W = np.loadtxt(W_override_path, delimiter=',')
        simsetup_main['FIELD_SEND'] = explicit_W
    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup_main['N'])
    print("\tsimsetup['P'],", simsetup_main['P'])

    # choices:
    replot_dir = RUNS_FOLDER + os.sep + 'explore' + os.sep + 'replot'
    if label == 'slide4':
        replot_dir = replot_dir + os.sep + 'slide4asW1'

        for k in range(0, 30):
            fname = 'X_%d.npz' % k
            fpath = replot_dir + os.sep + fname

            # 2) state to load
            #fnames = [a for a in os.listdir(source_dir) if a[-4:] == '.npz']
            qmod = fmod + '%d' % k

            #replot_overlap()
            X = state_load(fpath, cells_as_cols=True, num_genes=None, num_cells=None, txt=False)
            X = translate_lattice_state(X, sidelength, down=0, right=0)  # down 8

            outpath = replot_dir + os.sep + fname[:-4]
            outpath_ref = replot_dir + os.sep + 'ref0_' + fname[:-4]
            outpath_uniquecolours = replot_dir + os.sep + 'uniques_' + fname[:-4]
            outpath_scatter = replot_dir + os.sep + 'scatter_' + fname[:-4]

            replot_modern(
                X, simsetup_main, sidelength, outpath, version=version, fmod=qmod,
                state_int=state_int)
            replot_graph_lattice_reference_overlap_plotter(
                X, sidelength, outpath_ref, fmod=qmod, ref_node=0)
            replot_modern(
                X, simsetup_main, sidelength, outpath_uniquecolours, version='3', fmod=qmod,
                state_int=state_int)
            replot_scatter_dots(X, sidelength, outpath_scatter, fmod=qmod, state_int=state_int)

    elif label == 'slide5':
        replot_dir = replot_dir + os.sep + 'slide5_gamma1'

        #k_choice = np.arange(17)
        k_choice = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        flag_ranked_cmap = False

        # build ranked colourmap first
        if flag_ranked_cmap:
            state_data = []
            for k in k_choice:
                source_dir = replot_dir + os.sep + 'W%d' % k
                source_dir += os.sep + 'states'
                fpath = source_dir + os.sep + 'X_30.npz'
                X = state_load(fpath, cells_as_cols=True, num_genes=None, num_cells=None, txt=False)
                state_data.append(X)
            ranked_cmap = construct_cmap_from_ranked_states(state_data, simsetup_main['N'], show=True)
        else:
            ranked_cmap = None

        # plot data using custom colourmap
        for k in k_choice:
            source_dir = replot_dir + os.sep + 'W%d' % k
            source_dir += os.sep + 'states'

            # 2) state to load
            #fnames = [a for a in os.listdir(source_dir) if a[-4:] == '.npz']

            fnames = ['X_30.npz']
            fpaths = [source_dir + os.sep + a for a in fnames]
            print(fpaths)

            qmod = fmod + '_W%d' % k

            # 3) plot each file
            for idx, fpath in enumerate(fpaths):

                fname = fnames[idx]
                #replot_overlap()
                X = state_load(fpath, cells_as_cols=True, num_genes=None, num_cells=None, txt=False)
                if k_choice[idx] == '14':
                    X = translate_lattice_state(X, sidelength, down=8, right=0)  # down 8 for W14

                outpath = replot_dir + os.sep + fname[:-4]
                outpath_ref = replot_dir + os.sep + 'ref0_' + fname[:-4]
                outpath_uniquecolours = replot_dir + os.sep + 'uniques_' + fname[:-4]
                outpath_scatter = replot_dir + os.sep + 'scatter_' + fname[:-4]

                """
                replot_modern(
                    X, simsetup_main, sidelength, outpath, version=version, fmod=qmod,
                    state_int=state_int)
                replot_graph_lattice_reference_overlap_plotter(
                    X, sidelength, outpath_ref, fmod=qmod, ref_node=0)"""
                """replot_modern(
                    X, simsetup_main, sidelength, outpath_uniquecolours, version='3', fmod=qmod,
                    state_int=state_int)"""
                replot_scatter_dots(X, sidelength, outpath_scatter, fmod=qmod, state_int=state_int,
                                    cmap=ranked_cmap)

    elif label == 'slide6':
        replot_dir = replot_dir + os.sep + 'slide6'
        source_dir = replot_dir

        # 2) state to load
        fnames = [a for a in os.listdir(source_dir) if a[-4:] == '.npz']
        fpaths = [source_dir + os.sep + a for a in fnames]

        # 3) plot each file
        for idx, fpath in enumerate(fpaths):
            fname = fnames[idx]
            # replot_overlap()
            X = state_load(fpath, cells_as_cols=True, num_genes=None, num_cells=None, txt=False)
            outpath = replot_dir + os.sep + fname[:-4]
            replot_modern(X, simsetup_main, sidelength, outpath, version=version, fmod=fmod,
                          state_int=state_int)
            outpath_scatter = replot_dir + os.sep + 'scatter_' + fname[:-4]
            #replot_graph_lattice_reference_overlap_plotter(X, sidelength, outpath_ref,
            #                                               fmod=fmod, ref_node=0)
            replot_scatter_dots(X, sidelength, outpath_scatter, fmod=fmod, state_int=state_int)
    else:

        replot_dir = replot_dir + os.sep + 'plot_specific_points'

        from multicell.unsupervised_helper import plot_given_multicell

        agg_indices = [
            1395, 2904,        # bottom left: bottom left corner
            919, 771,          # bottom left: bottom right corner
            2325, 193,         # bottom left: top left corner
            3685, 556,         # bottom left: top right corner
            3717, 2652,        # bottom left: interior, left edge
            2357,                                    # top left: bottom left corner
            2847, 3563, 1334, 683, 2941, 728, 3995,  # top left: left interior edge
            1919,                                    # top left: top left corner
            74, 636,           # upper homogeneous tip
            1904, 454, 2656, 1569, 3277, 3473, 3034, 961, 186,  # upper cluster 2-phase
            345, 1089, 1252, 3964, 1765, 1497, 1252,      # right donut (upper): upper-right edge A
            2101, 347, 1135, 148,  # right donut (upper): upper-right edge B
            65, 2989, 1585,        # right donut (upper): left edge
            1467, 1262, 2345,  # right donut (lower): right edge A
            2602, 3952,        # right donut (lower): right edge B
        ]
        #outdir = RUNS_FOLDER + os.sep + 'explore' + os.sep + 'plot_specific_points'

        # where is the data?
        step = None
        # dirname = 'Wrandom0_gamma0.20_10k_periodic_fixedorderV3_p3_M100'
        #dirname = 'Wrandom0_gamma1.00_10k_periodic_fixedorderV3_p3_M100'
        #dirname = 'Wrandom0_gamma1.00_10k_fixedorder_p3_M4'
        dirname = 'W0_gamma1.00_10k_periodic_R1_p3_M100_machineEps'

        # step = 14
        # dirname = 'beta2.05_Wrandom0_gamma0.20_10k_periodic_fixedorderV3_p3_M100'

        manyruns_path = RUNS_FOLDER + os.sep + 'multicell_manyruns' + os.sep + dirname
        fpath_pickle = manyruns_path + os.sep + 'multicell_template.pkl'
        with open(fpath_pickle, 'rb') as pickle_file:
            multicell = pickle.load(pickle_file)  # unpickling multicell object

        for agg_index in agg_indices:
            # smod = ''
            smod = '_last'
            if step is not None:
                smod = '_%d' % step

            agg_dir = manyruns_path + os.sep + 'aggregate'
            fpath_state = agg_dir + os.sep + 'X_aggregate%s.npz' % smod
            fpath_energy = agg_dir + os.sep + 'X_energy%s.npz' % smod
            fpath_pickle = manyruns_path + os.sep + 'multicell_template.pkl'
            print(fpath_state)
            X = np.load(fpath_state)['arr_0'].T  # umap wants transpose
            X_state = X[agg_index, :]
            print(X_state.shape)

            # plot option 1)
            #step_hack = 0  # TODO care this will break if class has time-varying applied field
            #multicell.graph_state_arr[:, step_hack] = X_state[:]
            # assert np.array_equal(multicell_template.field_applied, np.zeros((total_spins, multicell_template.total_steps)))
            #plot_given_multicell(multicell, step_hack, agg_index, replot_dir)

            # plot option 2) using replot
            X_state = X_state.reshape(num_cells, simsetup_main['N'])
            """
            outpath_ref = replot_dir + os.sep + 'agg%d_ref0' % agg_index
            replot_graph_lattice_reference_overlap_plotter(
                X_state.T, sidelength, outpath_ref, fmod=fmod, ref_node=0)

            outpath = replot_dir + os.sep + 'agg%d_modern' % agg_index
            replot_modern(X_state.T, simsetup_main, sidelength, outpath,
                          version=version, fmod=fmod, state_int=state_int)"""

            outpath = replot_dir + os.sep + 'agg%d_scatter' % agg_index
            title = None #'test_title'
            replot_scatter_dots(X_state.T, sidelength, outpath,
                                fmod=fmod, state_int=state_int, title=title)
