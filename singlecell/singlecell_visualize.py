#import matplotlib as mpl            # Fix to allow intermediate compatibility of radar label rotation / PyCharm SciView
#mpl.use("TkAgg")                    # Fix to allow intermediate compatibility of radar label rotation / PyCharm SciView
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

from singlecell.singlecell_functions import label_to_state, state_to_label, hamiltonian, check_min_or_max, hamming, get_all_fp, calc_state_dist_to_local_min


def plot_as_bar(projection_vec, memory_labels, alpha=1.0):
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5)
    h = plt.bar(range(len(memory_labels)), projection_vec, label=memory_labels, alpha=alpha)
    plt.subplots_adjust(bottom=0.3)
    xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    plt.xticks(xticks_pos, memory_labels, ha='right', rotation=45, size=7)
    return fig, plt.gca()


def plot_as_radar(projection_vec, memory_labels, color='b', rotate_labels=True, fig=None, ax=None):
    """
    # radar plots not built-in to matplotlib
    # reference code uses pandas: https://python-graph-gallery.com/390-basic-radar-chart/
    """

    p = len(memory_labels)

    # Angle of each axis in the plot
    angles = [n / float(p) * 2 * pi for n in range(p)]

    # Add extra element to angles and data array to close off filled area
    angles += angles[:1]
    projection_vec_ext = np.zeros(len(angles))
    projection_vec_ext[0:len(projection_vec)] = projection_vec[:]
    projection_vec_ext[-1] = projection_vec[0]

    # Initialise the spider plot
    if fig is None:
        assert ax is None
        fig = plt.figure()
        ax = plt.subplot(111, projection='polar') #'polar=True)
        fig.set_size_inches(9, 5)
    else:
        fig = plt.gcf()
        fig.set_size_inches(9, 5)

    # Draw one ax per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(memory_labels)

    # Draw ylabels
    ax.set_rlabel_position(45)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
    ax.set_ylim(-1, 1)
    ax.tick_params(axis='both', color='grey', size=12)

    # Plot data
    ax.plot(angles, projection_vec_ext, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, projection_vec_ext, color, alpha=0.1)

    # Rotate the type labels
    if rotate_labels:
        fig.canvas.draw()  # trigger label positions to extract x, y coords
        angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles)
        labels=[]
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            lab = ax.text(x, y - 0.05, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(), size=8)
            lab.set_rotation(angle)
            labels.append(lab)
        ax.set_xticklabels([])

    return fig, ax


def plot_state_prob_map(intxn_matrix, beta=None, field=None, fs=0.0, ax=None, decorate_FP=True):
    if ax is None:
        ax = plt.figure(figsize=(8,6)).gca()

    fstring = 'None'
    if field is not None:
        fstring = '%.2f' % fs
    N = intxn_matrix.shape[0]
    num_states = 2 ** N
    energies = np.zeros(num_states)
    colours = ['blue' for i in range(num_states)]
    fpcolor = {True: 'green', False: 'red'}
    for label in range(num_states):
        state = label_to_state(label, N, use_neg=True)
        energies[label] = hamiltonian(state, intxn_matrix, field=field, fs=fs)
        if decorate_FP:
            is_fp, is_min = check_min_or_max(intxn_matrix, state, energy=energies[label], field=field, fs=fs)
            if is_fp:
                colours[label] = fpcolor[is_min]
    if beta is None:
        ax.scatter(list(range(2 ** N)), energies, c=colours)
        ax.set_title(r'$H(s), \beta=\infty$, field=%s' % (fstring))
        #ax.set_ylim((-10,10))
    else:
        ax.scatter(list(range(2 ** N)), np.exp(-beta * energies), c=colours)
        ax.set_yscale('log')
        ax.set_title(r'$e^{-\beta H(s)}, \beta=%.2f$, field=%s' % (beta, fstring))
    plt.show()
    return


def hypercube_visualize(simsetup, X_reduced, energies, num_cells=1, elevate3D=True, edges=True, all_edges=False,
                        minima=[], maxima=[], colours_dict=None, basin_labels=None, surf=True, beta=None, ax=None):
    """
    Plot types
        A - elevate3D=True, surf=True, colours_override=None     - 3d surf, z = energy
        B - elevate3D=True, surf=False, colours_override=None    - 3d scatter, z = energy, c = energy
        C - elevate3D=True, surf=False, colours_override=list(N) - 3d scatter, z = energy, c = predefined (e.g. basins colour-coded)
        D - elevate3D=False, colours_override=None               - 2d scatter, c = energy
        E - elevate3D=False, colours_override=list(N)            - 2d scatter, c = predefined (e.g. basins colour-coded)
        F - X_reduced is dim 2**N x 3, colours_override=None     - 3d scatter, c = energy
        G - X_reduced is dim 2**N x 3, colours_override=list(N)  - 3d scatter, c = predefined (e.g. basins colour-coded)
    All plots can have partial or full edges (neighbours) plotted
    """
    # TODO for trisurf possible to manually define GOOD triangulation?
    # TODO neighbour preserving?
    # TODO think there are duplicate points in hd rep... check this bc pics look too simple
    # TODO MDS - dist = dist to cell fate subspace as in mehta SI? try
    # TODO note cbar max for surf plot is half max of cbar for other plots why

    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

    # setup data
    N = simsetup['N'] * num_cells
    states = np.array([label_to_state(label, N) for label in range(2 ** N)])

    # setup cmap
    if beta is None:
        energies_norm = (energies + np.abs(np.min(energies))) / (np.abs(np.max(energies)) + np.abs(np.min(energies)))
        cbar_label = r'$H(s)$'
    else:
        energies = np.exp(-beta * energies)
        energies_norm = (energies + np.abs(np.min(energies))) / (np.abs(np.max(energies)) + np.abs(np.min(energies)))
        cbar_label = r'$exp(-\beta H(s))$'

    if colours_dict is None:
        colours = energies_norm
    else:
        assert surf is False
        colours = colours_dict['clist']

    if X_reduced.shape[1] == 3:
        # explicit 3D plot
        sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=colours, s=20)
    else:
        assert X_reduced.shape[1] == 2
        if elevate3D:
            # implicit 3D plot, height is energy
            if surf:

                surface_method = 2
                assert surface_method in [1,2]

                #cmap_str =  # plt.cm.viridis
                cmap_str = 'Spectral_r' # 'viridis'
                cmap = plt.cm.get_cmap(cmap_str)

                # original approach:
                if surface_method == 1:

                    sc = ax.plot_trisurf(X_reduced[:,0], X_reduced[:,1], energies_norm, cmap=cmap)
                    #sc = ax.plot_wireframe(X_reduced[:,0], X_reduced[:,1], energies_norm)
                # new approach: interpolate
                else:
                    assert surface_method == 2
                    from scipy.interpolate import griddata

                    x0 = X_reduced[:,0]
                    y0 = X_reduced[:,1]
                    z0 = energies_norm
                    nmeshpoints = 50  #len(x0)
                    x_mesh = np.linspace(x0.min(), x0.max(), nmeshpoints)
                    y_mesh = np.linspace(y0.min(), y0.max(), nmeshpoints)
                    z_interpolated = griddata(
                        (x0, y0),
                        z0,
                        (x_mesh[None,:], y_mesh[:,None]),
                        method='cubic',      # nearest, linear, or cubic
                        fill_value=np.nan)    # defaults to np.nan; try 0
                    print('energies:', energies.shape, energies.min(), energies.max())
                    print('z0:', z0.shape, z0.min(), z0.max())
                    print('z_interpolated:', z_interpolated.shape, z_interpolated.min(), z_interpolated.max())
                    print('np.isnan(z_interpolated).sum()', np.isnan(z_interpolated).sum())
                    # converts vectors to matrices (expanded representation of coordinates)
                    x_mesh_matrix, y_mesh_matrix = np.meshgrid(x_mesh, y_mesh)
                    sc = ax.plot_surface(
                        x_mesh_matrix, y_mesh_matrix, z_interpolated,
                        edgecolors='k',
                        linewidths=0.5,
                        cmap=cmap,
                        vmin=np.nanmin(z_interpolated),
                        vmax=np.nanmax(z_interpolated))  # scaled by 1.1 too reduce brightness of peak

                    # add contour lines on bottom of plot (contourf = filled)
                    #cset = ax.contourf(
                    #    x_mesh_matrix, y_mesh_matrix, z_interpolated,
                    #    zdir='z', offset=np.nanmin(z_interpolated), cmap=cmap)
                    contour_offset = -0.4 # np.nanmin(z_interpolated)
                    cset = ax.contour(
                        x_mesh_matrix, y_mesh_matrix, z_interpolated,
                        zdir='z', offset=contour_offset, cmap=cmap)

            else:
                if colours_dict is not None:
                    for key in list(colours_dict['basins_dict'].keys()):
                        indices = colours_dict['basins_dict'][key]
                        sc = ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], energies_norm[indices], s=20,
                                        c=colours_dict['fp_label_to_colour'][key],
                                        label='Basin ID#%d (size %d)' % (key, len(indices)))
                else:
                    sc = ax.scatter(X_reduced[:,0], X_reduced[:,1], energies_norm, c=colours, s=20)
        else:
            # 2D plot
            if colours_dict is None:
                sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colours)
            else:
                for key in list(colours_dict['basins_dict'].keys()):
                    indices = colours_dict['basins_dict'][key]
                    sc = ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], s=20,
                                    c=colours_dict['fp_label_to_colour'][key],
                                    label='Basin ID#%d (size %d)' % (key, len(indices)))
    # legend for colours
    if colours_dict is None:
        cbar = plt.colorbar(sc)
        cbar.set_label(cbar_label)
    else:
        ax.legend()

    # annotate minima
    if basin_labels is None:
        basin_labels = {a: 'ID: %d' % a for a in minima}
    for minimum in minima:
        txt = basin_labels[minimum]
        state_new = X_reduced[minimum, :]
        if elevate3D or X_reduced.shape[1] == 3:
            if elevate3D:
                z = energies_norm[minimum] - 0.05
            if X_reduced.shape[1] == 3:
                z = state_new[2]
            ax.text(state_new[0], state_new[1], z, txt, fontsize=10)
        else:
            ax.annotate(txt, xy=(state_new[0], state_new[1]), fontsize=12)

    if edges:
        print('Adding edges to plot...')  # TODO these appear incorrect for twocell visualization
        for label in range(2 ** N):
            state_orig = states[label, :]
            state_new = X_reduced[label, :]
            nbrs = [0] * N
            if all_edges or label in maxima or label in minima or abs(energies_norm[label] - 1.0) < 1e-4:
                for idx in range(N):
                    nbr_state = np.copy(state_orig)
                    nbr_state[idx] = -1 * nbr_state[idx]
                    nbrs[idx] = state_to_label(nbr_state)
                for nbr_int in nbrs:
                    nbr_new = X_reduced[nbr_int, :]
                    x = [state_new[0], nbr_new[0]]
                    y = [state_new[1], nbr_new[1]]
                    if X_reduced.shape[1] == 3:
                        z = [state_new[2], nbr_new[2]]
                    else:
                        z = [energies_norm[label], energies_norm[nbr_int]]
                    if elevate3D or X_reduced.shape[1] == 3:
                        ax.plot(x, y, z, alpha=0.8, color='grey', lw=0.5)
                    else:
                        ax.plot(x, y, alpha=0.8, color='grey', lw=0.5)
    ax.grid('off')
    ax.axis('off')
    plt.show()
    return


def save_manual(fig, dir, fname, close=True):
    filepath = dir + os.sep + fname + ".png"
    fig.savefig(filepath, dpi=100)
    if close:
        plt.close()
    return
