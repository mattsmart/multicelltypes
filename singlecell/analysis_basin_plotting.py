import matplotlib.pyplot as plt
import numpy as np
import os

from utils.file_io import RUNS_FOLDER


def plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, memory_labels, savepath, highlights=None):
    """
    proj_timeseries_array is expected dim p x time
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert proj_timeseries_array.shape[0] == len(memory_labels)
    plt.clf()
    if highlights is None:
        plt.plot(range(num_steps), proj_timeseries_array.T, color='blue', linewidth=0.75)
    else:
        plt.plot(range(num_steps), proj_timeseries_array.T, color='grey', linewidth=0.55, linestyle='dashed')
        for key in list(highlights.keys()):
            plt.plot(range(num_steps), proj_timeseries_array[key,:], color=highlights[key], linewidth=0.75, label=memory_labels[key])
        plt.legend()
    plt.title('Ensemble mean (n=%d) projection timeseries' % ensemble)
    plt.ylabel('Mean projection onto each memory')
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


def plot_basin_occupancy_timeseries(basin_occupancy_timeseries, num_steps, ensemble, memory_labels, threshold, spurious_list, savepath, highlights=None):
    """
    basin_occupancy_timeseries: is expected dim (p + spurious tracked) x time  note spurious tracked default is 'mixed'
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert basin_occupancy_timeseries.shape[0] == len(memory_labels) + len(spurious_list)  # note spurious tracked default is 'mixed'
    assert spurious_list[0] == 'mixed'
    plt.clf()
    if highlights is None:
        plt.plot(range(num_steps), basin_occupancy_timeseries.T, color='blue', linewidth=0.75)
    else:
        plt.plot(range(num_steps), basin_occupancy_timeseries.T, color='grey', linewidth=0.55, linestyle='dashed')
        for key in list(highlights.keys()):
            plt.plot(range(num_steps), basin_occupancy_timeseries[key,:], color=highlights[key], linewidth=0.75, label=memory_labels[key])
        if len(memory_labels) not in list(highlights.keys()):
            plt.plot(range(num_steps), basin_occupancy_timeseries[len(memory_labels), :], color='orange',
                     linewidth=0.75, label='mixed')
        plt.legend()
    plt.title('Occupancy timeseries (ensemble %d)' % ensemble)
    plt.ylabel('Occupancy in each memory (threshold proj=%.2f)' % threshold)
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


def plot_basin_step(basin_step_data, step, ensemble, memory_labels, memory_id, spurious_list, savepath, highlights={},
                    title_add='', autoscale=True, inset=False, init_mem=None):
    """
    basin_step_data: of length p or p + k where k tracks spurious states, p is encoded memories length
    highlights: dict of idx: color for certain memory projections to highlight
    """
    assert len(basin_step_data) in [len(memory_labels), len(memory_labels) + len(spurious_list)]
    if highlights is not None:
        if len(basin_step_data) == len(memory_labels):
            xticks = memory_labels
            bar_colors = ['grey' if label not in [memory_labels[a] for a in list(highlights.keys())]
                          else highlights[memory_id[label]]
                          for label in xticks]
        else:
            xticks = memory_labels + spurious_list
            bar_colors = ['grey' if label not in [memory_labels[a] for a in list(highlights.keys())]
                          else highlights[memory_id[label]]
                          for label in xticks]
    else:
        xticks = memory_labels + spurious_list
        bar_colors = ['blue' for _ in xticks]
        bar_colors[xticks.index('mixed')] = 'gray'
        if init_mem is not None:
            bar_colors[xticks.index(init_mem)] = 'black'
    # plotting
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 12})
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(21.5, 10.5)
    h = plt.bar(range(len(xticks)), basin_step_data, color=bar_colors)
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.3)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks, ha='right', rotation=45, fontsize=10)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    if not autoscale:
        ax.set_ylim(0.0, ensemble)
    plt.title('Ensemble coordinate at step %d (%d cells) %s' % (step, ensemble, title_add))
    plt.ylabel('Class occupancy count')
    plt.xlabel('Class labels')
    if inset:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, 2.0, 2.0, loc='upper right', bbox_to_anchor=(0.85, 0.85),
                           bbox_transform=ax.figure.transFigure)  # no zoom
        h = axins.bar(range(len(xticks)), basin_step_data, color=bar_colors)
        axins.set_ylim(0.0, ensemble)
    fig.savefig(savepath, bbox_inches='tight')
    return plt.gca()


def plot_basin_grid(grid_data, ensemble, steps, memory_labels, plotdir, spurious_list, ax=None, normalize=True,
                    fs=9, relmax=True, rotate_standard=True, extragrid=False, plotname='plot_basin_grid', ext='.jpg',
                    vforce=None, cmap_int=11, namemod=''):
    """
    plot matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    Args:
    - relmax means max of color scale will be data max
    - k represents the number of extra tracked states, by default this is 1 (i.e. mixed state, not in any basin)
    - rotate_standard: determine xlabel orientation
    """
    assert grid_data.shape == (len(memory_labels), len(memory_labels) + len(spurious_list))
    # adjust data normalization
    #assert normalize
    datamax = np.max(grid_data)
    datamin = np.min(grid_data)
    if np.max(grid_data) > 1.0 and normalize:
        grid_data = grid_data / ensemble
        datamax = datamax / ensemble
        datamin = datamax / ensemble
    # adjust colourbar max val
    if vforce is not None:
        vmax = vforce
        plotname += '_vforce%.2f%s' % (vforce, namemod)
    else:
        if relmax:
            vmax = datamax
        else:
            if normalize:
                vmax = 1.0
            else:
                vmax = ensemble
        plotname += namemod

    # plot setup
    if not ax:
        plt.clf()
        ax = plt.gca()
        plt.gcf().set_size_inches(18.5, 12.5)
    # plot the heatmap
    """
    default color: 'YlGnBu', alts are: 'bone_r', 'BuPu', 'PuBuGn', 'Greens', 'Spectral', 'Spectral_r' (with k grid),
                                       'cubehelix_r', 'magma_r'
    note: fix vmax at 0.5 of max works nice
    note: aspect None, 'auto', scalar, or 'equal'
    """
    if cmap_int is not None:
        cmap = plt.get_cmap('YlGnBu', cmap_int)
    else:
        cmap = plt.get_cmap('YlGnBu')
    imshow_kw = {'cmap': cmap, 'aspect': None, 'vmin': 0.0, 'vmax': vmax}  # note: fix at 0.5 of max works nice
    im = ax.imshow(grid_data, **imshow_kw)
    # create colorbar
    cbar_kw = {'aspect': 30, 'pad': 0.02}   # larger aspect, thinner bar
    cbarlabel = 'Basin occupancy fraction'
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fs+2, labelpad=20)
    # hack title placement
    plt.text(0.5, 1.3, 'Basin grid transition data (%d cells per basin, %d steps) (%s)' % (ensemble, steps, plotname),
             horizontalalignment='center', transform=ax.transAxes, fontsize=fs+4)
    # axis labels
    plt.xlabel('Ensemble fraction after %d steps' % steps, fontsize=fs+2)
    ax.xaxis.set_label_position('top')
    plt.ylabel('Ensemble initial condition (%d cells per basin)' % ensemble, fontsize=fs+2)
    # show all ticks
    ax.set_xticks(np.arange(grid_data.shape[1]))
    ax.set_yticks(np.arange(grid_data.shape[0]))
    # label them with the respective list entries.
    assert len(spurious_list) == 1  # TODO col labels as string types + k mixed etc
    ax.set_xticklabels(memory_labels + spurious_list, fontsize=fs)
    ax.set_yticklabels(memory_labels, fontsize=fs)
    # Rotate the tick labels and set their alignment.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    if rotate_standard:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    else:
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")
    # add gridlines
    ax.set_xticks(np.arange(-.5, grid_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)  # grey good to split, white looks nice though
    # hack to add extra gridlines (not clear how to have more than minor and major on one axis)
    if extragrid:
        for xcoord in np.arange(-.5, grid_data.shape[1], 8):
            ax.axvline(x=xcoord, ls='--', color='grey', linewidth=1)
        for ycoord in np.arange(-.5, grid_data.shape[0], 8):
            ax.axhline(y=ycoord, ls='--', color='grey', linewidth=1)
    plt.savefig(plotdir + os.sep + plotname + ext, dpi=100, bbox_inches='tight')
    return plt.gca()


def plot_overlap_grid(grid_data, memory_labels, plotdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      rotate_standard=True, extragrid=False, plotname=None, ext='.pdf',
                      hamming=False, vforce=None, cmap_int=11, namemod='', memory_labels_x=None):
    """
    Alteration of plot_basin_grid to support (simpler) overlap and hamming dist plots
    """
    if memory_labels_x is None:
        memory_labels_x = memory_labels
    print('plot_overlap_grid:', grid_data.shape, len(memory_labels_x), len(memory_labels), plotname, namemod)
    plotnames = ['celltypes_overlap', 'celltypes_hamming']
    if plotname is None:
        plotname = plotnames[hamming]
    datalabels = ['Overlap', 'Hamming distance']
    assert grid_data.shape == (len(memory_labels), len(memory_labels_x))
    datamax = np.max(grid_data)
    datamin = np.min(grid_data)
    # adjust colourbar max val
    if vforce is not None:
        vmax = vforce
        plotname += '_vforce%.2f%s' % (vforce, namemod)
    else:
        if relmax:
            vmax = datamax
        else:
            if normalize:
                vmax = 1.0
            else:
                vmax = N
        plotname += namemod
    # plot setup
    if not ax:
        plt.clf()
        ax = plt.gca()
        plt.gcf().set_size_inches(18.5, 12.5)
    # plot the heatmap
    if cmap_int is not None:
        cmap = plt.get_cmap('YlGnBu', cmap_int)
    else:
        cmap = plt.get_cmap('YlGnBu')
    imshow_kw = {'cmap': cmap, 'aspect': None, 'vmin': datamin, 'vmax': vmax}  # note: fix at 0.5 of max works nice
    #imshow_kw = {'cmap': 'YlGnBu', 'aspect': None, 'vmin': datamin, 'vmax': vmax}  # note: fix at 0.5 of max works nice
    im = ax.imshow(grid_data, **imshow_kw)
    # create colorbar
    cbar_kw = {'aspect': 30, 'pad': 0.02}   # larger aspect, thinner bar
    cbarlabel = '%s between memory i and j' % datalabels[hamming]
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fs+2, labelpad=20)
    # hack title placement
    plt.text(0.5, 1.3, '%s between memories (%s)' % (datalabels[hamming], plotname), horizontalalignment='center',
             transform=ax.transAxes, fontsize=fs+4)
    ax.set_xticks(np.arange(grid_data.shape[1]))
    ax.set_yticks(np.arange(grid_data.shape[0]))
    # label them with the respective list entries.
    ax.set_xticklabels(memory_labels_x, fontsize=fs)
    ax.set_yticklabels(memory_labels, fontsize=fs)
    # Rotate the tick labels and set their alignment.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    if rotate_standard:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    else:
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
    # add gridlines
    ax.set_xticks(np.arange(-.5, grid_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)  # grey good to split, white looks nice though
    # hack to add extra gridlines (not clear how to have more than minor and major on one axis)
    if extragrid:
        for xcoord in np.arange(-.5, grid_data.shape[1], 8):
            ax.axvline(x=xcoord, ls='--', color='grey', linewidth=1)
        for ycoord in np.arange(-.5, grid_data.shape[0], 8):
            ax.axhline(y=ycoord, ls='--', color='grey', linewidth=1)
    plt.savefig(plotdir + os.sep + plotname + ext, dpi=100, bbox_inches='tight')
    return plt.gca()


def grid_video(rundir, vidname, imagedir=None, ext='.mp4', fps=20):
    """
    Make a video of the grid over time using ffmpeg.
    Note: ffmpeg must be installed and on system path.
    Args:
        rundir: Assumes sequentially named images of grid over time are in "plot_lattice" subdir of rundir.
        vidname: filename for the video (no extension); it will be placed in a "video" subdir of rundir
        imagedir: override use of "plot_lattice" subdir of rundir as the source [Default: None]
        ext: only '.mp4' has been tested, seems to work on Windows Media Player but not VLC
        fps: video frames per second; 1, 5, 20 work well
    Returns:
        path to video
    """
    from utils.make_video import make_video_ffmpeg
    # args specify
    if imagedir is None:
        imagedir = rundir + os.sep + "plot_lattice"
    if not os.path.exists(rundir + os.sep + "video"):
        os.makedirs(rundir + os.sep + "video")
    videopath = rundir + os.sep + "video" + os.sep + vidname + ext
    # call make video fn
    print("Creating video at %s..." % videopath)
    make_video_ffmpeg(imagedir, videopath, fps=fps, ffmpeg_dir=None)
    print("Done")
    return videopath


if __name__ == '__main__':
    print("main not implemented")
