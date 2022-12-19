import errno
import os
from scipy import misc


def get_cropped_array(png_array):
    """Crop the array neatly to the center 100x100 points
    Notes:
        - this has only been tested on 1000x1000 size lattice images
    """
    xlen, ylen, _ = png_array.shape
    cropped_png_array = png_array[0.4165*xlen:0.5143*xlen, 0.4627*ylen:0.5623*ylen, :]
    return cropped_png_array


def copy_and_crop_plots(plot_lattice_dir, output_dir):
    """Given a plot_lattice or plot_grid directory, copies it and crops all the images
    Notes:
        - this has only been tested on 1000x1000 size lattice images
    """
    # make new folder
    try:  # check existence again to handle concurrency problems
        os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise
    # crop the files
    for i, filename in enumerate(os.listdir(plot_lattice_dir)):
        png_array = misc.imread(os.path.join(plot_lattice_dir, filename))
        cropped_png_array = get_cropped_array(png_array)
        misc.imsave(os.path.join(output_dir, filename[:-4] + '_cropped.png'), cropped_png_array)
    return
