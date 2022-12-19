import errno
import os
import re
import shutil
import subprocess
import sys


def natural_sort(unsorted_list):
    """Used to sort lists like a human
    e.g. [1, 10, 11, 2, 3] to [1, 2, 3, 10, 11]
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alphanum_key)


def copy_and_rename_plots(source_dir, output_dir, fhead='lattice_at_time_', fmod='%05d', ftype='.jpg', nmax=20):
    """
    Given a plot_lattice or plot_grid directory, copies it and renames the files in an order that
    ffmpeg.exe likes, e.g. (0001, 0002, 0003, ... , 0010, 0011, ...)
    Notes:
        - assumes less than 10000 files are being copied (for ffmpeg simplicity)
    """
    # make temp dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # copy the files matching the template
    for obj in os.listdir(source_dir):
        if obj[:len(fhead)] == fhead:
            shutil.copy(source_dir + os.sep + obj, output_dir)
    # naturally sort the copied files
    print(output_dir)
    unsorted_files = os.listdir(output_dir)
    assert len(unsorted_files) <= 99999  # assume <= 5 digits later
    sorted_files = natural_sort(unsorted_files)
    # rename the files accordingly
    for i, filename in enumerate(sorted_files):
        num = fmod % i
        newname = fhead + num + ftype
        os.rename(os.path.join(output_dir, filename), os.path.join(output_dir, newname))
        if i >= nmax:
            break
    return


def make_video_ffmpeg(source_dir, outpath, fps=5, fhead='lattice_at_time_', fmod='%05d', ftype='.jpg', nmax=20, ffmpeg_dir=None):
    """Makes a video using ffmpeg - also copies the lattice plot dir, changes filenames, and deletes the copy
    Args:
        source_dir: source directory
        outpath: path and filename of the output video
        fps: frames per second of the output video
        ffmpeg_dir: [default: None] location of the ffmpeg directory (root where bin containing ffmpeg.exe is)
    Returns:
        None
    Notes:
        - assumes ffmpeg has been extracted on your system and added to the path
        - if it's not added to path, point to it (the directory containing ffmpeg bin) using ffmpeg_dir arg
        - assumes less than 10000 images are being joined (for ffmpeg simplicity)
        - .mp4 seems to play best with Windows Media Player, not VLC
    """
    # make sure video directory exists
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    # make temp dir
    pardir = os.path.abspath(os.path.join(source_dir, os.pardir))
    temp_plot_dir = os.path.join(pardir, "temp")
    copy_and_rename_plots(source_dir, temp_plot_dir, fhead=fhead, fmod=fmod, ftype=ftype, nmax=nmax)
    template = fhead + fmod + ftype
    # make video
    command_line = ["ffmpeg",
                    "-framerate", "%d" % fps,                              # *force* video frames per second
                    "-i", os.path.join(temp_plot_dir, template),           # set the input files
                    "-vcodec", "libx264",                                  # set the video codec
                    "-r", "30",                                            # copies image r times (use 30 for VLC)
                    "-pix_fmt", "yuv420p",                                 # pixel formatting
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",            # fix if height/width not even ints
                    "%s" % outpath]                                        # output path of video
    if ffmpeg_dir is not None:
        app_path = os.path.join(ffmpeg_dir, "bin", "ffmpeg.exe")
        sp = subprocess.Popen(command_line, executable=app_path, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    else:
        sp = subprocess.Popen(command_line, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        while True:
            out = sp.stderr.read(1)
            if out == b'' and sp.poll() != None:
                break
            if out != b'':
                print(out)
                #sys.stdout.write(out)
                sys.stdout.flush()
    out, err = sp.communicate()
    print(out, err, sp.returncode)
    # delete temp dir
    shutil.rmtree(temp_plot_dir)
    return


if __name__ == '__main__':
    pardir = os.path.abspath(os.path.join('.', os.pardir))
    ld = os.listdir(pardir)
    print(list(filter(os.path.isfile, ld)))
