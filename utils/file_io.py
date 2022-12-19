import datetime
import numpy as np
import os
import sys

# LIBRARY GLOBAL MODS
CELLTYPES = os.path.dirname(os.path.dirname(__file__))
INPUT_FOLDER = CELLTYPES + os.sep + "input"
RUNS_FOLDER = CELLTYPES + os.sep + "runs"                      # store timestamped runs here
sys.path.append(CELLTYPES)
print("Appended to sys path", CELLTYPES)  # TODO can maybe move this too simetup fn call and call once somewhere else...


def run_subdir_setup(run_subfolder=None, timedir_override=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%I.%M.%S%p")
    experiment_dir = RUNS_FOLDER
    if timedir_override is not None:
        time_folder = timedir_override
    else:
        time_folder = current_time
    if run_subfolder is None:
        current_run_dir = experiment_dir + os.sep + time_folder
    else:
        if os.path.isabs(run_subfolder):
            current_run_dir = run_subfolder + os.sep + time_folder
        else:
            current_run_dir = experiment_dir + os.sep + run_subfolder + os.sep + time_folder

    # make subfolders in the timestamped run directory:
    data_dir = os.path.join(current_run_dir, "data")
    plot_data_dir = os.path.join(current_run_dir, "plot_data")
    lattice_dir = os.path.join(current_run_dir, "lattice")
    plot_lattice_dir = os.path.join(current_run_dir, "plot_lattice")
    simsetup_dir = os.path.join(current_run_dir, "simsetup")
    states_dir = os.path.join(current_run_dir, "states")
    dir_list = [RUNS_FOLDER, current_run_dir, plot_data_dir, data_dir, lattice_dir,
                plot_lattice_dir, simsetup_dir, states_dir]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    # io path storage to pass around
    io_dict = {'basedir': current_run_dir,
               'datadir': data_dir,
               'plotdatadir': plot_data_dir,
               'latticedir': lattice_dir,
               'plotlatticedir': plot_lattice_dir,
               'simsetupdir': simsetup_dir,
               'statesdir': states_dir,
               'runinfo': current_run_dir + os.sep + 'run_info.txt'}

    # make minimal run_info settings file with first line as the base output dir
    runinfo_append(io_dict, ('basedir', current_run_dir))

    return io_dict


def state_write(state, row_vals, col_vals, dataname, rowname, colname, output_dir):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    datapath = output_dir + os.sep + dataname + ".txt"
    rowpath = output_dir + os.sep + dataname + '_' + rowname + ".txt"
    colpath = output_dir + os.sep + dataname + '_' + colname + ".txt"
    np.savetxt(datapath, np.array(state), delimiter=",", fmt="%d")
    np.savetxt(rowpath, np.array(row_vals), delimiter=",")
    np.savetxt(colpath, np.array(col_vals), delimiter=",", fmt="%s")
    return datapath, rowpath, colpath


def state_read(datapath, rowpath, colpath):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    state = np.loadtxt(datapath, delimiter=",")
    row = np.loadtxt(rowpath, delimiter=",", dtype=float)
    col = np.loadtxt(colpath, delimiter=",", dtype=str)
    return state, row, col


def runinfo_append(io_dict, info_list, multi=False):
    # multi: list of list flag
    if multi:
        with open(io_dict['runinfo'], 'a') as runinfo:
            for line in info_list:
                runinfo.write(','.join(str(s) for s in line) + '\n')
    else:
        with open(io_dict['runinfo'], 'a') as runinfo:
            runinfo.write(','.join(str(s) for s in info_list) + '\n')
    return


def write_general_arr(X, data_folder, fname, txt=True, compress=False):
    """
    Writes general data array (txt, npy, or compressed npz)
    """
    if txt:
        assert not compress
        fpath = data_folder + os.sep + fname + '.txt'
        np.savetxt(fpath, X, delimiter=',')
    else:
        if compress:
            fpath = data_folder + os.sep + fname + '.npy'
            np.save(fpath, X)
        else:
            fpath = data_folder + os.sep + fname + '.npz'
            np.savez(fpath, a=X)
    return fpath


def read_general_arr(fpath, txt=True, compress=False):
    """
    Reads general data array (txt, npy, or compressed npz)
    """
    if txt:
        assert not compress
        return np.loadtxt(fpath, delimiter=',')
    else:
        X = np.load(fpath)
        if compress:
            return X['a']
        else:
            return X
