import numpy as np
import os
import pandas as pd

from dataprocess.data_settings import DATADIR


def save_npz_of_xi_genes_cells_signal(npzpath, arr, genes, cells, signals):
    np.savez_compressed(npzpath, arr=arr, genes=genes, cells=cells, signals=signals)
    return


def load_npz_of_arr_genes_cells_signals(npzpath, verbose=True):
    """
    Can also use for memory array, gene labels, and cell cluster names!
    """
    if verbose:
        print("loading npz of arr genes cells at", npzpath, "...")
    loaded = np.load(npzpath, allow_pickle=True)
    arr = loaded['arr']
    genes = loaded['genes']
    cells = loaded['cells']
    signals = loaded['signals']
    if verbose:
        print("loaded arr, genes, cells, signals:", arr.shape, genes.shape, cells.shape, signals.shape)
    return arr, genes, cells, signals


def read_xi_with_row_col_labels(xi_path, verbose=True):
    df = pd.read_csv(xi_path, index_col=0)
    cell_labels = np.array(df.columns)
    gene_labels = np.array(df.index)
    xi = df.values
    if verbose:
        print("Loaded xi_path: %s" % xi_path)
        print("xi.shape:", xi.shape, '\n', cell_labels, '\n', gene_labels)
    return xi, gene_labels, cell_labels


def read_no_headers_signal_csv(signal_csv):
    df = pd.read_csv(signal_csv, header=None)
    return df.values


def conv_xi_and_signal_csv_to_npz(xi_path, signal_path, outpath='unfold_exp_mems_genes_types_signals.npz'):
    xi, gene_names, cell_names = read_xi_with_row_col_labels(xi_path, verbose=True)
    signals = read_no_headers_signal_csv(signal_path)
    assert signals.shape[0] == signals.shape[1] and signals.shape[0] == xi.shape[0]
    save_npz_of_xi_genes_cells_signal(outpath, xi, gene_names, cell_names, signals)
    return


if __name__ == '__main__':
    # convert all "experiment" subdirs in input/unfolding/ to npzs
    UNFOLDINGDIR = DATADIR + os.sep + 'unfolding'
    for expt in os.listdir(UNFOLDINGDIR):
        expt_path = UNFOLDINGDIR + os.sep + expt
        expt_files = os.listdir(expt_path)
        print(expt_files)
        if len(expt_files) != 2:
            print("Skipping, require exactly 2 files (Xi and W) for processing")
        else:
            xi = 'unfolding_%s_XI.csv' % expt
            signal = 'unfolding_%s_W.csv' % expt
            assert xi in expt_files and signal in expt_files
            npzpath = UNFOLDINGDIR + os.sep + 'unfold_%s_mems_genes_types_signals.npz' % expt
            conv_xi_and_signal_csv_to_npz(expt_path + os.sep + xi, expt_path + os.sep + signal, outpath=npzpath)
