import numpy as np
import os
import re

from dataprocess.data_settings import DATADIR, RAWDATA_2014MEHTA, RAWDATA_2018SCMCA, PIPELINES_VALID

"""
Standardize: convert different formats of scRNA expression data into local standard
    - store row labels (genes), column labels (cell ID), and data (expression levels) in one format
    - this format is compressed ".npz" files 
    - save via: save_npz_of_arr_genes_cells(...) calls numpy's savez_compressed(...)
      i.e. np.savez_compressed(npzpath, arr=arr, genes=genes, cells=cells) -- all as numpy arrays
    - load via: load_npz_of_arr_genes_cells(...) calls numpy's load(...)
    - loaded npz acts similar to dictionary
        loaded = np.load(npzpath)
        arr = loaded['arr']
        genes = loaded['genes']
        cells = loaded['cells']
"""
# TODO test and optimize read_exptdata_from_files
# TODO unit tests for data loading


def read_datafile_simple(datapath, verbose=True, txt=False):
    """
    Loads file at datapath, which is either a txt file or npy file
    """
    if txt:
        assert datapath[-4:] == '.txt'
        arr = np.loadtxt(datapath)
    else:
        arr = np.load(datapath)
    if verbose:
        print("Loaded dim %s data at %s" % (arr.shape, datapath))
    return arr


def read_datafile_manual(datapath, verbose=True, save_as_sep_npy=False):
    """
    Datafile form is N+1 x M+1 array with column and row labels (NxM expression matrix)
    Function created for loading mouse cell atlas data formats
    http://bis.zju.edu.cn/MCA/contact.html (DGE matrices)
    """
    assert datapath[-4:] == '.txt'
    with open(datapath) as f:
        count = 0
        # do first pass to get gene count and cell names
        for idx, line in enumerate(f):
            if idx == 0:
                line = line.rstrip()
                line = line.split('\t')
                cell_names = [a.strip('"') for a in line]
            else:
                count += 1
        arr = np.zeros((count, len(cell_names)), dtype=np.int16)  # max size ~ 33k (unsigned twice that)
        gene_names = [0] * count
        print("data size will be (genes x cells):", arr.shape)
        # second pass to get gene names and array data
        f.seek(0)
        for idx, line in enumerate(f):
            if idx > 0:
                if idx % 1000 == 0:
                    print(idx)
                line = line.rstrip()
                line = line.split('\t')
                gene_names[idx-1] = line[0].strip('"')
                arr[idx-1, :] = [int(val) for val in line[1:]]
    if verbose:
        print("Loaded dim %s data at %s" % (arr.shape, datapath))
        print("Max val in arr:", np.max(arr))
        print("Size of arr in memory (bytes)", arr.nbytes)

    datadir = os.path.abspath(os.path.join(datapath, os.pardir))
    if save_as_sep_npy:
        np.save(datadir + os.sep + "raw_arr.npy", arr)
        np.save(datadir + os.sep + "raw_genes.npy", np.array(gene_names))
        np.save(datadir + os.sep + "raw_cells.npy", np.array(cell_names))
    else:
        genes = np.array(gene_names)
        cells = np.array(cell_names)
        np.savez_compressed(datadir + os.sep + "arr_genes_cells_compressed.npz", arr=arr, genes=genes, cells=cells)
    return arr, gene_names, cell_names


def load_singlecell_data(zscore_datafile=RAWDATA_2014MEHTA, savenpz='mems_genes_types_compressed'):
    """
    Returns list of cell types (size p), list of TFs (size N), and xi array where xi_ij is ith TF value in cell type j
    Note the Mehta SI file has odd formatting (use regex to parse); array text file is read in as single line:
    http://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003734.s005&type=supplementary
    """
    gene_labels = []
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines = origline.split('\r')
        for idx_row, row in enumerate(filelines):
            row_split = re.split(r'\t', row)
            if idx_row == 0:  # celltypes line, first row
                celltype_labels = row_split[1:]
            else:
                gene_labels.append(row_split[0])
    # reloop to get data without excessive append calls
    expression_data = np.zeros((len(gene_labels), len(celltype_labels)))
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines_dataonly = origline.split('\r')[1:]
        for idx_row, row in enumerate(filelines_dataonly):
            row_split_dataonly = re.split(r'\t', row)[1:]
            expression_data[idx_row,:] = [float(val) for val in row_split_dataonly]
    if savenpz is not None:
        datadir = os.path.abspath(os.path.join(zscore_datafile, os.pardir))
        npzpath = datadir + os.sep + savenpz
        save_npz_of_arr_genes_cells(npzpath, expression_data, gene_labels, celltype_labels)
    return expression_data, gene_labels, celltype_labels


def load_npz_of_arr_genes_cells(npzpath, verbose=True):
    """
    Can also use for memory array, gene labels, and cell cluster names!
    """
    if verbose:
        print("loading npz of arr genes cells at", npzpath, "...")
    loaded = np.load(npzpath)
    arr = loaded['arr']
    genes = loaded['genes']
    cells = loaded['cells']
    if verbose:
        print("loaded arr, genes, cells:", arr.shape, genes.shape, cells.shape)
    return arr, genes, cells


def save_npz_of_arr_genes_cells(npzpath, arr, genes, cells):
    np.savez_compressed(npzpath, arr=arr, genes=genes, cells=cells)
    return


if __name__ == '__main__':

    # choose pipeline from PIPELINES_VALID
    pipeline = "2018_scMCA"
    assert pipeline in PIPELINES_VALID
    datadir = DATADIR + os.sep + pipeline

    if pipeline == "2018_scMCA":
        datapath = RAWDATA_2018SCMCA
        arr, genes, cells = read_datafile_manual(datapath, verbose=True)
    elif pipeline == "2014_mehta":
        # part 1: load their zscore textfile, save in standard npz format
        expression_data, genes, celltypes = load_singlecell_data(zscore_datafile=RAWDATA_2014MEHTA,
                                                                 savenpz='mems_genes_types_zscore_compressed.npz')
    else:
        # run flags
        flag_load_simple = False
        flag_load_compressed_npz = True

        # simple data load
        if flag_load_simple:
            datapath = "insert path"
            is_txtfile = False
            arr = read_datafile_simple(datapath, verbose=True, txt=is_txtfile)

        if flag_load_compressed_npz:
            compressed_file = datadir + os.sep + "2014_mehta_mems_genes_types_boolean_compressed.npz"
            arr, genes, cells = load_npz_of_arr_genes_cells(compressed_file)

            rowfile = datadir + os.sep + "rows_to_delete.txt"
            rows_to_delete = read_datafile_simple(rowfile, verbose=True, txt=True)
            rows_to_delete = rows_to_delete.astype(int)
            for idx, row in enumerate(rows_to_delete):
                print(row, genes[row])
