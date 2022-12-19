import numpy as np
import os

from dataprocess.data_standardize import load_npz_of_arr_genes_cells, save_npz_of_arr_genes_cells
from dataprocess.unfolding_csv_to_npz import load_npz_of_arr_genes_cells_signals
from singlecell.singlecell_constants import \
    NETWORK_METHOD, DEFAULT_MEMORIES_NPZPATH, J_RANDOM_DELETE_RATIO, \
    FLAG_PRUNE_INTXN_MATRIX, MEMORIESDIR, MEMS_UNFOLD
from singlecell.singlecell_linalg import \
    memory_corr_matrix_and_inv, interaction_matrix, predictivity_matrix
from singlecell.singlecell_simsetup_curated import \
    CURATED_XI, CURATED_W, CURATED_CELLTYPE_LABELS, CURATED_GENE_LABELS

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""

REQUIRE_W_SYMMETRY = False


def singlecell_simsetup(flag_prune_intxn_matrix=FLAG_PRUNE_INTXN_MATRIX,
                        npzpath=DEFAULT_MEMORIES_NPZPATH, curated=False,
                        unfolding=False, random_mem=False, random_W=False, housekeeping=0):
    """
    gene_labels, celltype_labels, xi = load_singlecell_data()
    """
    assert NETWORK_METHOD in ["projection", "hopfield"]
    # comments
    if flag_prune_intxn_matrix:
        print("Note FLAG_PRUNE_INTXN_MATRIX is True with ratio %.2f" % J_RANDOM_DELETE_RATIO)
    # unfolding block
    if unfolding:
        print("Using unfolding npz")
        xi, gene_labels, celltype_labels, field_send = \
            load_npz_of_arr_genes_cells_signals(npzpath, verbose=True)
    else:
        assert npzpath != MEMS_UNFOLD
        xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npzpath, verbose=True)
        field_send = None
    # store string arrays as lists (with python3 UTF-8 decoding of strings stored as bytes
    gene_labels = gene_labels.tolist()
    celltype_labels = celltype_labels.tolist()
    if isinstance(gene_labels[0], bytes):
        gene_labels = [a.decode("utf-8") for a in gene_labels]
    if isinstance(celltype_labels[0], bytes):
        celltype_labels = [a.decode("utf-8") for a in celltype_labels]

    if curated:
        print('CURATED selected, resetting simsetup vars')
        xi = CURATED_XI
        field_send = CURATED_W
        celltype_labels = CURATED_CELLTYPE_LABELS
        gene_labels = CURATED_GENE_LABELS
    # model extension block to delete anti-minima
    if housekeeping != 0:
        assert type(housekeeping) == int and housekeeping > 0
        print('Note housekeeping ON, adding %d genes' % housekeeping)
        # need to extend gene_labels, xi, and cell-cell signalling accordingly
        print(type(gene_labels), len(gene_labels))
        gene_labels += ['artificial_%d' % i for i in range(housekeeping)]
        hk_block = np.ones((housekeeping, len(celltype_labels)))
        xi = np.concatenate((xi, hk_block))
        if field_send is not None:
            field_send = np.pad(field_send, ((0, housekeeping), (0, housekeeping)), 'constant')
    N = len(gene_labels)
    P = len(celltype_labels)
    # optional random XI (memories)
    if random_mem:
        print("WARNING: simsetup random_mem, half -1 half 1, size %d x %d" % (N, P))
        xi = np.random.choice([-1, 1], (N, P))
    # optional random W (cell-cell signalling)
    if random_W:
        print("WARNING: simsetup random_W, creating symmetric U[-1,1], size %d x %d" % (N, N))
        W_0 = np.random.rand(N, N)*2 - 1  # scale to Uniform [-1, 1]
        W_lower = np.tril(W_0, k=-1)
        W_diag = np.diag(np.diag(W_0))
        field_send = W_lower + W_lower.T + W_diag
        # subsample block -- randomly remove 2/3 of columns representing non-signalling genes
        """
        print "WARNING: subsampling random W"
        cols_to_remove = np.random.choice(N, int(N*0.67), replace=False)
        field_send[:, cols_to_remove] = 0
        """
    # currently require symmetry of cell-cell signal matrix W
    if field_send is not None:
        W_is_symmetric = np.all(np.abs(field_send - field_send.T) < 1e-8)
        if REQUIRE_W_SYMMETRY:
            assert W_is_symmetric
        else:
            if not W_is_symmetric:
                print('Warning, W_is_symmetric is False in simsetup')
    # data processing into sim object
    xi = xi.astype(np.float64)
    a, a_inv = memory_corr_matrix_and_inv(xi)
    j = interaction_matrix(xi, a_inv, method=NETWORK_METHOD,
                           flag_prune_intxn_matrix=flag_prune_intxn_matrix)
    eta = predictivity_matrix(xi, a_inv)
    if NETWORK_METHOD == "hopfield":
        #a = np.eye(len(celltype_labels))      # identity p x p
        #a_inv = np.eye(len(celltype_labels))  # identity p x p
        #eta = np.copy(xi)
        print("Warning, NOT changing correlation matrix to identity")
    celltype_id = {k: v for v, k in enumerate(celltype_labels)}
    gene_id = {k: v for v, k in enumerate(gene_labels)}
    # store in sim object (currently just a dict)
    simsetup = {
        'memories_path': npzpath,
        'N': N,
        'P': P,
        'GENE_LABELS': gene_labels,
        'CELLTYPE_LABELS': celltype_labels,
        'GENE_ID': gene_id,
        'CELLTYPE_ID': celltype_id,
        'XI': xi,
        'A': a,
        'A_INV': a_inv,
        'J': j,
        'ETA': eta,
        'NETWORK_METHOD': NETWORK_METHOD,
        'FIELD_SEND': field_send,
        'bool_gpu': False,
        'random_mem': random_mem,
        'random_W': random_W,
        'K': housekeeping
    }
    return simsetup


def unpack_simsetup(simsetup):
    N = simsetup['N']
    P = simsetup['P']
    GENE_LABELS = simsetup['GENE_LABELS']
    CELLTYPE_LABELS = simsetup['CELLTYPE_LABELS']
    GENE_ID = simsetup['GENE_ID']
    CELLTYPE_ID = simsetup['CELLTYPE_ID']
    XI = simsetup['XI']
    A = simsetup['A']
    A_INV = simsetup['A_INV']
    J = simsetup['J']
    ETA = simsetup['ETA']
    return N, P, GENE_LABELS, CELLTYPE_LABELS, GENE_ID, CELLTYPE_ID, XI, A, A_INV, J, ETA


if __name__ == '__main__':

    print_genes = True
    print_celltypes = True
    print_gene_expression_row = False
    npzpath_override = False
    npzpath_alternate = MEMORIESDIR + os.sep + \
                        '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
    unfolding = True
    # print block
    if npzpath_override:
        simsetup = singlecell_simsetup(npzpath=npzpath_alternate, unfolding=unfolding)
    else:
        simsetup = singlecell_simsetup(unfolding=unfolding)
    if print_gene_expression_row:
        gene_name = 'S100a4'
        gene_int = simsetup['GENE_ID'][gene_name]
    if print_genes:
        print('Genes:')
        for idx, label in enumerate(simsetup['GENE_LABELS']):
            print(idx, label)
    if print_celltypes:
        print('Celltypes:')
        for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
            if print_gene_expression_row:
                print(idx, label, '|', gene_name, simsetup['XI'][gene_int, idx])
            else:
                print(idx, label)

    edit_npz = False
    # edit npz block
    if edit_npz:
        npz_to_edit = MEMORIESDIR + os.sep + \
                      '2018_scmca_mems_genes_types_boolean_compressed_pruned_AB.npz'
        npz_outname = MEMORIESDIR + os.sep + \
                      '2018_scmca_mems_genes_types_boolean_compressed_pruned_AB_edit.npz'
        xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npz_to_edit, verbose=True)
        # edits go here
        celltype_labels[58] = 'Mixed NK T cell'
        save_npz_of_arr_genes_cells(npz_outname, xi, gene_labels, celltype_labels)
        xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npz_outname, verbose=True)
        if print_celltypes:
            print('Celltypes:')
            for idx, label in enumerate(celltype_labels):
                print(idx, label)

    projection_check = True
    if projection_check:
        N = simsetup['N']
        P = simsetup['P']
        state_check = np.array([-1,-1,-1, 1,1,1, -1,-1,-1])
        overlaps = np.dot(simsetup['XI'].T, state_check) / N
        print('overlaps', overlaps)
        proj = np.dot(simsetup['A_INV'], overlaps)
        print('proj', proj)
