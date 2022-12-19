import numpy as np


# Constants for the curation dict of presets
MULTICELL_PRESET = '3MemSym'  # ferro
N_FERRO = 9
REFINE_W = True
DIAG_W = True
RANDOM_W = False  # TODO care how this interacts with random W in singlecell_simsetup

curated = {
    'mutual_inhibition':
        {'XI': np.array([
            [1, -1],  # TF 1
            [-1, 1],  # TF 2
            [1, -1],  # identity gene
            [-1, 1],  # identity gene
            [1, 1],   # housekeeping gene (hardcoded)
            [1, 1]]),  # housekeeping gene (hardcoded)
            #    when gene 0 (col 1) is ON as in mem A, it promoted mem A and inhibits mem B
            #    when gene 1 (col 2) is ON as in mem B, it promoted mem A and inhibits mem B
          'W': np.array([
            [-5, 0, 0, 0, 0, 0],
            [0, -5, 0, 0, 0, 0],
            [-5, 5, 0, 0, 0, 0],
            [5, -5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]),
          'celltype_labels': ['mem_A', 'mem_B'],
          'gene_labels': ['A_signal', 'B_signal', 'A_identity', 'B_identity', 'HK_1', 'HK_2']
         },
    'ferro':
        {'XI': np.ones((N_FERRO, 1)),
          'W': np.zeros((N_FERRO, N_FERRO)),
          'celltype_labels': [r'$\xi$'],
          'gene_labels': ['gene_%d' % idx for idx in range(N_FERRO)],
         },
    'ferroPerturb':
        {'XI': np.array([[1], [1], [1], [1], [2.5]]),
         'W': np.zeros((5, 5)),
         'celltype_labels': [r'$\xi$'],
         'gene_labels': ['gene_%d' % idx for idx in range(5)],
         },
    '3MemOrthog':
        {'XI': np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1],
                         [ 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1]]).T,
         'W': np.zeros((12, 12)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(12)],
         },
    '3MemCorr':
        {'XI': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1],
                         [1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1]]).T,
         'W': np.zeros((12, 12)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(12)],
         },
    '3MemSym':
        {'XI': np.array([[1, 1, 1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, 1, 1, 1]]).T,
         'W': np.zeros((9, 9)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(9)],
         },
    '3MemCorrPerturb':
        {'XI': np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [4, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                         [1, 2, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]]).T,
         'W': np.zeros((12, 12)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(12)],
         },
    '3Mem2019':
        {'XI': np.array([[1, -1, -1, 1, -1, 1, 1, 1, 1],
                         [-1, 1, 1, -1, -1, 1, 1, -1, -1],
                         [1, 1, 1, -1, -1, 1, -1, -1, -1]]).T,
         'W': np.zeros((9, 9)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(9)],
         }
}

assert MULTICELL_PRESET in list(curated.keys())

if REFINE_W:
    # manually refine the W matrix of the chosen scheme
    Ntot = curated[MULTICELL_PRESET]['XI'].shape[0]
    if RANDOM_W:
        assert not DIAG_W
        W_0 = np.random.rand(Ntot, Ntot) * 2 - 1  # scale to Uniform [-1, 1]
        W_lower = np.tril(W_0, k=-1)
        W_diag = np.diag(np.diag(W_0))
        curated[MULTICELL_PRESET]['W'] = (W_lower + W_lower.T + W_diag) / Ntot
    elif DIAG_W:
        curated[MULTICELL_PRESET]['W'] = np.eye(Ntot)
    else:
        curated[MULTICELL_PRESET]['W'][1, 1] = 10.0 / Ntot
        curated[MULTICELL_PRESET]['W'][2, 3] = -10.0 / Ntot
        curated[MULTICELL_PRESET]['W'][3, 2] = -10.0 / Ntot

CURATED_XI = curated[MULTICELL_PRESET]['XI']
CURATED_W = curated[MULTICELL_PRESET]['W']
CURATED_CELLTYPE_LABELS = curated[MULTICELL_PRESET]['celltype_labels']
CURATED_GENE_LABELS = curated[MULTICELL_PRESET]['gene_labels']
