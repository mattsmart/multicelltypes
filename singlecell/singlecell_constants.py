import os

from utils.file_io import CELLTYPES

# IO
SINGLECELL = CELLTYPES + os.sep + "singlecell"
DATADIR = CELLTYPES + os.sep + "input"
MEMORIESDIR = DATADIR + os.sep + "memories"
UNFOLDINGDIR = DATADIR + os.sep + "unfolding"

# MAIN MEMORY FILES
MEMS_MEHTA = MEMORIESDIR + os.sep + "2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz"
MEMS_SCMCA = MEMORIESDIR + os.sep + "2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz"

# UNFOLDING DEFAULT
MEMS_UNFOLD = UNFOLDINGDIR + os.sep + "unfold_expC1_mems_genes_types_signals.npz"

# MODEL SPECIFICATION -- TODO print used vars in simsetup dict, write to run_info.txt
DEFAULT_MEMORIES_NPZPATH = MEMS_UNFOLD    # choose which memories to embed
NETWORK_METHOD = "projection"             # supported: 'projection' or 'hopfield'
HOLLOW_INTXN_MATRIX = True                # set diagonals of J_ij to zero (standard: True)
BETA = 2.2*1e1                            # value used in Mehta 2014 (low temperature: BETA=2.2)
FIELD_SIGNAL_STRENGTH = 0.30              # relative strength of exosome local field effect
FIELD_APPLIED_PROTOCOL = None             # e.g. None, 'yamanaka', 'miR_21'
FIELD_APPLIED_STRENGTH = 1.0              # relative strength of manual applied fields
FLAG_BINARY_STATE = True                  # use binarized states (up/down vs continuous)  # TODO unused remove/adjust
FLAG_PRUNE_INTXN_MATRIX = False           # flag for non-eq dilution of the symmetric J intxn matrix
J_RANDOM_DELETE_RATIO = 0.2               # this ratio of elements randomly pruned from J intxn matrix

# SPIN FLIP DYNAMICS -- TODO print used vars in simsetup dict, write to run_info.txt
NUM_FULL_STEPS = 100        # number of full TF grid updates in the single cell simulation
ASYNC_BATCH = True          # options: 'async_indiv' (select one spin at a time) or 'async_batch'
FLAG_BURST_ERRORS = False   # forced spin swaps/errors to randomly apply every T full spin updates
BURST_ERROR_PERIOD = 5      # apply every k 'full spin updates' (k*N individual spin updates)

# DISTINCT COPLOURS
# see https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
DISTINCT_COLOURS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#f032e6', '#46f0f0',
                    '#bcf60c', '#911eb4', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
                    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff',
                    '#000000']
