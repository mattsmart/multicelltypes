import os

"""
Store constants and default parameters here
"""

# io locations
OUTPUTDIR = "runs"
DATADIR = ".." + os.sep + "input"

PIPELINES_VALID = ["2014_mehta", "2018_scMCA", "misc"]
PIPELINES_DIRS = {k: DATADIR + os.sep + k for k in PIPELINES_VALID}

# 2014 mehta data
RAWDATA_2014MEHTA = DATADIR + os.sep + "2014_mehta" + os.sep + "SI_mehta_zscore_table.txt"
NPZ_2014MEHTA_ORIG = DATADIR + os.sep + "2014_mehta" + os.sep + "mems_genes_types_zscore_compressed.npz"
NPZ_2014MEHTA_MEMS = DATADIR + os.sep + "2014_mehta" + os.sep + "mems_genes_types_compressed.npz"
NPZ_2014MEHTA_MEMS_PRUNED = DATADIR + os.sep + "2014_mehta" + os.sep + "mems_genes_types_compressed_pruned_A.npz"

# 2018 scMCA data
RAWDATA_2018SCMCA = DATADIR + os.sep + "2018_scMCA" + os.sep + "SI_Figure2-batch-removed.txt"
CELLSTOCLUSTERS_2018SCMCA = DATADIR + os.sep + "2018_scMCA" + os.sep + "SI_cells_to_clusters.csv"
NPZ_2018SCMCA_ORIG = DATADIR + os.sep + "2018_scMCA" + os.sep + "arr_genes_cells_compressed.npz"
NPZ_2018SCMCA_ORIG_WITHCLUSTER = DATADIR + os.sep + "2018_scMCA" + os.sep + "arr_genes_cells_withcluster_compressed.npz"
NPZ_2018SCMCA_PRUNED = DATADIR + os.sep + "2018_scMCA" + os.sep + "arr_genes_cells_compressed_pruned_A.npz"
NPZ_2018SCMCA_MEMS = DATADIR + os.sep + "2018_scMCA" + os.sep + "mems_genes_types_compressed.npz"
NPZ_2018SCMCA_MEMS_PRUNED = DATADIR + os.sep + "2018_scMCA" + os.sep + "mems_genes_types_compressed_pruned_A.npz"
