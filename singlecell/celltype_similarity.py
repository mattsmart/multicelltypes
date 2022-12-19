import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell.analysis_basin_plotting import plot_overlap_grid
from singlecell.singlecell_constants import DATADIR, MEMORIESDIR
from utils.file_io import RUNS_FOLDER
from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_simsetup_query import collect_mygene_hits, write_genelist_id_csv, read_gene_list_csv, check_target_in_gene_id_dict
from singlecell.singlecell_visualize import plot_as_radar, plot_as_bar


def write_celltype_csv(gene_labels, celltype_state, csvpath):
    with open(csvpath, 'w') as f:
        for idx, label in enumerate(gene_labels):
            line = '%s,%d\n' % (label, celltype_state[idx])
            f.write(line)
    return csvpath


def load_external_celltype(csvpath):
    gene_labels_raw = []
    gene_states_raw = []
    with open(csvpath, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            assert len(r) == 2
            gene_labels_raw.append(r[0])
            gene_states_raw.append(float(r[1]))
    return gene_labels_raw, gene_states_raw


def build_entrez_synonyms_for_celltype(gene_labels_raw, entrez_name='entrez_id_ext_celltype.csv'):
    entrez_path = DATADIR + os.sep + 'misc' + os.sep + 'genelist_entrezids' + os.sep + entrez_name
    if os.path.exists(entrez_path):
        print("Warning, path %s exists" % entrez_path)
        print("Delete file to have it be remade (using existing one)")
    else:
        gene_hits, hitcounts = collect_mygene_hits(gene_labels_raw)
        write_genelist_id_csv(gene_labels_raw, gene_hits, outpath=entrez_path)
    return entrez_path


def truncate_celltype_data(matches, simsetup, gene_labels_raw, gene_states_raw):
    print('Loaded gene count from external celltype:', len(gene_labels_raw))
    print('Loaded gene count from memories:', len(simsetup['GENE_LABELS']))
    print('Number of matches, with possible redundancies:', len(matches))
    print('NOTE: choosing FIRST gene match in cases of degenerate matches')
    new_memory_rows = []
    new_celltype_rows = []
    for match in matches:
        matched_memory_gene = match[0]
        matched_celltype_gene = match[1]
        matched_memory_gene_idx = simsetup['GENE_ID'][matched_memory_gene]
        matched_celltype_gene_idx = gene_labels_raw.index(matched_celltype_gene)
        if matched_celltype_gene_idx in new_celltype_rows or matched_memory_gene_idx in new_memory_rows:
            continue
        else:
            new_memory_rows.append(matched_memory_gene_idx)
            new_celltype_rows.append(matched_celltype_gene_idx)
    assert len(new_memory_rows) == len(new_celltype_rows)
    print('Number of genes which match simsetup gene list:', len(new_memory_rows))
    xi_truncated = simsetup['XI'][new_memory_rows, :]
    ext_gene_states_truncated = np.array(gene_states_raw)[new_celltype_rows]
    print(xi_truncated.shape, ext_gene_states_truncated.shape)
    return xi_truncated, ext_gene_states_truncated


def truncate_celltype_data_for_dual_npz(matches, simsetup_A, simsetup_B):
    print('Loaded gene count from npz A:', len(simsetup_A['GENE_LABELS']))
    print('Loaded gene count from npz B:', len(simsetup_B['GENE_LABELS']))
    print('Number of matches, with possible redundancies:', len(matches))
    print('NOTE: choosing FIRST gene match in cases of degenerate matches')
    new_npzA_rows = []
    new_npzB_rows = []
    for match in matches:
        matched_A_gene = match[0]
        matched_B_gene = match[1]
        matched_A_gene_idx = simsetup_A['GENE_ID'][matched_A_gene]
        matched_B_gene_idx = simsetup_B['GENE_ID'][matched_B_gene]
        if matched_B_gene_idx in new_npzB_rows or matched_A_gene_idx in new_npzA_rows:
            continue
        else:
            new_npzA_rows.append(matched_A_gene_idx)
            new_npzB_rows.append(matched_B_gene_idx)
    assert len(new_npzA_rows) == len(new_npzB_rows)
    print('Number of genes which match both simsetup gene lists:', len(new_npzA_rows))
    xi_truncated_A = simsetup_A['XI'][new_npzA_rows, :]
    xi_truncated_B = simsetup_B['XI'][new_npzB_rows, :]
    print(xi_truncated_A.shape, xi_truncated_B.shape)
    return xi_truncated_A, xi_truncated_B


def compute_and_plot_score(simsetup, xi_truncated, ext_gene_states_truncated, celltype_name='unspecified_celltype', use_proj=False):
    scorelabel = 'Overlaps'
    filemod = 'overlap'
    scores = np.dot(xi_truncated.T, ext_gene_states_truncated) / ext_gene_states_truncated.shape[0]
    if use_proj:
        scorelabel = 'Projections'
        filemod = 'proj'
        scores = np.dot(simsetup['A_INV'], scores)
    for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
        print("Celltype %s, %s %.3f" % (label, scorelabel, scores[idx]))
    # plot overlaps
    plot_as_bar(scores, simsetup['CELLTYPE_LABELS'])
    plt.title('%s between %s and loaded memories (num matching genes: %d)' % (scorelabel, celltype_name, len(ext_gene_states_truncated)))
    plt.savefig(os.path.join(os.path.dirname(external_celltype), 'score_%s_vs_mems_%s.png' % (celltype_name, filemod)))
    plt.show()
    return


def score_similarity(external_celltype, memories_npz, memories_entrez_path, celltype_name='unspecified_celltype', use_proj=False):
    # load external celltype gene labels and states
    gene_labels_raw, gene_states_raw = load_external_celltype(external_celltype)
    # build synonym list from the gene labels
    ext_entrez_path = build_entrez_synonyms_for_celltype(gene_labels_raw,
                                                          entrez_name='entrez_id_ext_celltype_%s.csv' % celltype_name)
    # reload synonym lists for memories and ext celltype
    target_genes_id = read_gene_list_csv(ext_entrez_path, aliases=True)
    memories_genes_id = read_gene_list_csv(memories_entrez_path, aliases=True)
    # compare synonym lists to get matches
    matches = check_target_in_gene_id_dict(memories_genes_id, target_genes_id,
                                           outpath='tmp_matches_%s_vs_memories.txt' % (celltype_name))
    # use matches to truncate ext gene states to match truncated memory states and
    simsetup = singlecell_simsetup(npzpath=memories_npz)
    xi_truncated, ext_gene_states_truncated = truncate_celltype_data(matches, simsetup, gene_labels_raw, gene_states_raw)
    # use truncated data to score overlaps
    compute_and_plot_score(simsetup, xi_truncated, ext_gene_states_truncated, use_proj=use_proj, celltype_name=celltype_name)
    return simsetup, xi_truncated, ext_gene_states_truncated


def plot_overlaps_for_npz(xi, celltypes, plotname, outdir, namemod='_orig_overlap'):
    grid_data = np.dot(xi.T, xi) / xi.shape[0]
    plot_overlap_grid(grid_data, celltypes, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=plotname, ext='.pdf', vforce=None, namemod=namemod)
    return


def plot_all_grids(simsetup_A, simsetup_B, xi_truncated_A, xi_truncated_B, A_label, B_label, outdir=RUNS_FOLDER):
    """
    Creates the following grids
    - A auto orig            - overlap only (skip proj, just diagonal)
    - B auto orig            - overlap only (skip proj, just diagonal)
    - A auto truncated       - overlaps + untruncated projection measure (re-created proj would be diagonal)
    - B auto truncated       - overlaps + untruncated projection measure (re-created proj would be diagonal)
    - A vs B                 - overlaps + proj_A + proj_B
    """
    xi_A = simsetup_A['XI']
    xi_B = simsetup_B['XI']
    celltypes_A = simsetup_A['CELLTYPE_LABELS']
    celltypes_B = simsetup_B['CELLTYPE_LABELS']
    # plot A auto orig
    grid_data_overlap_A = np.dot(xi_A.T, xi_A) / xi_A.shape[0]
    plot_overlap_grid(grid_data_overlap_A, celltypes_A, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=A_label, ext='.pdf', vforce=None, namemod='_orig_overlap')
    # plot B auto orig
    grid_data_overlap_B = np.dot(xi_B.T, xi_B) / xi_B.shape[0]
    plot_overlap_grid(grid_data_overlap_B, celltypes_B, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=B_label, ext='.pdf', vforce=None, namemod='_orig_overlap')
    # plot A auto truncated
    grid_data_overlap_A_trunc = np.dot(xi_truncated_A.T, xi_truncated_A) / xi_truncated_A.shape[0]
    grid_data_pseudoproj_A_trunc = np.dot(simsetup_A['A_INV'], grid_data_overlap_A_trunc)
    plot_overlap_grid(grid_data_overlap_A_trunc, celltypes_A, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=A_label, ext='.pdf', vforce=None, namemod='_overlap_truncated')
    plot_overlap_grid(grid_data_pseudoproj_A_trunc, celltypes_A, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=A_label, ext='.pdf', vforce=None, namemod='_pseudoproj_truncated')
    # plot B auto truncated
    grid_data_overlap_B_trunc = np.dot(xi_truncated_B.T, xi_truncated_B) / xi_truncated_B.shape[0]
    grid_data_pseudoproj_B_trunc = np.dot(simsetup_B['A_INV'], grid_data_overlap_B_trunc)
    plot_overlap_grid(grid_data_overlap_B_trunc, celltypes_B, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=B_label, ext='.pdf', vforce=None, namemod='_overlap_truncated')
    plot_overlap_grid(grid_data_pseudoproj_B_trunc, celltypes_B, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=B_label, ext='.pdf', vforce=None, namemod='_pseudoproj_truncated')
    # plot A vs B
    grid_data_overlap_B_trunc = np.dot(xi_truncated_B.T, xi_truncated_B) / xi_truncated_B.shape[0]
    grid_data_pseudoproj_B_trunc = np.dot(simsetup_B['A_INV'], grid_data_overlap_B_trunc)
    plot_overlap_grid(grid_data_overlap_B_trunc, celltypes_B, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=B_label, ext='.pdf', vforce=None, namemod='_overlap_truncated')
    plot_overlap_grid(grid_data_pseudoproj_B_trunc, celltypes_B, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=B_label, ext='.pdf', vforce=None, namemod='_pseudoproj_truncated',)
    # memory_labels_y=celltypes_B
    # gen AB data
    grid_B_overlap_A = np.dot(xi_truncated_A.T, xi_truncated_B) / np.sqrt(xi_truncated_A.shape[0] * xi_truncated_B.shape[0])
    grid_B_apply_proj_A = np.dot(simsetup_A['A_INV'], grid_B_overlap_A)
    grid_A_apply_proj_B = np.dot(simsetup_B['A_INV'], grid_B_overlap_A.T)
    # plot AB data
    label_1 = '%s_overlap_%s' % (B_label, A_label)
    label_2 = '%s_apply_proj_%s' % (B_label, A_label)
    label_3 = '%s_apply_proj_%s' % (A_label, B_label)
    plot_overlap_grid(grid_B_overlap_A, celltypes_A, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=label_1, ext='.pdf', vforce=None, namemod='_matching', memory_labels_x=celltypes_B)
    plot_overlap_grid(grid_B_apply_proj_A, celltypes_A, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=label_2, ext='.pdf', vforce=None, namemod='_matching', memory_labels_x=celltypes_B)
    plot_overlap_grid(grid_A_apply_proj_B, celltypes_B, outdir, ax=None, N=None, normalize=True, fs=9, relmax=True,
                      extragrid=False, plotname=label_3, ext='.pdf', vforce=None, namemod='_matching', memory_labels_x=celltypes_A)
    return


if __name__ == '__main__':
    create_ext_celltype = False
    score_ext_celltype = False
    score_local_celltype = False
    check_npz_grids = True
    compare_two_npz = False

    # local constants
    outdir = RUNS_FOLDER + os.sep + 'celltype_similarity'
    external_celltypes_dir = DATADIR + os.sep + 'misc' + os.sep + 'external_celltypes'
    use_proj = True

    if create_ext_celltype:
        celltype_choice = 'fibroblast - skin'
        csvpath = external_celltypes_dir + os.sep + '2014mehta_dermal_fibroblast.csv'
        memories_npz = MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz'
        simsetup = singlecell_simsetup(npzpath=memories_npz)
        gene_labels = simsetup['GENE_LABELS']
        celltype_choice_idx = simsetup['CELLTYPE_ID'][celltype_choice]
        celltype_state = simsetup['XI'][:, celltype_choice_idx]
        write_celltype_csv(gene_labels, celltype_state, csvpath)

    if score_ext_celltype:
        # settings
        ext_celltype_name = 'dermal_fibroblast'
        external_celltype = external_celltypes_dir + os.sep + '2014mehta_dermal_fibroblast.csv'
        memories_npz = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
        memories_entrez_path = DATADIR + os.sep + 'misc' + os.sep + 'genelist_entrezids' + os.sep + 'entrez_id_2018scMCA_pruned_TFonly.csv'
        # scoring
        print('Scoring similarity between %s and celltypes in %s...' % (external_celltype, memories_npz))
        simsetup, xi_truncated, ext_gene_states_truncated = \
            score_similarity(external_celltype, memories_npz, memories_entrez_path, celltype_name=ext_celltype_name, use_proj=True)
        compute_and_plot_score(simsetup, xi_truncated, ext_gene_states_truncated, celltype_name=ext_celltype_name, use_proj=False)

    if score_local_celltype:
        memories_npz = MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz'
        simsetup = singlecell_simsetup(npzpath=memories_npz)
        local_celltype = 'fibroblast - skin'
        local_celltype_idx = simsetup['CELLTYPE_ID'][local_celltype]
        # use truncated data to score overlaps
        local_celltype_data = simsetup['XI'][:, local_celltype_idx]
        compute_and_plot_score(simsetup, simsetup['XI'], local_celltype_data, celltype_name=local_celltype, use_proj=True)
        compute_and_plot_score(simsetup, simsetup['XI'], local_celltype_data, celltype_name=local_celltype, use_proj=False)

    if check_npz_grids:
        npzs = {'2014': MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed.npz',
                '2014pruneA': MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz',
                '2018': MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed.npz',
                '2018pruneA': MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A.npz',
                '2018_TF': MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_TFonly.npz',
                '2018pruneA_TF': MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'}
        for k, v in npzs.items():
            simsetup = singlecell_simsetup(npzpath=v)
            plot_overlaps_for_npz(simsetup['XI'], simsetup['CELLTYPE_LABELS'], k, outdir, namemod='_orig_overlap')

    if compare_two_npz:
        # settings
        label_A = '2018pruneA_TF'
        label_B = '2014pruneA'
        npz_A = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
        npz_B = MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz'
        entrez_name_A = 'entrez_id_2018scMCA_pruned_TFonly.csv'
        entrez_name_B = 'entrez_id_2014mehta.csv'
        # perform simsetup
        simsetup_A = singlecell_simsetup(npzpath=npz_A)
        simsetup_B = singlecell_simsetup(npzpath=npz_B)
        # create synonym list if needed (for Mehta and scMCA they should exist already)
        entrez_path_A = build_entrez_synonyms_for_celltype(simsetup_A['GENE_LABELS'], entrez_name=entrez_name_A)
        entrez_path_B = build_entrez_synonyms_for_celltype(simsetup_B['GENE_LABELS'], entrez_name=entrez_name_B)
        # reload synonym lists for memories and ext celltype
        gene_synonyms_A = read_gene_list_csv(entrez_path_A, aliases=True)
        gene_synonyms_B = read_gene_list_csv(entrez_path_B, aliases=True)
        # compare synonym lists to get matches
        matches = check_target_in_gene_id_dict(gene_synonyms_A, gene_synonyms_B, outpath='tmp_matches_npz_A_vs_B.txt')
        # use matches to truncate ext gene states to match truncated memory states and
        xi_truncated_A, xi_truncated_B = truncate_celltype_data_for_dual_npz(matches, simsetup_A, simsetup_B)
        # do proj and overlaps for both
        plot_all_grids(simsetup_A, simsetup_B, xi_truncated_A, xi_truncated_B, label_A, label_B, outdir=outdir)
