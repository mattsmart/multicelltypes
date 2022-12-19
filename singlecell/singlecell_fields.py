from matplotlib import pyplot as plt
import numpy as np
import os
from random import random

from singlecell.analysis_basin_plotting import plot_overlap_grid
from singlecell.singlecell_constants import BETA, FIELD_SIGNAL_STRENGTH, MEMS_MEHTA, MEMS_SCMCA, FIELD_APPLIED_PROTOCOL, MEMORIESDIR
from utils.file_io import RUNS_FOLDER
from singlecell.singlecell_functions import hamiltonian
from singlecell.singlecell_simsetup import singlecell_simsetup, unpack_simsetup
from singlecell.singlecell_visualize import plot_as_bar


EXPT_FIELDS = {
    # mir 21 field note:
    #   level 1 is main ref
    #   level 2 adds wiki mir21
    #   level 4 adds targetscan hits
    'miR_21': {
        '2014mehta': {
            'level_1': ['Klf5'],
            'level_2': ['Klf5', 'Trp63', 'Mef2c'],
            'level_3': ['Klf5', 'Trp63', 'Mef2c', 'Smarcd1', 'Crebl2', 'Thrb', 'Nfat5', 'Gata2', 'Nkx6-1', 'Terf2',
                        'Zkscan5', 'Glis2', 'Egr3', 'Foxp2', 'Smad7', 'Tbx2', 'Cbx4', 'Myt1l', 'Satb1', 'Yap1', 'Foxp1',
                        'Foxg1', 'Pcbd1', 'Bahd1', 'Bcl11b', 'Pitx2', 'Sox7', 'Sox5', 'Alx1', 'Npas3', 'Adnp', 'Klf6',
                        'Sox2', 'Klf3', 'Msx1', 'Plag1', 'Osr1', 'Mycl1', 'Nfib', 'Nfia', 'Bnc2']},
        '2018scMCA': {
            'level_1': ['Klf5', 'Pten'],
            'level_2': ['Klf5', 'Pten', 'Anp32a', 'Hnrnpk', 'Mef2c', 'Pdcd4', 'Smarca4', 'Trp63'],
            'level_3': ['Klf5', 'Pten', 'Anp32a', 'Hnrnpk', 'Mef2c', 'Pdcd4', 'Smarca4', 'Trp63', 'Adnp', 'Ago2', 'Alx1',
                        'Asf1a', 'Bcl11b', 'Bnc2', 'Cbx4', 'Chd7', 'Cnot6', 'Crebl2', 'Crebrf', 'Csrnp3', 'Egr3', 'Elf2',
                        'Foxg1', 'Foxp1', 'Foxp2', 'Gata2', 'Gatad2b', 'Glis2', 'Hipk3', 'Hnrnpu', 'Kdm7a', 'Klf3',
                        'Klf6', 'Lcor', 'Msx1', 'Mycl', 'Myt1l', 'Nfat5', 'Nfia', 'Nfib', 'Nipbl', 'Nkx6-1', 'Notch2',
                        'Npas3', 'Osr1', 'Pbrm1', 'Pcbd1', 'Pdcd4', 'Peli1', 'Pik3r1', 'Pitx2', 'Plag1', 'Pspc1', 'Pura',
                        'Purb', 'Purg', 'Rbpj', 'Rnf111', 'Satb1', 'Ski', 'Smad7', 'Smarcd1', 'Sox2', 'Sox2ot', 'Sox5',
                        'Sox7', 'Stat3', 'Suz12', 'Tbx2', 'Terf2', 'Thrb', 'Tnks', 'Trim33', 'Wwp1', 'Yap1', 'Zfp36l2',
                        'Zkscan5', 'Zfp367']}
    },
    # yamanaka field notes: Pou5f1 is alias Oct4, these are OSKM + Nanog
    'yamanaka': {
        '2014mehta': {
            'level_1': ['Sox2', 'Pou5f1', 'Klf4', 'Myc'],
            'level_2': ['Sox2', 'Pou5f1', 'Klf4', 'Myc', 'Nanog']},
        '2018scMCA': {
            'level_1': ['Sox2', 'Pou5f1', 'Klf4', 'Myc'],
            'level_2': ['Sox2', 'Pou5f1', 'Klf4', 'Myc', 'Nanog']},
    },
    # empty field list for None protocol
    None: {}
}


def construct_app_field_from_genes(gene_name_effect, gene_id, num_steps=0):
    """
    Args:
    - gene_name_effect: dict of gene_name: +-1 (on or off)
    - gene_id: map of gene name to idx for the input memories file
    - num_steps: optional numsteps (return 2d array if nonzero)
    Return:
    - applied field array of size N x 1 or N x num_steps
    """
    print("Constructing applied field:")
    N = len(list(gene_id.keys()))
    #app_field = np.zeros((N, num_steps))  $ TODO implement time based
    app_field = np.zeros(N)
    for label, effect in gene_name_effect.items():
        if label in list(gene_id.keys()):
            #print label, gene_id[label], 'effect:', effect
            app_field[gene_id[label]] += effect
        else:
            print("Field construction warning: label %s not in gene_id.keys()" % label)
    return app_field


def field_setup(simsetup, protocol=FIELD_APPLIED_PROTOCOL, level=None):
    """
    Construct applied field vector (either fixed or on varying under a field protocol) to bias the dynamics
    Notes on named fields
    - Yamanaka factor (OSKM) names in mehta datafile: Sox2, Pou5f1 (oct4), Klf4, Myc, also nanog
    """
    # TODO must optimize: naive implement brings i7-920 row: 16x200 from 56sec (None field) to 140sec (not parallel)
    # TODO support time varying cleanly
    # TODO speedup: initialize at the same time as simsetup
    # TODO speedup: pre-multiply the fields so it need not to be scaled each glauber step (see singlecell_functions.py)
    # TODO there are two non J_ij fields an isolated single cell experiences: TF explicit mod and type biasing via proj
    # TODO     need to include the type biasing one too
    assert protocol in ["yamanaka", "miR_21", None]
    field_dict = {'protocol': protocol,
                  'time_varying': False,
                  'app_field': None,
                  'app_field_strength': 1e5}  # TODO calibrate this to be very large compared to J*s scale
    gene_id = simsetup['GENE_ID']

    # preamble
    if simsetup['memories_path'] == MEMS_MEHTA:
        npz_label = '2014mehta'
    elif simsetup['memories_path'] == MEMS_SCMCA:
        npz_label = '2018scMCA'
    else:
        print("Note npz mems not supported:", simsetup['memories_path'])
        npz_label = None
    if level is None:
        print("Warning: Arg 'level' is None -- setting field level to 'level_1'")
        level = 'level_1'

    if protocol == "yamanaka":
        print("Note: field_setup using", protocol, npz_label, level)
        field_genes = EXPT_FIELDS[protocol][npz_label][level]
        field_genes_effects = {label: 1.0 for label in field_genes}  # this ensure all should be ON
        app_field_start = construct_app_field_from_genes(field_genes_effects, gene_id, num_steps=0)
        field_dict['app_field'] = app_field_start
    elif protocol == 'miR_21':
        """
        - 2018 Nature comm macrophage -> fibroblast paper lists KLF-5 and PTEN as primary targets of miR-21
        - 2014 mehta dataset does not contain PTEN, but 2018 scMCA does
        """
        print("Note: field_setup using", protocol, npz_label, level)
        field_genes = EXPT_FIELDS[protocol][npz_label][level]
        field_genes_effects = {label: -1.0 for label in field_genes}  # this ensure all should be OFF
        app_field_start = construct_app_field_from_genes(field_genes_effects, gene_id, num_steps=0)
        field_dict['app_field'] = app_field_start
    else:
        assert protocol is None
    return field_dict


if __name__ == '__main__':
    # local defs
    npz_mehta = MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz'
    npz_scmca = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
    FIELD_EFFECT_FOLDER = RUNS_FOLDER + os.sep + 'field_effect'

    # settings
    plot_field_impact_all = False
    plot_specific_field = True

    def make_field_plots(field_type, field_level, npz_type, simsetup, outdir=FIELD_EFFECT_FOLDER):
        plot_subtitle = "Field effect of %s, %s on %s" % (field_type, field_level, npz_type)
        print(plot_subtitle)
        field_dict = field_setup(simsetup, protocol=field_type, level=field_level)
        app_field_vector = field_dict['app_field']
        xi_orig = simsetup['XI']
        xi_under_field = np.zeros(xi_orig.shape)
        if app_field_vector is None:
            app_field_vector = np.zeros(xi_orig.shape[0])
        print(app_field_vector.shape)
        for idx in range(app_field_vector.shape[0]):
            if app_field_vector[idx] == 0:
                xi_under_field[idx, :] = xi_orig[idx, :]
            else:
                xi_under_field[idx, :] = app_field_vector[idx]
        # compute field term
        field_term = np.dot(xi_orig.T, app_field_vector)
        plot_as_bar(field_term, simsetup['CELLTYPE_LABELS'])
        plt.axhline(y=0.0, linewidth=1, color='k', linestyle='--')
        plt.title('%s field term xi^T dot h (unperturbed=%.2f)' % (plot_subtitle, 0.0))
        filepath = outdir + os.sep + 'mems_%s_field_term_%s_%s' % (npz_type, field_type, field_level)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        # compute energies of shifted celltypes
        E0 = -0.5 * xi_orig.shape[0] + 0.5 * xi_orig.shape[1]  # i.e. -N/2 + p/2
        energies = np.zeros(xi_orig.shape[1])
        for col in range(xi_orig.shape[1]):
            energies[col] = hamiltonian(xi_under_field[:, col], simsetup['J']) - field_term[col]
        plot_as_bar(energies, simsetup['CELLTYPE_LABELS'])
        plt.axhline(y=E0, linewidth=1, color='k', linestyle='--')
        plt.title('%s minima depth (unperturbed=%.2f)' % (plot_subtitle, E0))
        plt.ylim(E0 * 1.05, 0.8 * np.max(energies))
        filepath = outdir + os.sep + 'mems_%s_energy_under_field_%s_%s' % (npz_type, field_type, field_level)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        # compute overlaps of shifted celltypes
        self_overlaps = np.zeros(xi_orig.shape[1])
        for idx in range(xi_orig.shape[1]):
            self_overlaps[idx] = np.dot(xi_orig[:, idx], xi_under_field[:, idx]) / xi_orig.shape[0]
        plot_as_bar(self_overlaps, simsetup['CELLTYPE_LABELS'])
        plt.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
        plt.title('%s overlaps (unperturbed=%.2f)' % (plot_subtitle, 1.0))
        plt.ylim(0.8 * np.min(self_overlaps), 1.01)
        filepath = outdir + os.sep + 'mems_%s_overlap_under_field_%s_%s' % (npz_type, field_type, field_level)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        # compute projections of shifted celltypes
        self_proj = np.zeros(xi_orig.shape[1])
        for idx in range(xi_orig.shape[1]):
            proj_vector_of_shifted_mem = np.dot(simsetup['A_INV'], np.dot(xi_orig.T, xi_under_field[:, idx])) / \
                                         xi_orig.shape[0]
            self_proj[idx] = proj_vector_of_shifted_mem[idx]
        plot_as_bar(self_proj, simsetup['CELLTYPE_LABELS'])
        plt.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
        plt.title('%s projections (unperturbed=%.2f)' % (plot_subtitle, 1.0))
        plt.ylim(0.8 * np.min(self_proj), 1.01)
        filepath = outdir + os.sep + 'mems_%s_proj_under_field_%s_%s' % (npz_type, field_type, field_level)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        # compute celltype specific overlaps of shifted celltypes
        cell_idx_A = 7
        cell_idx_B = 86
        print(simsetup['CELLTYPE_LABELS'][cell_idx_A], simsetup['CELLTYPE_LABELS'][cell_idx_B])
        hetero_overlaps_A = np.zeros(xi_orig.shape[1])
        hetero_overlaps_B = np.zeros(xi_orig.shape[1])
        for idx in range(xi_orig.shape[1]):
            hetero_overlaps_A[idx] = np.dot(xi_orig[:, cell_idx_A], xi_under_field[:, idx]) / xi_orig.shape[0]
            hetero_overlaps_B[idx] = np.dot(xi_orig[:, cell_idx_B], xi_under_field[:, idx]) / xi_orig.shape[0]
        plot_as_bar(hetero_overlaps_A, simsetup['CELLTYPE_LABELS'], alpha=0.8)
        plot_as_bar(hetero_overlaps_B, simsetup['CELLTYPE_LABELS'], alpha=0.8)
        plt.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
        plt.title('%s hetero_overlaps (unperturbed=%.2f)' % (plot_subtitle, 1.0))
        #plt.ylim(0.8 * np.min(self_overlaps), 1.01)
        filepath = outdir + os.sep + 'mems_%s_hetero_overlaps_under_field_%s_%s' % \
                   (npz_type, field_type, field_level)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        # compute grid under the field
        grid_data = np.dot(xi_under_field.T, xi_under_field) - np.dot(xi_orig.T, xi_orig)
        plot_overlap_grid(grid_data, simsetup['CELLTYPE_LABELS'], outdir,
                          ax=None, N=None, normalize=True, fs=9,
                          relmax=True, extragrid=False, ext='.pdf', vforce=None,
                          plotname='overlap_diff_under_field_%s_%s_%s' %
                                   (npz_type, field_type, field_level))
        return

    if plot_field_impact_all:
        npz = npz_scmca
        if npz == npz_scmca:
            npz_type = '2018scMCA'
        else:
            npz_type = '2014mehta'
        simsetup = singlecell_simsetup(npzpath=npz)
        for field_type in list(EXPT_FIELDS.keys()):
            if field_type is None:
                continue
            field_levels_dict = EXPT_FIELDS[field_type][npz_type]
            for field_level in list(field_levels_dict.keys()):
                make_field_plots(field_type, field_level, npz_type, simsetup)

    if plot_specific_field:
        # npz load
        npz = npz_scmca
        if npz == npz_scmca:
            npz_type = '2018scMCA'
        else:
            npz_type = '2014mehta'
        simsetup = singlecell_simsetup(npzpath=npz)
        xi_orig = simsetup['XI']
        # field choose
        field_type = 'miR_21'
        field_level = 'level_3'
        if field_type is not None:
            assert field_type in list(EXPT_FIELDS.keys()) and field_level in list(EXPT_FIELDS[field_type][npz_type].keys())
        make_field_plots(field_type, field_level, npz_type, simsetup)
