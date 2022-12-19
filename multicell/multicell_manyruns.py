import natsort
import numpy as np
import os
import pickle
import shutil

from multicell.graph_helper import state_load
from multicell.multicell_constants import DYNAMICS_FIXED_UPDATE_ORDER
from multicell.multicell_metrics import calc_graph_energy
from multicell.multicell_class import Multicell
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import RUNS_FOLDER, INPUT_FOLDER


def aggregate_manyruns(runs_basedir, agg_subdir='aggregate',
                       agg_states=True,
                       agg_energy=True,
                       agg_plot=True,
                       only_last=True):
    agg_dir = runs_basedir + os.sep + agg_subdir
    if not os.path.exists(agg_dir):
        os.mkdir(agg_dir)

    # Step 0) get all the run directories
    fpaths = [runs_basedir + os.sep + a for a in os.listdir(runs_basedir)]
    run_dirs = [a for a in fpaths
                if os.path.isdir(a) and os.path.basename(a) not in ('aggregate', 'dimreduce')]
    run_dirs = natsort.natsorted(run_dirs)

    # Step 1) get info from the first run directory (require at least one)
    ppath = runs_basedir + os.sep + 'multicell_template.pkl'
    with open(ppath, 'rb') as pickle_file:
        multicell_template = pickle.load(pickle_file)  # Unpickling the object
    num_genes = multicell_template.num_genes
    num_cells = multicell_template.num_cells
    total_spins = num_genes * num_cells

    # Step 2) aggregate file containing all the fixed points
    # X_aggregate.npz -- 2D, total_spins x num_runs, full state of each FP
    # X_energies.npz  -- 2D,           5 x num_runs, energy tuple of each FP
    # if fastplot, produce plot of each state
    num_runs = len(run_dirs)
    fixedpoints_ensemble = np.zeros((total_spins, num_runs), dtype=int)
    energies = np.zeros((5, num_runs), dtype=float)

    if only_last:
        X_labels = ['last']
    else:
        X_labels = ['%d' % idx for idx in range(multicell_template.total_steps)]

    for label in X_labels:
        for i, run_dir in enumerate(run_dirs):
            if i % 200 == 0:
                print(i, run_dir[-40:])
            fpath = run_dir + os.sep + 'states' + os.sep + 'X_%s.npz' % label
            X = state_load(fpath, cells_as_cols=False, num_genes=num_genes,
                           num_cells=num_cells, txt=False)
            step_hack = 0  # care: this will break if class has time-varying applied field
            multicell_template.graph_state_arr[:, step_hack] = X[:]
            assert np.array_equal(multicell_template.field_applied,
                                  np.zeros((total_spins, multicell_template.total_steps)))

            # 2.1) get final state
            if agg_states:
                fixedpoints_ensemble[:, i] = X

            # 2.2) get state energy for bokeh
            if agg_energy:
                state_energy = calc_graph_energy(multicell_template, step_hack, norm=True)
                energies[:, i] = state_energy

            # 2.3) get state image for bokeh
            if agg_plot:
                fpaths = [runs_basedir + os.sep + 'aggregate' + os.sep + a for a in
                          ['agg_%s_%d_compOverlap.png' % (label, i),
                           'agg_%s_%d_compProj.png' % (label, i),
                           'agg_%s_%d_ref0_overlap.png' % (label, i),
                           'agg_%s_%d_scatter_dots.png' % (label, i)]
                           ]
                multicell_template.step_datadict_update_global(step_hack, fill_to_end=False)
                multicell_template.step_state_visualize(step_hack, fpaths=fpaths)  # visualize

        if agg_states:
            np.savez_compressed(agg_dir + os.sep + 'X_aggregate_%s' % label, fixedpoints_ensemble)

        if agg_energy:
            np.savez_compressed(agg_dir + os.sep + 'X_energy_%s' % label, energies)


def gen_random_W(simsetup, seed):
    N = simsetup['N']
    np.random.seed(seed)
    W_0 = np.random.rand(N, N) * 2 - 1  # scale to Uniform [-1, 1]
    W_lower = np.tril(W_0, k=-1)
    W_diag = np.diag(np.diag(W_0))
    W_sym = (W_lower + W_lower.T + W_diag)
    return W_sym


if __name__ == '__main__':

    # Approach A (switch = True): fix W, vary the initial condition
    # Approach B (switch = False): fix the initial condition, vary W
    switch_vary_initcond = True
    load_manual_init = False
    if switch_vary_initcond:
        flag_fixed_initcond = False
        flag_fixed_W = True
    else:
        flag_fixed_initcond = True
        flag_fixed_W = False

    generate_data = True  # True
    aggregate_data = True
    agg_states = True  # setting used with aggregate_data
    agg_energy = True  # setting used with aggregate_data
    agg_plot = False   # setting used with aggregate_data

    # key runtime settings
    num_cells = 10 ** 2  # global GRIDSIZE
    total_steps = 500     # global NUM_LATTICE_STEPS
    num_runs = int(1e4)  # int(1e4)

    # place to generate many runs
    gamma_list = [0.0, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                  0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 20.0]

    beta_main = np.Inf #np.Inf   # 2000.0
    if beta_main == np.Inf:
        print('Note (beta is np.Inf): using deterministic settings')
        agg_only_last = True
        end_at_fp = True
        beta_str = ''
    else:
        assert beta_main == 2000.0
        agg_only_last = True
        end_at_fp = True
        beta_str = '2000.0'
        #agg_only_last = False
        #end_at_fp = False
        #beta_str = 'beta%.2f_' % beta_main

    for gamma_main in gamma_list:
        multirun_name = 'W14_gamma%.2f_10k_periodic_R1_p3_M100' % (gamma_main)
        #multirun_name = '%sW1pattern_gamma%.2f_10k_p3_M100' % (beta_str, gamma_main)
        #multirun_name = '%sWvary_s0randomInit_gamma%.2f_10k_periodic_fixedorderV3_p3_M100' % (beta_str, gamma_main)
        #multirun_name = '%sWvary_dualInit_gamma%.2f_10k_periodic_fixedorderV3_p3_M100' % (beta_str, gamma_main)
        multirun_path = RUNS_FOLDER + os.sep + 'multicell_manyruns' + os.sep + multirun_name

        if generate_data:
            assert not os.path.exists(multirun_path)

            # 1) create simsetup
            simsetup_seed = 0
            curated = True
            random_mem = False        # TODO incorporate seed in random XI
            random_W = False          # TODO incorporate seed in random W
            #W_override_path = None
            #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_maze.txt'
            #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_random1.txt'
            #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W1pattern.txt'
            W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W14blob.txt'
            #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W12hetero.txt'
            #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W13pattern.txt'
            #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W15maze.txt'

            simsetup_main = singlecell_simsetup(
                unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)

            if W_override_path is not None:
                print('Note: in main, overriding W from file...')
                explicit_W = np.loadtxt(W_override_path, delimiter=',')
                simsetup_main['FIELD_SEND'] = explicit_W

            print("simsetup checks:")
            print("\tsimsetup['N'],", simsetup_main['N'])
            print("\tsimsetup['P'],", simsetup_main['P'])

            # setup 2.1) multicell sim core parameters
            plot_period = 1
            flag_state_int = True
            flag_blockparallel = False
            if aggregate_data:
                assert not flag_blockparallel
            gamma = gamma_main         # i.e. field_signal_strength
            kappa = 0.0               # i.e. field_applied_strength

            # setup 2.2) graph options
            autocrine = False
            graph_style = 'lattice_square'
            graph_kwargs = {'search_radius': 1,
                            'periodic': True,
                            'initialization_style': 'random'}

            # setup 2.3) signalling field (exosomes + cell-cell signalling via W matrix)
            # Note: consider rescale gamma as gamma / num_cells * num_plaquette
            # global gamma acts as field_strength_signal, it tunes exosomes AND sent field
            exosome_string = "no_exo_field"  # on/off/all/no_exo_field; 'off' = send info only 'off' genes
            exosome_remove_ratio = 0.0       # amount of exo field idx to randomly prune from each cell

            # setup 2.4) applied/manual field (part 1)
            # size [N x steps] or size [NM x steps] or None
            # field_applied = construct_app_field_from_genes(
            #    IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)
            field_applied = None

            # setup 2.5) applied/manual field (part 2) add housekeeping field with strength kappa
            flag_housekeeping = False
            field_housekeeping_strength = 0.0  # aka Kappa
            assert not flag_housekeeping
            if flag_housekeeping:
                assert field_housekeeping_strength > 0
                # housekeeping auto (via model extension)
                field_housekeeping = np.zeros(simsetup_main['N'])
                if simsetup_main['K'] > 0:
                    field_housekeeping[-simsetup_main['K']:] = 1.0
                    print(field_applied)
                else:
                    print('Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories')
                    print('Note gene 4 (off), 5 (on) are HK in C1 memories')
                    field_housekeeping[4] = 1.0
                    field_housekeeping[5] = 1.0
                if field_applied is not None:
                    field_applied += field_housekeeping_strength * field_housekeeping
                else:
                    field_applied = field_housekeeping_strength * field_housekeeping
            else:
                field_housekeeping = None

            # setup 2.6) optionally load an initial state for the lattice
            load_manual_init = False
            init_state_path = None
            if load_manual_init:
                init_state_path = INPUT_FOLDER + os.sep + 'manual_graphstate' + os.sep + 'X_8.txt'
                print('Note: in main, loading init graph state from file...')

            # 3) prep args for Multicell class instantiation
            multicell_kwargs_base = {
                'run_basedir': multirun_path,
                'beta': beta_main,
                'total_steps': total_steps,
                'num_cells': num_cells,
                'flag_blockparallel': flag_blockparallel,
                'graph_style': graph_style,
                'graph_kwargs': graph_kwargs,
                'autocrine': autocrine,
                'gamma': gamma,
                'exosome_string': exosome_string,
                'exosome_remove_ratio': exosome_remove_ratio,
                'kappa': kappa,
                'field_applied': field_applied,
                'flag_housekeeping': flag_housekeeping,
                'flag_state_int': flag_state_int,
                'plot_period': plot_period,
                'init_state_path': init_state_path,
            }

            ensemble = 1  # currently not used
            run_dirs = [''] * num_runs

            # note we pickle the first runs instance for later loading
            for i in range(num_runs):
                if i % 200 == 0:
                    print("On run %d (%d total)" % (i, num_runs))
                multicell_kwargs_local = multicell_kwargs_base.copy()
                simsetup_local = simsetup_main.copy()

                # 1) modify multicell kwargs for each run
                seed = i                                             # set local seed
                multicell_kwargs_local['run_subdir'] = 's%d' % seed  # set run label
                if switch_vary_initcond:
                    # Note multicell seed controls:
                    # - init cond (if 'random' initialization style is used)
                    # - dynamics if the following global is False: DYNAMICS_FIXED_UPDATE_ORDER
                    multicell_kwargs_local['seed'] = seed
                else:
                    # Note multicell seed controls:
                    # - init cond (if 'random' initialization style is used)
                    # - dynamics if the following global is False: DYNAMICS_FIXED_UPDATE_ORDER
                    if DYNAMICS_FIXED_UPDATE_ORDER:
                        multicell_kwargs_local['seed'] = 0  # so that 'random init' will use seed 0
                    else:
                        multicell_kwargs_local['seed'] = seed
                        assert init_state_path is not None  # if its None, then seed will affect IC
                    simsetup_local['FIELD_SEND'] = gen_random_W(simsetup_local, seed)

                # 2) instantiate
                multicell = Multicell(simsetup_local, verbose=False, **multicell_kwargs_local)
                run_dirs[i] = multicell.io_dict['basedir']

                # 2.1) save full state to file for the first run (place in parent dir)
                if i == 0:
                    if not os.path.exists(multirun_path):
                        os.mkdir(multirun_path)
                    ppath = multirun_path + os.sep + 'multicell_template.pkl'
                    with open(ppath, 'wb') as fp:
                        pickle.dump(multicell, fp)

                # 3) run sim
                multicell.simulation_fast(no_datatdict=True, no_visualize=True, end_at_fp=end_at_fp,
                                          verbose=False)

        # aggregate data from multiple runs
        if aggregate_data:
            print('Aggregating data in %s' % multirun_path)
            aggregate_manyruns(
                multirun_path, agg_states=agg_states, agg_energy=agg_energy, agg_plot=agg_plot,
                only_last=agg_only_last)
