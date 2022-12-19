import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcdefaults()

from multicell.multicell_class import Multicell
from multicell.multicell_constants import DYNAMICS_FIXED_UPDATE_ORDER
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import INPUT_FOLDER


# TODO convert to main attribute of class, passable (defaults to False?)
print('DYNAMICS_FIXED_UPDATE_ORDER:', DYNAMICS_FIXED_UPDATE_ORDER)


if __name__ == '__main__':

    # 1) create simsetup
    main_seed = 0  #np.random.randint(1e6)
    curated = True
    random_mem = False        # TODO incorporate seed in random XI in simsetup/curated
    random_W = True          # TODO incorporate seed in random W in simsetup/curated

    #W_override_path = None
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_2018mazeUpTri.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_manual_ABv2.txt'
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W15maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W7maze.txt'

    simsetup_main = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    if W_override_path is not None:
        print('Note: in main, overriding W from file...')
        explicit_W = np.loadtxt(W_override_path, delimiter=',')
        simsetup_main['FIELD_SEND'] = explicit_W
    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup_main['N'])
    print("\tsimsetup['P'],", simsetup_main['P'])
    print(simsetup_main['XI'])

    # setup 2.1) multicell sim core parameters
    search_radius = 1
    num_cells = 20**2           # global GRIDSIZE
    total_steps = 41            # global NUM_LATTICE_STEPS
    plot_period = 1
    flag_state_int = False
    flag_blockparallel = False
    beta = np.Inf  # 2000.0 use np.Inf instead of fixed 1e3, can cause rare bugs otherwise
    gamma = 0.5                # i.e. field_signal_strength
    kappa = 0.0                # i.e. field_applied_strength

    # setup 2.2) graph options
    autocrine = False
    graph_style = 'lattice_square'
    graph_kwargs = {'search_radius': search_radius,
                    'periodic': True,
                    'initialization_style': 'dual'}

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
        init_state_path = INPUT_FOLDER + os.sep + 'manual_graphstate' + os.sep + 'X_7.txt'
        print('Note: in main, loading init graph state from file...')

    # 3) prep args for Multicell class instantiation
    multicell_kwargs = {
        'beta': beta,
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
        'seed': main_seed,
        'run_subdir': None  #'s%d' % main_seed
    }

    # 3) instantiate
    multicell = Multicell(simsetup_main, verbose=True, **multicell_kwargs)

    # 4) run sim
    multicell.simulation_standard()
    #multicell.simulation_fast()

    # looped version of steps 3) and 4):
    """
    for gstep in [0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 0.75, 1.0, 4.0, 20.0]:
        multicell_kwargs_step = dict(multicell_kwargs, gamma=gstep)

        multicell = Multicell(simsetup_main, verbose=True, **multicell_kwargs_step)
        multicell.simulation_fast()
    """
