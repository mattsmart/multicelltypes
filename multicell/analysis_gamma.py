import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from multicell.multicell_class import Multicell
from multicell.multicell_replot import replot_scatter_dots
from multicell.graph_helper import state_load
from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_linalg import sorted_eig
from utils.file_io import RUNS_FOLDER, INPUT_FOLDER


def scan_plaquette_gamma_dynamics(J, W, state, coordnum=8, verbose=False, use_01=False):
    critgamma = None

    def get_state_send(state_send):
        if use_01:
            state_send = (state_send + np.ones_like(state_send)) / 2.0
        return state_send

    for gamma in np.linspace(0.001, 0.8, 10000):
        Js_internal = np.dot(J, state)

        # conditional 01 state send
        state_send = get_state_send(state)
        h_field_nbr = gamma * coordnum * np.dot(W, state_send)

        updated_state = np.sign(Js_internal + h_field_nbr)
        if np.array_equal(updated_state, state):
            if verbose:
                print(gamma, True)
        else:
            if critgamma is None:
                critgamma = gamma
            if verbose:
                print(gamma, False)
    return critgamma


def descend_to_fp(multicell):
    """
    Helper function for gamma scan functions
    - scan_gamma_bifurcation_candidates()
    - manyruns_gamma_bifurcation_candidates()
    """
    multicell.dynamics_full(
        flag_visualize=False, flag_datastore=False, flag_savestates=False,
        end_at_fp=True, verbose=False)
    current_step = multicell.current_step
    fp = multicell.graph_state_arr[:, current_step]
    return fp


def check_still_fp(test_fp, J_multicell):
    """
    Helper function for gamma scan functions
    - scan_gamma_bifurcation_candidates()
    - manyruns_gamma_bifurcation_candidates()
    """
    A = test_fp
    B = np.sign(np.dot(J_multicell, test_fp))
    return np.array_equal(A, B)


def scan_gamma_bifurcation_candidates(
        multicell_kwargs, simsetup_base, anchored=True, verbose=True, dg=1e-1, gmin=0.0, gmax=20.0,
        save_states_all=False, save_states_shift=True):
    """
    For fixed initial condition and multicell parameters, slowly vary gamma.
    Find {gamma*}, the observed points where the fixed point has changed.
    The fixed point shifting is not a symptom of a bifurcation, for example consider pitchfork
     bifurcation, the two fixed points continue to shift after the (singular) bifurcation.
    Note:
        Consider continuous dynamical system.
        When a static FP suddenly starts shifting in almost continuous fashion, that's a signature
         of a bifurcation (e.g. transcritical or pitchfork).
        What, if any, is the discrete (discrete time AND discrete state) analog of this?
    Args:
        multicell_kwargs: kwargs to form Multicell which is recreated for each gamma during the scan
        simsetup_base: simsetup dict template storing J, W, singlecell parameters
        anchored: if True, use a fixed initial condition for each gradient descent;
                  else will use the previous fixed point as the initial condition
        save_states_all: saved the fixed point for each gamma as 'X_all_g%.5f.npz'
        save_states_shift: saved the fixed point for each 'fp shift' gamma as 'X_shift_g%.5f.npz'
    Returns:
         list: the sequence of points {gamma*_n} where bifurcations have occurred
    Notes:
        - naively can do full gradient descent to reach each fixed point
        - once a fixed point is found, there is a faster way to check that it remains a fixed point:
          simply check that s* = sgn(J_multicell s*)   -- this is useful in "not anchored" case
        - if this vector condition holds then the FP is unchanged; when it breaks there is a
          bifurcation point (which is recorded) and the new FP should be found via descent

    SPEEDUP CHANGES:
    - only multicell instantiation before the loop (not during)
    - REQUIRES no varying of simsetup (J, W, N) in the loop - will need to refactor if this changes
    """

    # build gamma_space
    gamma_space = np.arange(gmin, gmax, dg)
    num_gamma = len(gamma_space)
    bifurcation_candidate_sequence = []

    if save_states_all:
        print('Warning: save_states_all is inefficient, use inferred gamma step size to recreate '
              'distribution of fixed points (i.e. fill in gaps between FP shifts)')

    # construct multicell_base from kwargs
    save_init = True
    multicell_base = Multicell(simsetup_base, verbose=False, **multicell_kwargs)
    if save_init:
        init_cond = multicell_base.graph_state_arr[:, 0]
        fpath = multicell_base.io_dict['statesdir'] + os.sep + 'X_init.npz'
        np.savez_compressed(fpath, init_cond, fmt='%d')

    # prep: perform gradient descent on the init cond to get our (potentially anchored) fixed point
    init_fp = descend_to_fp(multicell_base)
    prev_fp = np.copy(init_fp)  # used for iterative comparisons

    # speedup:
    multicell_local = multicell_base  # change attributes on the fly

    for i, gamma in enumerate(gamma_space):
        if i % 200 == 0:
            print("Checking %d/%d (gamma=%.4f)" % (i, num_gamma, gamma))
        multicell_kwargs_local = multicell_kwargs.copy()
        multicell_kwargs_local['gamma'] = gamma

        # 1) Re-build Multicell for gamma
        J_multicell = multicell_base.build_J_multicell(gamma=gamma, plot=False)
        multicell_local.gamma = gamma
        multicell_local.matrix_J_multicell = J_multicell

        # 2) gradient descent to fixed point
        if anchored:
            multicell_local.simulation_reset(provided_init_state=init_fp)
            step_fp = descend_to_fp(multicell_local)
            fp_unchanged = np.array_equal(step_fp, prev_fp)
            prev_fp = step_fp
        else:
            fp_unchanged = check_still_fp(prev_fp, J_multicell)
            if not fp_unchanged:
                multicell_local.simulation_reset(provided_init_state=prev_fp)
                prev_fp = descend_to_fp(multicell_local)

        if save_states_all:
            glabel = 'all_g%.5f' % gamma
            fpath = multicell_local.io_dict['statesdir'] + os.sep + 'X_%s.npz' % glabel
            np.savez_compressed(fpath, prev_fp, fmt='%d')

    # 3) report a bifurcation whenever the fixed point moves
        if not fp_unchanged:
            if verbose:
                print('fixed point shift at gamma=%.5f' % gamma)
            if save_states_shift:
                glabel = 'fpshift_g%.5f' % gamma
                fpath = multicell_local.io_dict['statesdir'] + os.sep + 'X_%s.npz' % glabel
                np.savez_compressed(fpath, prev_fp, fmt='%d')
            bifurcation_candidate_sequence.append(gamma)

    # save primary data from gammascan loop
    fpath_x = multicell_local.io_dict['datadir'] + os.sep + 'bifurcation_candidates.txt'
    fpath_gamma = multicell_local.io_dict['datadir'] + os.sep + 'gamma_space.txt'
    np.savetxt(fpath_x, bifurcation_candidate_sequence, '%.5f')
    np.savetxt(fpath_gamma, gamma_space, '%.5f')

    return bifurcation_candidate_sequence, gamma_space, multicell_base


def plot_bifurcation_candidates_curve(bifurcation_candidates, gamma_space, outdir, show=False):
    # plot type A
    x = np.arange(len(bifurcation_candidates))
    y = np.array(bifurcation_candidates)
    plt.scatter(x, y, marker='x')
    plt.xlabel(r'$n$')
    plt.ylabel(r'${\gamma}^*_n$')
    plt.savefig(outdir + os.sep + 'bifurc_A.jpg')
    if show:
        plt.show()

    # plot type B
    x = np.arange(len(bifurcation_candidates))
    y_construct = np.zeros(len(gamma_space))
    k = 0
    g0 = 0.0
    total_bifurcation_candidates = len(bifurcation_candidates)
    for i, gamma in enumerate(gamma_space):
        if bifurcation_candidates[k] > gamma:
            y_construct[i] = g0
        else:
            g0 = bifurcation_candidates[k]
            if k < total_bifurcation_candidates - 1:
                k += 1
            y_construct[i] = g0
    plt.plot(gamma_space, y_construct, '--', c='k')
    plt.plot(gamma_space, gamma_space, '-.', c='k', alpha=0.5)
    plt.scatter(bifurcation_candidates, bifurcation_candidates, marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'${\gamma}^*_n$ Transitions (n=%d)' % len(bifurcation_candidates))
    plt.savefig(outdir + os.sep + 'bifurc_B.jpg')
    if show:
        plt.show()


if __name__ == '__main__':
    force_symmetry_W = True
    destabilize_celltypes_gamma = True
    flag_plot_multicell_evals = False
    flag_bifurcation_sequence = True

    main_seed = 0 #np.random.randint(1e6)
    curated = True
    random_mem = False        # TODO incorporate seed in random XI in simsetup/curated
    random_W = False          # TODO incorporate seed in random W in simsetup/curated

    #W_override_path = None
    #W_id = 'W_9_W15maze'
    W_id = 'W_9_maze'
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_%s.txt' % W_id
    simsetup = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    if W_override_path is not None:
        print('Note: in main, overriding W from file...')
        explicit_W = np.loadtxt(W_override_path, delimiter=',')
        simsetup['FIELD_SEND'] = explicit_W
    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup['N'])
    print("\tsimsetup['P'],", simsetup['P'])

    if force_symmetry_W:
        W = simsetup['FIELD_SEND']
        # V1: take simple sym
        #simsetup['FIELD_SEND'] = (W + W.T)/2
        # V2: take upper triangular part
        Wdiag = np.diag(np.diag(W))
        Wut = np.triu(W, 1)
        simsetup['FIELD_SEND'] = Wut + Wut.T + Wdiag
        # V3: take lower triangular part
        #Wdiag = np.diag(np.diag(W))1
        #Wut = np.tril(W, -1)
        #simsetup['FIELD_SEND'] = Wut + Wut.T + Wdiag
        # Save symmetrized W
        np.savetxt('Wsym.txt', simsetup['FIELD_SEND'], '%.4f', delimiter=',')
    print(simsetup['FIELD_SEND'])

    if destabilize_celltypes_gamma:
        coordnum = 8  # num neighbours which signals are received from
        W = simsetup['FIELD_SEND']
        J = simsetup['J']
        celltypes = [simsetup['XI'][:, a] for a in range(simsetup['P'])]
        print('Scanning for monotype destabilizing gamma (for coordination number %d)' % coordnum)
        for idx, celltype in enumerate(celltypes):
            critgamma = scan_plaquette_gamma_dynamics(J, W, celltype, coordnum=coordnum, verbose=False)
            print(idx, simsetup['CELLTYPE_LABELS'][idx], critgamma)
        print('Scanning for inverted monotype destabilizing gamma (for coordination number %d)' % coordnum)
        for idx, celltype in enumerate(celltypes):
            inverted_celltype = -1 * celltype
            critgamma = scan_plaquette_gamma_dynamics(J, W, inverted_celltype, coordnum=coordnum, verbose=False)
            print(idx, 'flip of celltype:', simsetup['CELLTYPE_LABELS'][idx], critgamma)
        print('Scanning for spurious monotype destabilizing gamma (for coordination number %d)' % coordnum)
        Splus = np.sign(celltypes[0] + celltypes[1] + celltypes[2])
        Sminus = -1 * Splus
        print('S+:', Splus)
        critgamma = scan_plaquette_gamma_dynamics(J, W, Splus, coordnum=coordnum, verbose=False)
        print(0, 'spurious S+', critgamma)
        critgamma = scan_plaquette_gamma_dynamics(J, W, Sminus, coordnum=coordnum, verbose=False)
        print(1, 'spurious S-', critgamma)

    if flag_plot_multicell_evals:
        # TODO implement or take from ipynb
        J_multicell = 1
        evals, evecs = sorted_eig(J_multicell)

    if flag_bifurcation_sequence:

        # 1) choose BASE simsetup (core singlecell params J, W)
        simsetup_base = simsetup

        # 2) choose BASE Multicell class parameters
        sidelength = 10
        num_cells = sidelength ** 2
        autocrine = False
        graph_style = 'lattice_square'
        assert graph_style == 'lattice_square' and not autocrine
        search_radius = 1
        init_style = 'dual'
        graph_kwargs = {'search_radius': search_radius,
                        'periodic': True,
                        'initialization_style': init_style}
        load_manual_init = False
        init_state_path = None
        if load_manual_init:
            init_style = 'manual'
            print('Note: in main, loading init graph state from file...')
            init_state_path = INPUT_FOLDER + os.sep + 'manual_graphstate' + os.sep + 'X_8.txt'

        # specify gamma scan parameters
        dgS = '5e-3' # '5e-4'
        gminS = '0'
        gmaxS = '0.15' # '1.0' or '4.0'
        dg, gmin, gmax = float(dgS), float(gminS), float(gmaxS)
        anchored = False
        save_all = False
        save_shifts = True
        plot_all_tissues = True

        #seed = 0
        for seed in range(1):
            print('WORKING ON seed:', seed)

            # create run basedir label based on specified parameters
            run_subdir = 'gscan_anchor%d_gLow%s_gHigh%s_gStep%s_%s_R%d_init_%s_s%d_M%d' % \
                         (int(anchored), gminS, gmaxS, dgS, W_id, search_radius, init_style, seed, num_cells)
            if save_all or save_shifts:
                run_basedir_path = RUNS_FOLDER + os.sep + 'multicell_manyruns'
            else:
                run_basedir_path = RUNS_FOLDER + os.sep + 'explore' + os.sep + 'bifurcation'
            assert not os.path.exists(run_basedir_path + os.sep + run_subdir)

            multicell_kwargs_base = {
                'seed': seed,
                'run_basedir': run_basedir_path,
                'run_subdir': run_subdir,
                'beta': np.Inf,
                'total_steps': 500,
                'num_cells': num_cells,
                'flag_blockparallel': False,
                'graph_style': graph_style,
                'graph_kwargs': graph_kwargs,
                'autocrine': autocrine,
                'gamma': 0.0,
                'exosome_string': 'no_exo_field',
                'exosome_remove_ratio': 0.0,
                'kappa': 0.0,
                'field_applied': None,
                'flag_housekeeping': False,
                'flag_state_int': True,
                'plot_period': 1,
                'init_state_path': init_state_path,
            }

            bifurcation_candidates, gamma_space, multicell = scan_gamma_bifurcation_candidates(
                multicell_kwargs_base, simsetup_base, anchored=anchored, verbose=True,
                dg=dg, gmin=gmin, gmax=gmax, save_states_all=save_all, save_states_shift=save_shifts)

            outdir = multicell.io_dict['datadir']
            plot_bifurcation_candidates_curve(bifurcation_candidates, gamma_space, outdir, show=False)

            # plot each bifurcation candidate as fancy tissue
            if plot_all_tissues:
                print(multicell.io_dict.keys())
                plotlattice_dir = multicell.io_dict['plotlatticedir']
                states_dir = multicell.io_dict['statesdir']

                # plot initial state first
                init_state = state_load(
                    states_dir + os.sep + 'X_init' + '.npz',
                    cells_as_cols=True,
                    num_genes=multicell.num_genes,
                    num_cells=num_cells,
                    txt=False)
                replot_scatter_dots(
                    init_state,
                    sidelength,
                    plotlattice_dir + os.sep + 'X_init',
                    fmod='', state_int=False, cmap=None, title=None,
                    ext=['.jpg', '.svg'], rasterized=True)

                for idx, gval in enumerate(bifurcation_candidates):
                    print('Plotting fpshift tissue #%d, gamma=%.5f...' % (idx, gval))

                    fname = 'X_fpshift_g%.5f' % gval
                    lattice_state_path = states_dir + os.sep + fname + '.npz'
                    lattice_state = state_load(
                        lattice_state_path,
                        cells_as_cols=True,
                        num_genes=multicell.num_genes,
                        num_cells=num_cells,
                        txt=False)
                    print(lattice_state.shape)
                    outpath = plotlattice_dir + os.sep + fname

                    replot_scatter_dots(
                        lattice_state,
                        sidelength,
                        outpath,
                        fmod='', state_int=False, cmap=None, title=None,
                        ext=['.jpg', '.svg'], rasterized=True)
