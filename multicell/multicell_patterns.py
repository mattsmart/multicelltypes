import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell.multicell_lattice import build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells, \
    write_grid_state_int
from multicell.multicell_visualize_old import lattice_uniplotter, reference_overlap_plotter, lattice_projection_composite
from singlecell.singlecell_class import Cell
from utils.file_io import run_subdir_setup, runinfo_append, write_general_arr, read_general_arr
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from utils.make_video import make_video_ffmpeg
from functools import reduce


def build_lattice_memories(simsetup, M):
    assert M == 144
    sqrtM = 12
    num_y = sqrtM
    num_x = sqrtM
    assert simsetup['P'] >= 2

    def conv_grid_to_vector(grid):
        def label_to_celltype_vec(label):
            cellytpe_idx = simsetup['CELLTYPE_ID'][label]
            return simsetup['XI'][:, cellytpe_idx]
        lattice_vec = np.zeros(M * simsetup['N'])
        for row in range(num_y):
            for col in range(num_x):
                posn = sqrtM * row + col
                label = grid[row][col]
                celltype_vec = label_to_celltype_vec(label)
                start_spin = posn * simsetup['N']
                end_spin = (posn+1) * simsetup['N']
                lattice_vec[start_spin:end_spin] = celltype_vec
        return lattice_vec

    mem1 = [['mem_A' for _ in range(sqrtM)] for _ in range(sqrtM)]
    mem2 = [['mem_A' for _ in range(sqrtM)] for _ in range(sqrtM)]
    mem3 = [['mem_A' for _ in range(sqrtM)] for _ in range(sqrtM)]
    # build mem 1 -- number 1 on 12x12 grid
    for row in range(num_y):
        for col in range(num_x):
            if row in (1,2) and col in range(2,7):
                mem1[row][col] = 'mem_B'
            if row in range(3,9) and col in (5,6):
                mem1[row][col] = 'mem_B'
            if row in (9,10) and col in range(2,10):
                mem1[row][col] = 'mem_B'
    mem1 = conv_grid_to_vector(mem1)
    # build mem 2 -- number 2 on 12x12 grid
    for row in range(num_y):
        for col in range(num_x):
            if row in (1,2,5,6,9,10) and col in range(2,10):
                mem2[row][col] = 'mem_B'
            if row in (3,4) and col in (8,9):
                mem2[row][col] = 'mem_B'
            if row in (7,8) and col in (2,3):
                mem2[row][col] = 'mem_B'
    mem2 = conv_grid_to_vector(mem2)
    # build mem 3 -- number 3 on 12x12 grid
    for row in range(num_y):
        for col in range(num_x):
            if row in (1,2,5,6,9,10) and col in range(2,10):
                mem3[row][col] = 'mem_B'
            if row in (3,4,7,8) and col in (8,9):
                mem3[row][col] = 'mem_B'
    mem3 = conv_grid_to_vector(mem3)
    # slot into array
    lattice_memories = np.zeros((M * simsetup['N'], 3))
    lattice_memories[:, 0] = mem1
    lattice_memories[:, 1] = mem2
    lattice_memories[:, 2] = mem3
    return lattice_memories


def hopfield_on_lattice_memories(simsetup, M, lattice_memories):
    xi = lattice_memories
    corr_matrix = np.dot(xi.T, xi) / float(xi.shape[0])
    print(xi[6 * (0):6 * (1), :])
    print('and')
    print(xi[6*(17):6*(18),:])
    corr_inv = np.linalg.inv(corr_matrix)
    intxn_matrix = reduce(np.dot, [xi, corr_inv, xi.T]) / float(xi.shape[0])
    intxn_matrix = intxn_matrix - np.kron(np.eye(M), np.ones((simsetup['N'], simsetup['N'])))
    return intxn_matrix


def sim_lattice_as_cell(simsetup, num_steps, beta, app_field, app_field_strength):
    # build multicell intxn matrix
    gamma = 1.0
    M = 144
    sqrtM = 12
    total_spins = M * simsetup['N']
    lattice_memories = build_lattice_memories(simsetup, M)
    lattice_memories_list = ['%d' % idx for idx, elem in enumerate(lattice_memories)]
    lattice_gene_list =['site_%d' % idx for idx in range(total_spins)]
    lattice_intxn_matrix = np.kron(np.eye(M), simsetup['J']) + \
                           gamma * (hopfield_on_lattice_memories(simsetup, M, lattice_memories))
    # build lattice applied field (extended)
    app_field_on_lattice = np.array([app_field for _ in range(M)]).reshape(total_spins)
    # setup IO
    io_dict = run_subdir_setup()
    # initialize
    init_cond = np.random.randint(0,2,total_spins)*2 - 1  # TODO alternatives
    lattice_as_cell = Cell(init_cond, 'lattice_as_cell', lattice_memories_list, lattice_gene_list, state_array=None, steps=None)
    lattice = build_lattice_main(sqrtM, None, "explicit", simsetup,
                                 state=lattice_as_cell.get_current_state())  # TODO hacky
    # simulate for t steps
    for turn in range(num_steps):
        print('step', turn)
        lattice_as_cell.update_state(lattice_intxn_matrix, beta=beta, field_applied=app_field_on_lattice,
                                     field_applied_strength=app_field_strength, async_batch=True)
        # fill in lattice info from update  # TODO convert to spatial cell method from explicit lattice vec?
        lattice_vec = lattice_as_cell.get_current_state()
        for i in range(sqrtM):
            for j in range(sqrtM):
                cell = lattice[i][j]
                posn = sqrtM * i + j
                start_spin = posn * simsetup['N']
                end_spin = (posn + 1) * simsetup['N']
                new_cellstate = lattice_vec[start_spin:end_spin].T
                state_array_ext = np.zeros((simsetup['N'], np.shape(cell.state_array)[1] + 1))
                state_array_ext[:, :-1] = cell.state_array  # TODO: make sure don't need array copy
                state_array_ext[:, -1] = new_cellstate
                cell.steps += 1
                cell.state = new_cellstate            # TODO: make sure don't need array copy
                cell.state_array = state_array_ext
        lattice_projection_composite(lattice, turn, sqrtM, io_dict['latticedir'], simsetup, state_int=True, cmap_vary=True)
        reference_overlap_plotter(lattice, turn, sqrtM, io_dict['latticedir'], simsetup, state_int=True)
        #if flag_uniplots:
        #    for mem_idx in memory_idx_list:
        #        lattice_uniplotter(lattice, turn, n, io_dict['latticedir'], mem_idx, simsetup)

    return lattice, io_dict


if __name__ == '__main__':
    num_steps = 20
    beta = 10.0
    make_video = True

    # specify single cell model
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, curated=True, housekeeping=0)

    # TODO housekeeping field
    app_field = None
    # housekeeping genes block
    KAPPA = 1.0
    if KAPPA > 0:
        # housekeeping auto (via model extension)
        app_field = np.zeros(simsetup['N'])
        if simsetup['K'] > 0:
            app_field[-simsetup['K']:] = 1.0
            print(app_field)
        else:
            print('Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories')
            print('Note gene 4 (off), 5 (on) are HK in C1 memories')
            app_field[4] = 1.0
            app_field[5] = 1.0

    lattice, io_dict = sim_lattice_as_cell(simsetup, num_steps, beta, app_field, KAPPA)
    print('Done')


    """
    plot_period = 1
    state_int = True
    beta = BETA  # 2.0
    mc_sim_wrapper(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
           field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
           app_field_strength=KAPPA, beta=beta, plot_period=plot_period, state_int=state_int,
           meanfield=meanfield)
    """

    """
    for beta in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 4.0, 5.0, 10.0, 100.0]:
        mc_sim_wrapper(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
               field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
               app_field_strength=app_field_strength, beta=beta, plot_period=plot_period, state_int=state_int, meanfield=meanfield)
    """
    if make_video:
        print('Making video...')
        # fhead = "composite_lattice_step"
        fhead = "composite_lattice_step"
        fps = 2
        outpath = io_dict['basedir'] + os.sep + 'movie2_expC1_fsHigh_beta1.mp4'
        make_video_ffmpeg(io_dict['latticedir'], outpath, fps=1, fhead=fhead, ftype=".png", nmax=num_steps)
