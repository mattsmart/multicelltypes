import numpy as np
import os
import random
import shutil
import matplotlib.pyplot as plt
plt.rcdefaults()

from multicell.graph_adjacency import \
    lattice_square_int_to_loc, adjacency_lattice_square, adjacency_general, general_exosome_field, \
    general_paracrine_field
from multicell.graph_helper import state_load
from multicell.multicell_constants import \
    VALID_BUILDSTRINGS, VALID_EXOSOME_STRINGS, EXOSTRING, EXOSOME_REMOVE_RATIO, \
    BLOCK_UPDATE_LATTICE, AUTOCRINE, SEND_01, DYNAMICS_FIXED_UPDATE_ORDER
from multicell.multicell_lattice import \
    build_lattice_main, write_grid_state_int_alt
from multicell.multicell_metrics import \
    calc_compression_ratio, calc_graph_energy
from multicell.multicell_visualize import \
    graph_lattice_uniplotter, graph_lattice_reference_overlap_plotter, graph_lattice_projection_composite
from multicell.multicell_replot import replot_scatter_dots
from singlecell.singlecell_class import Cell
from singlecell.singlecell_constants import ASYNC_BATCH
from singlecell.singlecell_functions import \
    state_memory_overlap_alt, state_memory_projection_alt, state_to_label, update_state_infbeta_simple
from utils.file_io import run_subdir_setup, runinfo_append, write_general_arr


# TODO convert to main attribute of class, passable (defaults to False?)
print('DYNAMICS_FIXED_UPDATE_ORDER:', DYNAMICS_FIXED_UPDATE_ORDER)


class Multicell:
    """
    Primary object for multicell simulations.
    Current state of the graph is tracked using attribute: graph_state
     - graph_state_arr is an [N*M x T] 2D array
     - get state of a specific cell using get_cell_state(idx)

    simsetup: dictionary created using singlecell_simsetup() in singlecell_simsetup.py

    Expected kwargs for __init__
        num_cells
        graph_style
        gamma
        autocrine
        flag_housekeeping
        beta
        num_steps
        plot_period

    Optional kwargs for __init__
        graph_kwargs
        exosome_string
        exosome_remove_ratio
        field_applied
        kappa
        flag_state_int
        seed
        init_state_path

    Attributes
        simsetup:          (dict) simsetup with internal and external gene regulatory rules
        num_genes:         (int) aka N -- internal dimension of each cell
        num_celltypes:     (int) aka P -- patterns encoded in each cell
        num_cells:         (int) aka M -- number of nodes in cell-cell graph
        total_spins:       (int) N * M
        matrix_J:          (arr) N x N -- governs internal dynamics
        matrix_W:          (arr) N x N -- governs cell-cell signalling
        matrix_A:          (arr) M x M -- adjacency matrix
        matrix_J_multicell:(arr) NM x NM -- ising interaction matrix for the entire graph
        graph_style:       (str) style of adjacency matrix for cell-cell interactions
            supported: meanfield, general, or lattice_square
        graph_kwargs:      (dict) optional kwargs for call to self.build_adjacency(...)
            Case: meanfield:       N/A
            Case: general:         expects 'prebuilt_adjacency'
            Case: lattice_square:  expects 'initialization_style', 'search_radius', 'periodic'
        autocrine:         (bool) do cells signal/interact with themselves?
        gamma:             (float) cell-cell signalling field strength
        exosome_string:    (str) see valid exosome strings; adds exosomes to field_signal
        exosome_remove_ratio: (float) if exosomes act, how much of the cell state to subsample?
        field_applied:     (arr) NM x T (total_steps) -- the external/manual applied field
        kappa:             (float) scales overall strength of applied/manual field
        flag_housekeeping: (bool) is there a housekeeping component to the manual field?
        seed:              (int) controls all random calls

    Attributes specific to graph state and its dynamics
        init_state_path: (None, arr) arr storing init state of the multicell graph
        graph_kwargs:    (dict) see above
        beta:            (float or arr T) inverse temperature for dynamics
        total_steps:     (int) aka T - total 'lattice steps' to simulate
        current_step:    (int) step counter
        dynamics_blockparallel: (bool) synchronized lattice updates (can use GPU when True)
        plot_period:     (int) lattice plot period

    Data storage attributes
        run_basedir:      (str) dir below runs folder, defaults to 'multicell_sim'
        run_subdir:       (str) override the datetime format for multicell_sim sub-dirs
        flag_state_int:  (bool) track and plot the int rep of cell state (asserts low N)
        io_dict:         (dict) stores output file paths according to utils.file_io
        data_dict:       (dict) live data storage for graph state and computed properties
    """
    def __init__(self, simsetup, verbose=True, **kwargs):
        if verbose:
            print('Initializing Multicell class object...',)
        # core parameters
        self.simsetup = simsetup
        self.matrix_J = simsetup['J']
        self.matrix_W = simsetup['FIELD_SEND']
        self.num_genes = simsetup['N']
        self.num_celltypes = simsetup['P']
        self.num_cells = kwargs['num_cells']
        self.total_spins = self.num_cells * self.num_genes
        self.graph_style = kwargs['graph_style']
        self.autocrine = kwargs.get('autocrine', AUTOCRINE)
        self.verbose = verbose
        # random seed
        # see https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
        self.seed = kwargs.get('seed',
                               np.random.randint(low=0, high=1e5))    # TODO use throughout and test
        # field 'signal': cell-cell signalling
        self.gamma = kwargs['gamma']  # aka field_signal_strength
        self.exosome_string = kwargs.get('exosome_string', EXOSTRING)
        self.exosome_remove_ratio = kwargs.get('exosome_remove_ratio', EXOSOME_REMOVE_RATIO)
        # field 'applied': manual/applied field including possible housekeeping gene portion
        self.field_applied = kwargs.get('field_applied', None)
        self.kappa = kwargs.get('kappa', 0.0)
        self.flag_housekeeping = kwargs['flag_housekeeping']
        self.num_housekeeping = self.simsetup['K']
        # simulation/dynamics properties
        self.beta = kwargs['beta']
        self.current_step = 0
        self.total_steps = kwargs['total_steps']
        self.plot_period = kwargs['plot_period']
        self.dynamics_blockparallel = kwargs.get('flag_blockparallel', BLOCK_UPDATE_LATTICE)  # GPU?
        # bool: speedup dynamics (check vs graph attributes)
        # self.dynamics_meanfield = ...  # TODO reimplement?
        # graph initialization
        # TODO replace lattice by graph everywhere
        self.graph_kwargs = kwargs.get('graph_kwargs', {})
        self.initialization_style = self.graph_kwargs.get('initialization_style', None)
        self.matrix_A = self.build_adjacency()
        self.graph_state_arr = np.zeros((self.total_spins, self.total_steps), dtype=int)
        self.init_state_path = kwargs.get('init_state_path', None)
        if self.init_state_path is not None:
            flag_txt = self.init_state_path[0:4] == '.txt'
            manual_init_state = state_load(self.init_state_path, cells_as_cols=False,
                                           num_genes=self.num_genes, num_cells=self.num_cells,
                                           txt=flag_txt)
            self.graph_state_arr[:, 0] = manual_init_state
        else:
            self.graph_state_arr[:, 0] = self.init_graph_state()
        # initialize matrix_J_multicell (used explicitly for parallel dynamics)
        self.matrix_J_multicell = self.build_J_multicell()
        # metadata
        self.run_basedir = kwargs.get('run_basedir', 'multicell_sim')
        self.run_subdir = kwargs.get('run_subdir', None)
        self.flag_state_int = kwargs.get('flag_state_int', False)
        self.io_dict = self.init_io_dict()
        self.data_dict = self.init_data_dict()
        # final assertion of attributes
        self.init_assert_and_sanitize()
        if verbose:
            print('done')

    # TODO cleanup
    def init_assert_and_sanitize(self):

        # field signal
        assert self.exosome_string in VALID_EXOSOME_STRINGS
        assert 0.0 <= self.exosome_remove_ratio < 1.0
        assert 0.0 <= self.gamma <= 2000.0

        # field applied (not it has length NM
        if self.num_housekeeping > 0:
            assert self.flag_housekeeping

        if self.field_applied is not None:
            # first: rescale by kappa
            self.field_applied = self.field_applied * self.kappa
            # check that shape is as expected (N*M x timesteps)
            assert len(np.shape(self.field_applied)) in (1, 2)
            if len(self.field_applied.shape) == 2:
                assert self.field_applied.shape[1] == self.total_steps
                if self.field_applied.shape[0] != self.total_spins:
                    # if size N we duplicate it for each cell if needed)
                    assert self.field_applied.shape[0] == self.num_genes
                    print('Warning: field_applied is size N x T but expect NM x T,'
                          'it will be duplicated onto each cell')
                    self.field_applied = np.array(
                        [self.field_applied for _ in range(self.num_cells)]). \
                        reshape(self.total_spins, self.total_steps)
            else:
                if self.field_applied.shape[0] != self.total_spins:
                    # if size N we duplicate it for each cell if needed)
                    assert self.field_applied.shape[0] == self.num_genes
                    print('Warning: field_applied is size N but expect NM x T,'
                          'it will be duplicated onto each cell and each timestep')
                    self.field_applied = np.array([self.field_applied for _ in range(self.num_cells)]).\
                        reshape(self.total_spins)
                self.field_applied = np.array([self.field_applied for _ in range(self.total_steps)]).T
        else:
            self.field_applied = np.zeros((self.total_spins, self.total_steps))
        print('field_applied.shape:', self.field_applied.shape)         # TODO remove?

        # beta (temperature) check
        if isinstance(self.beta, np.ndarray):
            assert self.beta.shape == (self.total_steps,)
        else:
            assert isinstance(self.beta, float)
            self.beta = np.array([self.beta for _ in range(self.total_steps)])

        # TODO other checks to reimplement
        # misc checks
        assert type(self.total_steps) is int
        assert type(self.plot_period) is int

        # graph = lattice square case
        if self.graph_style == 'lattice_square':
            self.graph_kwargs['sidelength'] = int(np.sqrt(self.num_cells) + 0.5)
            assert 'periodic' in self.graph_kwargs.keys()
            assert self.graph_kwargs['search_radius'] <= 0.5 * self.graph_kwargs['sidelength']
            assert self.graph_kwargs['initialization_style'] in VALID_BUILDSTRINGS

        assert self.graph_style in ['general', 'meanfield', 'lattice_square']
        return

    def simulation_reset(self, provided_init_state=None):
        # reset counters
        self.current_step = 0
        # reset graph state
        self.graph_state_arr = np.zeros((self.total_spins, self.total_steps), dtype=int)
        # reset initial state
        if provided_init_state is None:
            if self.init_state_path is not None:
                flag_txt = self.init_state_path[0:4] == '.txt'
                manual_init_state = state_load(self.init_state_path, cells_as_cols=False,
                                               num_genes=self.num_genes, num_cells=self.num_cells,
                                               txt=flag_txt)
                self.graph_state_arr[:, 0] = manual_init_state
            else:
                self.graph_state_arr[:, 0] = self.init_graph_state()
        else:
            self.graph_state_arr[:, 0] = provided_init_state

    # TODO cleanup
    def init_io_dict(self):
        io_dict = run_subdir_setup(run_subfolder=self.run_basedir,
                                   timedir_override=self.run_subdir)
        info_list = [['seed', self.seed],
                     ['memories_path', self.simsetup['memories_path']],
                     ['script', 'multicell_simulate_old.py'],
                     ['num_cells', self.num_cells],
                     ['total_steps', self.total_steps],
                     ['graph_style', self.graph_style],
                     ['initialization_style', self.initialization_style],
                     ['search_radius', self.graph_kwargs.get('search_radius', None)],
                     ['gamma', self.gamma],
                     ['autocrine', self.autocrine],
                     ['exosome_string', self.exosome_string],
                     ['exosome_remove_ratio', self.exosome_remove_ratio],
                     ['kappa', self.kappa],
                     ['field_applied', self.field_applied],
                     ['flag_housekeeping', self.flag_housekeeping],
                     ['num_housekeeping', self.num_housekeeping],
                     ['beta', self.beta],
                     ['random_mem', self.simsetup['random_mem']],
                     ['random_W', self.simsetup['random_W']],
                     ['dynamics_blockparallel', self.dynamics_blockparallel],
                     ['init_state_path', self.init_state_path],
                     ]
        runinfo_append(io_dict, info_list, multi=True)
        # conditionally store random mem XI, signalling W, and adjacency A
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'matrix_XI.txt',
                   self.simsetup['XI'], delimiter=',', fmt='%d')
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'matrix_W.txt',
                   self.matrix_W, delimiter=',', fmt='%.4f')
        # save matrix 'A' to file
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'matrix_A.txt',
                   self.matrix_A, fmt='%.4f')
        # save matrix 'J' (single cell) to file
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'matrix_J.txt',
                   self.simsetup['J'], fmt='%.4f')
        # save matrix 'J_multicell' to file
        np.savez_compressed(io_dict['simsetupdir'] + os.sep + 'matrix_J_multicell.npz',
                            self.matrix_J_multicell)
        return io_dict

    # TODO full + support for non lattice too
    # TODO this can be achieved by hiding the grid aspect of cells -- just use their graph node index
    #  and convert to loc i, j on the fly (or as cell class attribute...)
    def init_data_dict(self):
        """
        Currently has the core keys
            memory_proj_arr:           arr M x P x T
            memory_overlap_arr:        arr M x P x T
            graph_energy:              arr T x 5
            compressibility_full:      arr T x 3
        Optional keys
            cell_state_int:            arr 1 x T
        """
        data_dict = {}
        # stores: H_multi, H_self, H_app, H_pairwise_scaled
        data_dict['graph_energy'] = np.zeros((self.total_steps, 5))
        # stores: compressibility ratio, eta, eta0
        data_dict['compressibility_full'] = np.zeros((self.total_steps, 3))
        # stores: memory projection/overlap for each memory for each cell
        data_dict['memory_proj_arr'] = np.zeros(
            (self.num_cells, self.simsetup['P'], self.total_steps))
        data_dict['memory_overlap_arr'] = np.zeros(
            (self.num_cells, self.simsetup['P'], self.total_steps))
        # stores: memory projection for each memory for each cell
        if self.flag_state_int:
            data_dict['cell_state_int'] = np.zeros((self.num_cells, self.total_steps), dtype=int)
        return data_dict

    def build_adjacency(self, plot=False):
        """
        Builds node adjacency matrix arr_A based on graph style and hyperparameters
        Supported graph styles
            'meanfield': all nodes are connected
                kwargs: N/A
            'general': will use the supplied array 'prebuilt_adjacency'
                kwargs: 'prebuilt_adjacency'
            'lattice_square': adjacency style for square lattice with r-nearest neighbour radius
                kwargs: 'search_radius'
        """
        if self.graph_style == 'meanfield':
            arr_A = np.ones((self.num_cells, self.num_cells))

        elif self.graph_style == 'general':
            arr_A = self.graph_kwargs.get('prebuilt_adjacency', None)
            if arr_A is None:
                arr_A = adjacency_general(self.num_cells)

        else:
            assert self.graph_style == 'lattice_square'
            # check the number is a perfect square
            sidelength = int(np.sqrt(self.num_cells) + 0.5)
            assert sidelength ** 2 == self.num_cells
            self.graph_kwargs['sidelength'] = sidelength
            search_radius = self.graph_kwargs['search_radius']
            arr_A = adjacency_lattice_square(sidelength, self.num_cells, search_radius,
                                             periodic_bc=self.graph_kwargs['periodic'])

        # autocrine loop before returning adjacency matrix (set all diags to 1 or 0)
        # note this will override 'prebuilt_adjacency' in case of self.graph_style == 'general'
        if self.autocrine:
            np.fill_diagonal(arr_A, 2)
        else:
            np.fill_diagonal(arr_A, 0)

        if plot:
            plt.imshow(arr_A)
            plt.title('Cell-cell adjacency matrix')
            plt.show()

        return arr_A

    def build_J_multicell(self, gamma=None, plot=False):

        if gamma is None:
            gamma = self.gamma

        W_scaled = gamma * self.matrix_W
        W_scaled_sym = 0.5 * (W_scaled + W_scaled.T)

        # Term A: self interactions for each cell (diagonal blocks of multicell J_block)
        if self.autocrine:
            J_diag_blocks = np.kron(np.eye(self.num_cells), self.simsetup['J'] + W_scaled_sym)
        else:
            J_diag_blocks = np.kron(np.eye(self.num_cells), self.simsetup['J'])

        # Term B of J_multicell (cell-cell interactions)
        adjacency_arr_lowtri = np.tril(self.matrix_A, k=-1)
        adjacency_arr_uptri = np.triu(self.matrix_A, k=1)
        J_offdiag_blocks = np.kron(adjacency_arr_lowtri, W_scaled.T) \
                           + np.kron(adjacency_arr_uptri, W_scaled)

        # build final J multicell matrix
        J_multicell = J_diag_blocks + J_offdiag_blocks

        if plot:
            plt.imshow(J_multicell)
            plt.title('Multicell gene interaction matrix')
            plt.show()
        return J_multicell

    # TODO fix this old approach
    def init_graph_state(self):
        """"
        Initialize the state of each cell in the Multicell dynamical system
        """
        assert self.graph_style == 'lattice_square' # TODO not this
        initialization_style = self.graph_kwargs['initialization_style']

        if self.graph_style == 'general':
            pass # TODO
        elif self.graph_style == 'meanfield':
            pass # TODO
        else:
            assert self.graph_style == 'lattice_square'
            sidelength = self.graph_kwargs['sidelength']

            # 1) use old lattice initializer
            if initialization_style == "mono":
                type_1_idx = 0
                list_of_type_idx = [type_1_idx]
            if initialization_style == "dual":
                type_1_idx = 0
                type_2_idx = 1
                list_of_type_idx = [type_1_idx, type_2_idx]
            if initialization_style == "memory_sequence":
                list_of_type_idx = list(range(self.simsetup['P']))
                # random.shuffle(list_of_type_idx)  # TODO shuffle or not?
            if initialization_style == "random":
                list_of_type_idx = list(range(self.simsetup['P']))
            lattice = build_lattice_main(
                sidelength, list_of_type_idx, initialization_style, self.simsetup,
                seed=self.seed, verbose=self.verbose)
            # print list_of_type_idx

            # 2) now convert the lattice to a graph state (tall NM vector)
            graph_state = self.TEMP_graph_state_from_lattice(lattice, sidelength, verbose=self.verbose)

        return graph_state

    # TODO remove
    def TEMP_graph_state_from_lattice(self, lattice, sidelength, verbose=True):
        if verbose:
            print('call to TEMP_graph_state_from_lattice() -- remove this function')
        assert self.graph_style == 'lattice_square'
        N = self.num_genes
        s_block = np.zeros(self.total_spins)
        for a in range(self.num_cells):
            arow, acol = lattice_square_int_to_loc(a, sidelength)
            cellstate = np.copy(
                lattice[arow][acol].get_current_state())
            s_block[a * N: (a+1) * N] = cellstate
        return s_block

    def step_dynamics_parallel(self, next_step, field_applied, beta):
        """
        Performs one "graph step": each cell has an opportunity to update its state
        Returns None (operates directly on self.graph_state_arr)
        """
        total_field = np.zeros(self.total_spins)
        internal_field = np.dot(self.matrix_J_multicell, self.graph_state_arr[:, next_step - 1])
        total_field += internal_field
        total_field += field_applied

        # probability that site i will be "up" after the timestep
        prob_on_after_timestep = 1 / (1 + np.exp(-2 * beta * total_field))
        np.random.seed(self.seed); rsamples = np.random.rand(self.total_spins)
        for idx in range(self.total_spins):
            if prob_on_after_timestep[idx] > rsamples[idx]:
                self.graph_state_arr[idx, next_step] = 1.0
            else:
                self.graph_state_arr[idx, next_step] = -1.0
        return

    # TODO how to support different graph types without using the SpatialCell class?
    #  currently all old code from multicell+simulate
    # TODO Meanfield setting previously used for speedups, but removed now:
    def step_dynamics_async(self, next_step, field_applied, beta):

        if DYNAMICS_FIXED_UPDATE_ORDER:
            seed_local = 0
        else:
            seed_local = self.seed

        cell_indices = list(range(self.num_cells))
        random.seed(seed_local); random.shuffle(cell_indices)

        # need to move current state to next step index, then operate on that column 'in place'
        self.graph_state_arr[:, next_step] = np.copy(self.graph_state_arr[:, next_step - 1])

        for idx, node_idx in enumerate(cell_indices):
            cell_state = self.get_cell_state(node_idx, next_step)
            spin_idx_low = self.num_genes * node_idx
            spin_idx_high = self.num_genes * (node_idx + 1)

            # extract applied field specific to cell at given node
            field_applied_on_cell = field_applied[spin_idx_low: spin_idx_high]

            #  note that a cells neighboursa are the ones which 'send' to it
            #  if A_ij != 0, then there is a connection from i to j
            #  to get all the senders to cell i, we need to look at col i
            graph_neighbours_col = self.matrix_A[:, node_idx]
            graph_neighbours = [node for node, i in enumerate(graph_neighbours_col) if i != 0]

            # TODO (?) scale field contributions by A_ij value (to account for A_ii = 1)

            # signaling field part 1
            # TODO pass seed_local to state_subsample in exo field
            field_signal_exo, _ = general_exosome_field(
                self, node_idx, next_step, neighbours=graph_neighbours)
            # signaling field part 2
            field_signal_W = general_paracrine_field(
                self, node_idx, next_step, flag_01=SEND_01, neighbours=graph_neighbours)
            # sum the two field contributions
            field_signal_unscaled = field_signal_exo + field_signal_W
            field_signal = self.gamma * field_signal_unscaled

            dummy_cell = Cell(np.copy(cell_state), 'fake_cell',
                              self.simsetup['CELLTYPE_LABELS'],
                              self.simsetup['GENE_LABELS'],
                              state_array=None,
                              steps=None)

            dummy_cell.update_state(
                beta=beta,
                intxn_matrix=self.matrix_J,  # TODO J diag block should be modified for autocrine case
                field_signal=field_signal,
                field_signal_strength=1.0,
                field_applied=field_applied_on_cell,
                field_applied_strength=1.0,
                seed=seed_local)

            self.graph_state_arr[spin_idx_low:spin_idx_high, next_step] = \
                dummy_cell.get_current_state()

            """
            if turn % (
                    120 * plot_period) == 0:  # proj vis of each cell (slow; every k steps)
                fig, ax, proj = cell. \
                    plot_projection(simsetup['A_INV'], simsetup['XI'], proj=cell_proj,
                                    use_radar=False, pltdir=io_dict['latticedir'])"""
        return

    def step_dynamics_async_deterministic_no_exo_fields(self, next_step, field_applied, beta):
        assert beta == np.Inf
        if DYNAMICS_FIXED_UPDATE_ORDER:
            seed_local = 0
        else:
            seed_local = self.seed

        cell_indices = list(range(self.num_cells))
        random.seed(seed_local); random.shuffle(cell_indices)

        # need to move current state to next step index, then operate on that column 'in place'
        self.graph_state_arr[:, next_step] = np.copy(self.graph_state_arr[:, next_step - 1])

        for idx, node_idx in enumerate(cell_indices):
            cell_state = self.get_cell_state(node_idx, next_step)
            spin_idx_low = self.num_genes * node_idx
            spin_idx_high = self.num_genes * (node_idx + 1)

            J_multicell_chunk = self.matrix_J_multicell[spin_idx_low:spin_idx_high, :]
            J_multicell_chunk_nodiag = np.copy(J_multicell_chunk)
            J_multicell_chunk_nodiag[0:self.num_genes, spin_idx_low:spin_idx_high] = 0
            paracrine_field = np.dot(J_multicell_chunk_nodiag,
                                     self.graph_state_arr[:, next_step])

            field_applied_on_cell = field_applied[spin_idx_low: spin_idx_high]
            netstatic_field_applied_on_cell = paracrine_field + field_applied_on_cell

            cell_next_state = update_state_infbeta_simple(
                cell_state, self.matrix_J, netstatic_field_applied_on_cell,
                async_batch=ASYNC_BATCH, async_flag=True, seed=seed_local)
            self.graph_state_arr[spin_idx_low:spin_idx_high, next_step] = cell_next_state
        return

    def count_lattice_states(self, step, verbose=False):
        """
        Returns list of unique labels and list of their occurrences at "step"
        """
        labels = [0] * self.num_cells
        for node_idx in range(self.num_cells):
            spin_idx_low = self.num_genes * node_idx
            spin_idx_high = self.num_genes * (node_idx + 1)

            cellstate = self.graph_state_arr[spin_idx_low:spin_idx_high, step]
            labels[node_idx] = state_to_label(tuple(cellstate))

        labels_unique, labels_counts = np.unique(labels, return_counts=True)
        if verbose:
            for i, elem in enumerate(labels_unique):
                print('Unique label: %d, with %d occurrences' % (elem, labels_counts[i]))

        return labels_unique, labels_counts

    def step_datadict_update_global(self, step, fill_to_end=False):
        """
        Following a simulation multicell step, update the data dict.
          fill_to_end: if True, fill all steps after 'step' with the calculated value (for early FP)
        See init_data_dict() for additional documentation.
        """
        if fill_to_end:
            step_indices = range(step, self.total_steps)
        else:
            step_indices = step
        # 1) compressibility statistics
        graph_state_01 = ((1 + self.graph_state_arr[:, step])/2).astype(int)
        comp_ratio = calc_compression_ratio(
            graph_state_01, eta_0=None, datatype='full', elemtype=np.int, method='manual')
        self.data_dict['compressibility_full'][step_indices, :] = comp_ratio
        # 2) energy statistics
        energy_values = calc_graph_energy(self, step, norm=True)
        self.data_dict['graph_energy'][step_indices, :] = energy_values
        # 3) node-wise projection on the encoded singlecell types
        for i in range(self.num_cells):
            cell_state = self.get_cell_state(i, step)
            # get the projections/overlaps
            overlap_vec = state_memory_overlap_alt(cell_state, self.num_genes, self.simsetup['XI'])
            proj_vec = state_memory_projection_alt(
                cell_state, self.simsetup['A_INV'], self.num_genes, self.simsetup['XI'],
                overlap_vec=overlap_vec)
            # store the projections/overlaps
            self.data_dict['memory_overlap_arr'][i, :, step_indices] = overlap_vec
            self.data_dict['memory_proj_arr'][i, :, step_indices] = proj_vec
            # 4) node-wise storage of the integer representation of the state
            if self.flag_state_int:
                self.data_dict['cell_state_int'][i, step_indices] = state_to_label(tuple(cell_state))
        return

    def step_state_save(self, step, prefix='X', suffix='', cells_as_cols=True):
        """
        Saves graph state at timestep as a txt file
            cells_as_cols: reshape to 2D arr so each column reps a cell
        To load:
            X = np.loadtxt(fpath)
            num_genes, num_cells = X.shape
            X = X.reshape((num_genes*num_cells), order='F')
        """
        outdir = self.io_dict['statesdir']
        X = self.graph_state_arr[:, step]
        fpath = outdir + os.sep + '%s_%d%s.npz' % (prefix, step, suffix)
        fmt = '%d'
        if cells_as_cols:
            np.savez_compressed(fpath, X.reshape((self.num_genes, self.num_cells), order='F'), fmt=fmt)
        else:
            np.savez_compressed(fpath, X, fmt=fmt)
        return fpath

    # TODO remove lattice square assert (generalize)
    def step_state_visualize(self, step, flag_uniplots=False, fpaths=None):
        assert self.graph_style == 'lattice_square'
        nn = self.graph_kwargs['sidelength']
        if fpaths is None:
            fpaths = [None, None, None, None]

        # plot type A
        graph_lattice_projection_composite(self, step, use_proj=False, fpath=fpaths[0])
        graph_lattice_projection_composite(self, step, use_proj=True, fpath=fpaths[1])
        # plot type B
        graph_lattice_reference_overlap_plotter(self, step, fpath=fpaths[2])
        # plot type C
        if flag_uniplots:
            for mu in range(self.simsetup['P']):
                graph_lattice_uniplotter(self, step, nn, self.io_dict['latticedir'],
                                         mu, use_proj=False)
                graph_lattice_uniplotter(self, step, nn, self.io_dict['latticedir'],
                                         mu, use_proj=True)
        # plot type D
        if self.num_genes == 9 and self.graph_style == 'lattice_square':
            nn = self.graph_kwargs['sidelength']
            outpath = fpaths[3]
            if outpath is None:
                outpath = self.io_dict['latticedir'] + os.sep + 'dots_%d' % step
            X = self.graph_state_arr[:, step]
            X = X.reshape((self.num_genes, self.num_cells), order='F')  # reshape as 2D arr
            replot_scatter_dots(X, nn, outpath, state_int=self.flag_state_int)
        return

    def step_state_visualize_alt(self, step, flag_uniplots=False, fpaths=None):
        assert self.graph_style == 'lattice_square'
        nn = self.graph_kwargs['sidelength']
        if fpaths is None:
            fpaths = [None, None, None]

        # plot type A
        graph_lattice_projection_composite(self, step, use_proj=False, fpath=fpaths[0])
        graph_lattice_projection_composite(self, step, use_proj=True, fpath=fpaths[1])
        # plot type B
        graph_lattice_reference_overlap_plotter(self, step, fpath=fpaths[2])
        # plot type C
        if flag_uniplots:
            for mu in range(self.simsetup['P']):
                graph_lattice_uniplotter(self, step, nn, self.io_dict['latticedir'],
                                         mu, use_proj=False)
                graph_lattice_uniplotter(self, step, nn, self.io_dict['latticedir'],
                                         mu, use_proj=True)
        return

    def check_if_fixed_point(self, state_to_compare, step, msg=None):
        """
        :param state_to_compare: to check array equality with current graph state
        :return: bool
        """
        if np.all(self.graph_state_arr[:, step] == state_to_compare):
            if msg is not None:
                print(msg)
            return True
        else:
            return False

    # TODO handle case of dynamics_async
    def dynamics_full(self, flag_visualize=True, flag_datastore=True, flag_savestates=True,
                      end_at_fp=False, verbose=False):
        """
            flag_visualize: optionally force off datadict updates (speedup)
            flag_datastore: optionally force off calls to step_state_visualize updates (speedup)
            end_at_fp:      terminate before self.total_steps if a fixed point is reached
        Notes:
            -can replace update_with_signal_field with update_state to simulate ensemble
            of non-interacting n**2 cells
        """
        # TODO removed features
        #  - meanfield_global_field() subfunction recreate if needed (cost savings if async)
        #  - cell_locations = get_cell_locations(lattice, n)
        #  - for loc in cell_locations:
        #        update_datadict_timestep_cell(lattice, loc, memory_idx_list, 0)
        #  - cell specific datastorage call

        # 1) initial data storage and plotting
        if flag_savestates:
            self.step_state_save(0)
        if flag_datastore:
            self.step_datadict_update_global(0)  # measure initial state
        if flag_visualize:
            self.step_state_visualize(0)

        if not verbose:
            fp_msg = None
            fp_msg_2flicker = None

        # 2) main loop
        for step in range(1, self.total_steps):
            if verbose:
                print('Dynamics step: ', step)

            # applied field and beta schedule
            field_applied_step = self.field_applied[:, step - 1]
            beta_step = self.beta[step - 1]

            # choose graph update function
            if self.dynamics_blockparallel:
                graph_update_fn = self.step_dynamics_parallel
            else:
                if beta_step == np.Inf and self.exosome_string == 'no_exo_field':
                    graph_update_fn = self.step_dynamics_async_deterministic_no_exo_fields
                else:
                    graph_update_fn = self.step_dynamics_async
            # call (slow step)
            graph_update_fn(step, field_applied_step, beta_step)

            # compute lattice properties (assess global state)
            # TODO 1 - consider lattice energy at each cell update (not lattice update)
            # TODO 2 - speedup lattice energy calc by using info from state update calls...
            if flag_savestates:
                self.step_state_save(step)
            if flag_datastore:
                self.step_datadict_update_global(step)

            # periodic plotting call
            if flag_visualize and step % self.plot_period == 0:  # plot the lattice
                self.step_state_visualize(step)
                #self.step_state_save(step, memory_idx_list)  # TODO call to save

            # update class attributes TODO any others to increment?
            self.current_step += 1

            # (optional) simulation termination condition: fixed point reached
            if end_at_fp:
                if verbose:
                    fp_msg = 'FP reached early (step %d of %d), terminating dynamics' % \
                             (step, self.total_steps)
                    fp_msg_2flicker = 'Flicker (2-state) ' + fp_msg
                fp_reached = self.check_if_fixed_point(
                    self.graph_state_arr[:, step - 1], step, msg=fp_msg)
                fp_reached_flicker = self.check_if_fixed_point(
                    self.graph_state_arr[:, step - 2], step, msg=fp_msg_2flicker)
                if fp_reached or fp_reached_flicker:
                    break

    def simulation_standard(self):
        # run the simulation
        self.dynamics_full()

        # """check the data dict"""
        self.plot_datadict_memory(use_proj=False)
        self.plot_datadict_memory(use_proj=True)

        # """write and plot cell state timeseries"""
        # write_state_all_cells(lattice, io_dict['datadir'])
        self.mainloop_data_dict_write()
        self.mainloop_data_dict_plot()

        print("\nMulticell simulation complete - output in %s" % self.io_dict['basedir'])
        return

    def simulation_fast(self, no_datatdict=False, no_visualize=False, end_at_fp=True, verbose=True):
        if no_datatdict: assert no_visualize

        # """get data dict from initial state"""
        if not no_datatdict:
            self.step_datadict_update_global(0, fill_to_end=False)    # measure initial state
        if not no_visualize:
            self.step_state_visualize(0)                            # visualize initial state

        # run the simulation
        self.dynamics_full(flag_visualize=False, flag_datastore=False, end_at_fp=end_at_fp,
                           verbose=verbose)

        # """get data dict from final two states"""
        # Note: if two-state limit cycle, the 'fill to end' is misleading
        if not no_datatdict:
            self.step_datadict_update_global(self.current_step - 1, fill_to_end=False)  # measure
            self.step_datadict_update_global(self.current_step, fill_to_end=False)      # measure
        if not no_visualize:
            self.step_state_visualize(self.current_step - 1)                            # visualize
            self.step_state_visualize(self.current_step)                                # visualize

        # """make copies of relevant save states"""
        sdir = self.io_dict['statesdir']
        # copy "final - 1" to X_secondlast.npz
        shutil.copyfile(sdir + os.sep + 'X_%d.npz' % (self.current_step - 1),
                        sdir + os.sep + 'X_secondlast.npz')
        # copy final to X_last.npz
        shutil.copyfile(sdir + os.sep + 'X_%d.npz' % (self.current_step),
                        sdir + os.sep + 'X_last.npz')

        # """check the data dict"""
        if not no_visualize:
            self.plot_datadict_memory(use_proj=False)
            self.plot_datadict_memory(use_proj=True)

        # """write and plot cell state timeseries"""
        # write_state_all_cells(lattice, io_dict['datadir'])
        if not no_visualize:
            self.mainloop_data_dict_write()
            self.mainloop_data_dict_plot()

        if verbose:
            print("\nMulticell simulation complete - output in %s" % self.io_dict['basedir'])
        return self.current_step

    def get_cell_state(self, cell_idx, step):
        assert 0 <= cell_idx < self.num_cells
        a = self.num_genes * cell_idx
        b = self.num_genes * (cell_idx + 1)
        return self.graph_state_arr[a:b, step]

    def get_field_on_cell(self, cell_idx, step):
        assert 0 <= cell_idx < self.num_cells
        a = self.num_genes * cell_idx
        b = self.num_genes * (cell_idx + 1)
        return self.field_applied[a:b, step]

    def cell_cell_overlap(self, idx_a, idx_b, step):
        s_a = self.get_cell_state(idx_a, step)
        s_b = self.get_cell_state(idx_b, step)
        return np.dot(s_a.T, s_b) / self.num_genes

    def plot_datadict_memory(self, use_proj=True):
        if use_proj:
            datakey = 'memory_proj_arr'
            datatitle = 'projection'
        else:
            datakey = 'memory_overlap_arr'
            datatitle = 'overlap'
        # check the data dict
        for mu in range(self.simsetup['P']):
            #print(self.data_dict[datakey][:, mu, :])
            plt.plot(self.data_dict[datakey][:, mu, :].T)
            plt.ylabel('%s of all cells onto type: %s' %
                       (datatitle, self.simsetup['CELLTYPE_LABELS'][mu]))
            plt.xlabel('Time (full lattice steps)')
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + '%s%d.png' % (datatitle, mu))
            plt.clf()  # plt.show()
        return

    # TODO make savestring property to append to files and plots?
    """e.g 
    '%s_%s_t%d_proj%d_remove%.2f_exo%.2f.png' %
    (self.exosome_string, self.initialization_style, self.total_steps, memory_idx,
     self.exosome_remove_ratio, field_signal_strength)"""
    def mainloop_data_dict_write(self):
        if self.flag_state_int:
            write_grid_state_int_alt(self.data_dict['cell_state_int'], self.io_dict['datadir'])
        if 'graph_energy' in list(self.data_dict.keys()):
            write_general_arr(self.data_dict['graph_energy'], self.io_dict['datadir'],
                              'graph_energy', txt=True, compress=False)
        if 'compressibility_full' in list(self.data_dict.keys()):
            write_general_arr(self.data_dict['compressibility_full'], self.io_dict['datadir'],
                              'compressibility_full', txt=True, compress=False)

    def mainloop_data_dict_plot(self):
        if 'graph_energy' in list(self.data_dict.keys()):
            plt.plot(self.data_dict['graph_energy'][:, 0], '-ok', label=r'$H_{\mathrm{quad}}$',
                     alpha=0.5)
            plt.plot(self.data_dict['graph_energy'][:, 1], '--ok', label=r'$H_{\mathrm{total}}$')
            plt.plot(self.data_dict['graph_energy'][:, 2], '--b', alpha=0.7,
                     label=r'$H_{\mathrm{self}}$')
            plt.plot(self.data_dict['graph_energy'][:, 3], '--g', alpha=0.7,
                     label=r'$H_{\mathrm{app}}$')
            plt.plot(self.data_dict['graph_energy'][:, 4], '--r', alpha=0.7,
                     label=r'$H_{\mathrm{pairwise}}$')
            plt.plot(self.data_dict['graph_energy'][:, 1] -
                     self.data_dict['graph_energy'][:, 3], '--o',
                     color='gray', label=r'$H_{\mathrm{total}} - H_{\mathrm{app}}$')
            plt.title(r'Multicell hamiltonian over time')
            plt.ylabel(r'Graph energy')
            plt.xlabel(r'$t$ (graph steps)')
            plt.legend()
            plt.savefig(self.io_dict['plotdatadir'] + os.sep +'hamiltonian.png')

            # zoom on relevant part
            ylow = min(np.min(self.data_dict['graph_energy'][:, [2, 4]]),
                       np.min(self.data_dict['graph_energy'][:, 1] -
                              self.data_dict['graph_energy'][:, 3]))
            yhigh = max(np.max(self.data_dict['graph_energy'][:, [2, 4]]),
                        np.max(self.data_dict['graph_energy'][:, 1] -
                               self.data_dict['graph_energy'][:, 3]))
            plt.ylim(ylow - 0.1, yhigh + 0.1)
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + 'hamiltonianZoom.png')
            plt.clf()  # plt.show()

        # TODO check validity or remove
        if 'compressibility_full' in list(self.data_dict.keys()):

            assert self.graph_style == 'lattice_square'
            nn = self.graph_kwargs['sidelength']

            plt.plot(self.data_dict['compressibility_full'][:, 0], '--o', color='orange')
            plt.title(r'File compressibility ratio of the full lattice spin state')
            plt.ylabel(r'$\eta(t)/\eta_0$')
            plt.axhline(y=1.0, ls='--', color='k')

            ref_0 = calc_compression_ratio(
                x=np.zeros((nn, nn, self.simsetup['N']), dtype=int),
                method='manual',
                eta_0=self.data_dict['compressibility_full'][0, 2], datatype='full', elemtype=np.int)
            ref_1 = calc_compression_ratio(
                x=np.ones((nn, nn, self.simsetup['N']), dtype=int),
                method='manual',
                eta_0=self.data_dict['compressibility_full'][0, 2], datatype='full', elemtype=np.int)
            plt.axhline(y=ref_0[0], ls='-.', color='gray')
            plt.axhline(y=ref_1[0], ls='-.', color='blue')
            #print(ref_0, ref_0, ref_0, ref_0, 'is', ref_0, 'vs', ref_1)
            plt.xlabel(r'$t$ (lattice steps)')
            plt.ylim(-0.05, 1.01)
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + 'compresibility.png')
            plt.clf()  # plt.show()
