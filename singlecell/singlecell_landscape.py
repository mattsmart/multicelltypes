import utils.init_multiprocessing  # BEFORE numpy
import matplotlib.pyplot as plt
import numpy as np

from singlecell.singlecell_constants import MEMS_UNFOLD, DISTINCT_COLOURS
from singlecell.singlecell_functions import sorted_energies, label_to_state, get_all_fp, partition_basins, reduce_hypercube_dim, state_to_label
from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize


if __name__ == '__main__':
    # TODO embed and/or vis seeds for repeatable figures?

    HOUSEKEEPING_EXTEND = 0
    KAPPA = 0  # 1.0
    housekeeping_manual = False  # if True, set housekeeping to 0 so model is not extended
    if housekeeping_manual:
        HOUSEKEEPING = 5
    else:
        HOUSEKEEPING = HOUSEKEEPING_EXTEND

    random_mem = False
    random_W = False
    #simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA, housekeeping=HOUSEKEEPING)
    simsetup = singlecell_simsetup(
        unfolding=True,
        random_mem=random_mem,
        random_W=random_W,
        npzpath=MEMS_UNFOLD,
        housekeeping=HOUSEKEEPING_EXTEND,
        curated=True)
    print('note: N =', simsetup['N'], 'P =', simsetup['P'])
    print(simsetup['J'])

    DIM = 2

    """METHOD = 'pca'  # diffusion_custom, spectral_custom, pca
    use_hd = False
    use_proj = False
    use_magnetization = False
    plot_X = False
    beta = 1  # 2.0
    seed_reduce = 4"""

    METHOD = 'pca'  # diffusion_custom, spectral_custom, pca
    use_hd = True
    use_proj = False
    use_magnetization = True
    plot_X = False
    beta = 1  # 2.0
    seed_reduce = 8  # similar seeds to try: 1, 3, 4, 8, 9

    """METHOD = 'spectral_custom'  # diffusion_custom, spectral_custom, pca
    use_hd = True
    use_proj = True
    use_magnetization = False
    plot_X = False
    beta = 1  # 2.0
    seed_reduce = 0"""

    """METHOD = 'spectral_custom'  # diffusion_custom, spectral_custom, pca
    use_hd = False
    use_proj = False
    use_magnetization = False
    plot_X = False
    beta = 0.5  # 2.0
    seed_reduce = 0"""

    """METHOD = 'umap'  # diffusion_custom, spectral_custom, pca
    use_hd = False
    use_proj = False
    use_magnetization = False
    plot_X = False
    beta = 0.5  # 2.0
    seed_reduce = 21"""

    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    gamma = 0.0                 # global FIELD_SIGNAL_STRENGTH tunes exosomes AND sent field
    app_field = None
    if KAPPA > 0:
        #app_field = np.zeros(simsetup['N'])
        #app_field[-HOUSEKEEPING:] = 1.0
        # weird field
        app_field = np.zeros(simsetup['N'])
        app_field[0:8] = +1 * 0  # delete anti mem basin
        app_field[5:7] = -1 * 0
    print("app_field", app_field)

    # additional visualizations based on field
    """
    for kappa_mult in xrange(10):
        # TODO 1 - pca incosistent but fast, any way to keep seed same between field applications? yes if we aren't using 'all minima' hd (i.e. use pca on full states or XI hamming)
        # TODO 2 - should pass energy around more to save computation
        # TODO 3 - note for housekeeping=1 we had nbrs of the anti-minima become minima, but at 2 we avoid this...
        # TODO 4 - FOR HIGH GAMMA should only visualize / use the housekeeping ON part statespace 2**N not 2**(N+k) -- faster and cleaner -- how?
        # TODO 5 - automatically annotate memories with celltype labels and their anti's? e.g. 'A B C' on plot...
        kappa = KAPPA * kappa_mult
        print kappa_mult, KAPPA, kappa
        # energy levels report
        # TODO singlecell simsetup vis of state energies
        sorted_data, energies = sorted_energies(simsetup, field=app_field, fs=kappa)
        print sorted_data.keys()
        print sorted_data[0]
        for elem in sorted_data[0]['labels']:
            state = label_to_state(elem, simsetup['N'])
            print state, hamiltonian(state, simsetup['J']), np.dot(simsetup['ETA'], state)

        fp_annotation, minima, maxima = get_all_fp(simsetup, field=app_field, fs=kappa)
        for key in fp_annotation.keys():
            print key, label_to_state(key, simsetup['N']), fp_annotation[key]
        hd = calc_state_dist_to_local_min(simsetup, minima, X=None)
        hypercube_visualize(simsetup, 'pca', energies=energies, elevate3D=True, edges=True, all_edges=False, minima=minima, maxima=maxima)
        print
    """

    # get & report energy levels data
    print("\nSorting energy levels, finding extremes...")
    energies, _ = sorted_energies(simsetup['J'], field=app_field, fs=KAPPA, flag_sort=False)
    fp_annotation, minima, maxima = get_all_fp(
        simsetup['J'],
        field=app_field,
        fs=KAPPA,
        energies=energies)  # TODO this may have bug where: it says something is maxima but partition_basins() says minima
    print('Minima labels:')
    print(minima)
    print('label, state vec, overlap vec, proj vec, energy')
    for minimum in minima:
        minstate = label_to_state(minimum, simsetup['N'])
        print(minimum, minstate, np.dot(simsetup['XI'].T, minstate)/simsetup['N'], np.dot(simsetup['ETA'], minstate), energies[minimum])
    print('\nMaxima labels:')
    print(maxima)
    print('label, state vec, overlap vec, proj vec, energy')
    for maximum in maxima:
        maxstate = label_to_state(maximum, simsetup['N'])
        print(maximum, maxstate, np.dot(simsetup['XI'].T, maxstate)/simsetup['N'], np.dot(simsetup['ETA'], maxstate), energies[maxstate])

    print("\nPartitioning basins...")
    basins_dict, label_to_fp_label = partition_basins(
        simsetup['J'], X=None, minima=minima, field=app_field, fs=KAPPA, dynamics='async_fixed')

    print("\nMore minima stats")
    print("key, label_to_state(key, simsetup['N']), len(basins_dict[key]), key in minima, energy")
    for key in list(basins_dict.keys()):
        print(key, label_to_state(key, simsetup['N']), len(basins_dict[key]), key in minima, energies[key])

    # reduce dimension
    X_new = reduce_hypercube_dim(
        simsetup, METHOD, dim=DIM, use_hd=use_hd, use_proj=use_proj, use_magnetization=use_magnetization,
        add_noise=False, plot_X=plot_X, field=app_field, fs=KAPPA, beta=beta, seed=seed_reduce)
    print(X_new.shape)
    print(X_new[0:4, :])

    # setup basin colours for visualization
    cdict = {}
    if label_to_fp_label is not None:
        basins_keys = list(basins_dict.keys())
        assert len(basins_keys) <= 20  # get more colours
        fp_label_to_colour = {a: DISTINCT_COLOURS[idx] for idx, a in enumerate(basins_keys)}
        cdict['basins_dict'] = basins_dict
        cdict['fp_label_to_colour'] = fp_label_to_colour
        cdict['clist'] = [0] * (2 ** simsetup['N'])
        for i in range(2 ** simsetup['N']):
            cdict['clist'][i] = fp_label_to_colour[label_to_fp_label[i]]

    # setup basin labels depending on npz
    basin_labels = {}
    for idx in range(simsetup['P']):
        state = simsetup['XI'][:, idx]
        antistate = state * -1
        label = state_to_label(state)
        antilabel = state_to_label(antistate)
        basin_labels[label] = r'$\xi^%d$' % (idx+1)       # note +1 for publication no zero indexing
        basin_labels[antilabel] = r'$-\xi^%d$' % (idx+1)  # note +1 for publication no zero indexing
    i = 1
    for label in minima:
        if label not in list(basin_labels.keys()):
            if label == 0:
                basin_labels[label] = r'$S-$'
            elif label == 511:
                basin_labels[label] = r'$S+$'
            else:
                basin_labels[label] = 'spurious: %d' % i
                print('unlabelled spurious minima %d: %s' % (i, label_to_state(label, simsetup['N'])))
            i += 1

    # conditionally plot housekeeping on subspace
    housekeeping_on_labels = []  # TODO cleanup
    for label in range(2**simsetup['N']):
        state = label_to_state(label, simsetup['N'])
        substate = state[-HOUSEKEEPING:]
        if np.all(substate == 1.0):
            housekeeping_on_labels.append(label)
    print(len(housekeeping_on_labels))

    # visualize with and without basins colouring
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=True, edges=False, all_edges=False, surf=True, colours_dict=None, beta=None)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=None, beta=None)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=None, beta=beta)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=True, edges=False, all_edges=False, surf=True, colours_dict=None, beta=beta)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=True, edges=False, all_edges=False, surf=False, colours_dict=cdict)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=cdict)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=False, edges=False, all_edges=False, surf=False, colours_dict=None)
    hypercube_visualize(
        simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
        elevate3D=False, edges=True, all_edges=False, surf=False, colours_dict=cdict)

    """
    import matplotlib.pyplot as plt
    plt.imshow(simsetup['J'])
    plt.show()
    print simsetup['J']
    plt.imshow(simsetup['A'])
    plt.show()
    print simsetup['A']
    plt.imshow(simsetup['ETA'])
    plt.show()
    print simsetup['ETA']
    """

    plot_state_prob_map(simsetup['J'], beta=None)
    plot_state_prob_map(simsetup['J'], beta=5.0)
    plot_state_prob_map(simsetup['J'], beta=None, field=app_field, fs=KAPPA)
    plot_state_prob_map(simsetup['J'], beta=1.0, field=app_field, fs=KAPPA)
