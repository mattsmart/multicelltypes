import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import umap

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

from utils.file_io import RUNS_FOLDER, INPUT_FOLDER


REDUCER_SEED = 100
REDUCER_COMPONENTS = 2
UMAP_KWARGS = {
    'random_state': REDUCER_SEED,
    'n_components': REDUCER_COMPONENTS,
    'metric': 'euclidean',
    'init': 'spectral',
    'unique': False,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'spread': 1.0,
}


def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


if __name__ == '__main__':

    use_01 = True
    nn = 4999  # runs with 1000, crash with 5000, 10000 -- try to restrict to more int gammas maybe
    kk = 1   # debug: subsample multicell spins to avoid memory issue

    # Step 0) which 'manyruns' dirs to work with
    # gamma_list = [0.0, 0.05, 0.1, 0.2, 1.0, 2.0, 20.0]

    #gamma_list = [0.0, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.4, 0.6, 0.8, 0.9, 1.0, 20.0]

    #gamma_list = [0.0, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.4, 0.6, 0.8, 0.9, 1.0, 20.0]
    #gamma_list = gamma_list[::-1]

    #gamma_list = [0.0, 0.05]  # , 0.06]# , 0.07] # , 0.08, 0.09, 0.10] #, 0.15, 0.20, 0.4, 0.6, 0.8, 0.9, 1.0, 20.0]
    gamma_list = [1.0, 20.0]

    # manyruns_dirnames = ['Wrandom1_gamma%.2f_10k_fixedorder_ferro' % a for a in gamma_list]
    # manyruns_dirnames = ['Wrandom0_gamma%.2f_10k_p3_M100' % a for a in gamma_list]
    # manyruns_dirnames = ['Wrandom0_gamma%.2f_10k_fixedorder_p3_M100' % a for a in gamma_list]
    # manyruns_dirnames = ['Wrandom1_gamma%.2f_10k_fixedorder_p3_M100' % a for a in gamma_list]

    manyruns_dirnames = ['Wrandom0_gamma%.2f_10k_periodic_fixedorderV3_p3_M100' % a for a in
                         gamma_list]

    manyruns_paths = [RUNS_FOLDER + os.sep + 'multicell_manyruns' + os.sep + dirname
                      for dirname in manyruns_dirnames]

    # Step 1) umap (or other dim reduction) kwargs
    n_components = 2

    X_multi = np.zeros((len(gamma_list), nn, kk), dtype=int)
    for j, manyruns_path in enumerate(manyruns_paths):

        gamma_val = gamma_list[j]

        umap_kwargs = UMAP_KWARGS.copy()
        umap_kwargs['n_components'] = n_components  # TODO don't need to spec 'live', can embed later?
        # modify umap settings
        # umap_kwargs['unique'] = True
        # umap_kwargs['n_neighbors'] = 100
        umap_kwargs['min_dist'] = 0.25
        # umap_kwargs['spread'] = 1.0
        # umap_kwargs['metric'] = 'euclidean'

        # Step 2) make/load data
        # ...

        smod = '_last'

        agg_dir = manyruns_path + os.sep + 'aggregate'
        fpath_state = agg_dir + os.sep + 'X_aggregate%s.npz' % smod
        fpath_energy = agg_dir + os.sep + 'X_energy%s.npz' % smod
        fpath_pickle = manyruns_path + os.sep + 'multicell_template.pkl'

        X = np.load(fpath_state)['arr_0'].T  # umap wants transpose
        X_energies = np.load(fpath_energy)['arr_0'].T  # umap wants transpose (?)
        with open(fpath_pickle, 'rb') as pickle_file:
            multicell_template = pickle.load(pickle_file)  # unpickling multicell object

        if use_01:
            X = (1 + X) / 2.0
            X = X.astype(int)

        print('accessing', j, manyruns_path)
        X_multi[j, :, :] = X[0:nn, 0:kk]

    # UMAP aligned needs a relationdict for the 'time varying' dataset
    # our relation is that each traj maps to itself (in time) -- constant relation

    constant_dict = {i: i for i in range(kk)}
    constant_relations = [constant_dict for i in range(len(gamma_list)-1)]

    #X_multi_as_list = [X_multi[i,:,:] for i in range(X_multi.shape[0])]
    print('Starting AlignedUMAP()...')
    #X_multi = tuple([X_multi[i,:,:] for i in range(len(gamma_list))])
    #print(type(X_multi[0]))
    aligned_mapper = umap.AlignedUMAP().fit(X_multi, relations=constant_relations)

    num_rows = 4
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 20))
    ax_bound = axis_bounds(np.vstack(aligned_mapper.embeddings_))
    for i, ax in enumerate(axs.flatten()):
        if i<len(gamma_list):
            print(i)
            #current_target = ordered_target[150 * i:min(ordered_target.shape[0], 150 * i + 400)]
            ax.scatter(*aligned_mapper.embeddings_[i].T, s=2, cmap="Spectral")
            ax.scatter(*aligned_mapper.embeddings_[i].T, s=2, cmap="Spectral_r")
            ax.axis(ax_bound)
            ax.set(xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig('aligned_%d_%d_gammas%d.jpg' % (nn, kk, len(gamma_list)), dpi=300)
