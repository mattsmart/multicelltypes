import csv
import mygene
import os
import pandas as pd

from singlecell.singlecell_constants import DATADIR
from singlecell.singlecell_simsetup import singlecell_simsetup

"""
use mygene package https://pypi.org/project/mygene/
allows for synonym search in gene names to answer questions such as:
'Does the single cell dataset contain targets of miR-21?'
"""


TAXID_MOUSE = 10090
MIR_21_TARGETS = ['mir21_misc', 'mir21_wiki', 'mir21_targetscan']
MOUSE_TFS = ['mouse_TF']


def print_simsetup_labels(simsetup):
    print('Genes:')
    for idx, label in enumerate(simsetup['GENE_LABELS']):
        print(idx, label)
    print('Celltypes:')
    for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
        print(idx, label)


def read_gene_list_csv(csvpath, aliases=False):
    with open(csvpath, 'r') as f:
        reader = csv.reader(f)
        if aliases:
            data = {r[0]: [val for val in r[1:] if val] for r in reader}
        else:
            data = [r[0] for r in reader]
    return data


def read_incomplete_gene_list_csv(csvpath, na='-', to_none=True):
    with open(csvpath, 'r') as f:
        reader = csv.reader(f)
        if to_none:
            data = [[r[0], None] if r[1] == na else [r[0], r[1]] for r in reader]
        else:
            data = [[r[0], r[1]] for r in reader]
    return data


def get_mygene_hits(gene_symbol, taxid=TAXID_MOUSE):
    mg = mygene.MyGeneInfo()
    query = mg.query(gene_symbol)
    # print query
    # print query.keys()
    hits = []
    for hit in query['hits']:
        if hit['taxid'] == taxid:
            #print hit
            hits.append(hit)
    return hits


def collect_mygene_hits(gene_symbol_list, taxid=TAXID_MOUSE, entrez_compress=True):
    """
    Returns: two dicts: gene_hits, hitcounts
    gene_hits is dictionary of form:
        {gene_symbol:
                [{hit_1}, {hit_2}, ..., {hit_N}]}
    if entrez_compress:
        {gene_symbol:
                [entrez_id_1, ..., entrez_id_P]} where P < N is number of unique entrez ids
    hitcounts is of the form
    """
    gene_hits = {}
    hitcounts = {}

    # if gene_list arg is path, load the list it contains
    if isinstance(gene_symbol_list, str):
        if os.path.exists(gene_symbol_list):
            print("loading data from %s" % gene_symbol_list)
            gene_symbol_list = read_gene_list_csv(gene_symbol_list)

    print("Searching through %d genes..." % len(gene_symbol_list))
    for idx, g in enumerate(gene_symbol_list):
        hits = get_mygene_hits(g, taxid=taxid)
        if entrez_compress:
            hits = [int(h.get('entrezgene', 0)) for h in hits]
            hits = [h for h in hits if h > 0]
            if len(hits) > 1:
                print("WARNING len(hits)=%d > 1 for %s (%d)" % (len(hits), g, idx))
            if len(hits) == 0:
                print("WARNING len(hits)=%d for %s (%d)" % (len(hits), g, idx))
        gene_hits[g] = hits

        if len(hits) in list(hitcounts.keys()):
            hitcounts[len(hits)][g] = hits
        else:
            hitcounts[len(hits)] = {g: hits}
        if idx % 100 == 0:
            print("Progress: %d of %d" % (idx, len(gene_symbol_list)))

    # print some stats
    for count in range(max(hitcounts.keys())+1):
        if count in list(hitcounts.keys()):
            print("Found %d with %d hits" % (len(list(hitcounts[count].keys())), count))
        else:
            print("Found 0 with %d hits" % count)
    return gene_hits, hitcounts


def write_genelist_id_csv(gene_list, gene_hits, outpath='genelist_id.csv'):
    with open(outpath, 'w') as fcsv:
        for idx, gene in enumerate(gene_list):
            print([gene], gene_hits[gene])
            info_list = [gene] + gene_hits[gene]
            fcsv.write(','.join(str(s) for s in info_list) + '\n')
    return outpath


def check_target_in_gene_id_dict(memories_genes_id, target_genes_id, outpath=None):
    """
    Returns:
        list of tuples (mem_symbol, target_symbol) if they are aliases
    """
    matches = []
    for target_key, target_val in target_genes_id.items():
        for mem_key, mem_val in memories_genes_id.items():
            #print target_key, target_val, mem_key, mem_val
            if target_key == mem_key:
                matches.append((mem_key, target_key))
            else:
                for target_id in target_val:
                    if target_id in mem_val:
                        matches.append((mem_key, target_key))
    if outpath is not None:
        with open(outpath, 'w') as f:
            lines = ['%s\n' % pair[0] for pair in matches]
            lines[-1] = lines[-1].strip()
            f.writelines(lines)
    return matches


if __name__ == '__main__':
    pipeline = 'mir_21'
    assert pipeline in ['TF', 'mir_21']
    process_mouse_TFs = False
    write_memories_id = False
    write_targets_id = False
    find_matches = True

    # process mouse TFs: convert original into symbol + entrez_id csv, fill in missing entrez_ids
    if process_mouse_TFs:
        # convert raw file into incomplete entrez id file
        mouse_dir = DATADIR + os.sep + 'misc' + os.sep + 'mouse_TF_list'
        mouse_tf_path = mouse_dir + os.sep + 'Mus_musculus_TF_union.txt'
        mouse_tf_id_csv_inomplete_path = mouse_dir + os.sep + 'entrez_id_mouse_TF_incomplete.csv'
        mouse_tf_id_csv_complete_path = mouse_dir + os.sep + 'entrez_id_mouse_TF.csv'
        df = pd.read_csv(mouse_tf_path, sep='\t')
        df.to_csv(mouse_tf_id_csv_inomplete_path, columns=['Symbol', 'Entrez ID'], sep=',', header=False, index=False)
        # now complete the entrez id csv file by filling in missing ones
        mouse_TF_incomplete = read_incomplete_gene_list_csv(mouse_tf_id_csv_inomplete_path, to_none=True)
        gene_list = [row[0] for row in mouse_TF_incomplete]
        gene_hits = {row[0]: [row[1]] for row in mouse_TF_incomplete if row[1] is not None}
        missing_genes = [row[0] for row in mouse_TF_incomplete if row[1] is None]
        print("Number of missing entrez IDs in the mouse TF list: %d" % len(missing_genes))
        gene_hits_missing, hitcounts = collect_mygene_hits(missing_genes)
        gene_hits.update(gene_hits_missing)
        write_genelist_id_csv(gene_list, gene_hits, outpath=mouse_tf_id_csv_complete_path)

    if write_memories_id:
        npzpath = DATADIR + os.sep + 'memories' + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_TFonly.npz'
        simsetup = singlecell_simsetup(npzpath=npzpath)
        memories_genes = simsetup['GENE_LABELS']
        memories_genes_lowercase = [g.lower() for g in memories_genes]
        memories_genes_id, memories_hitcounts = collect_mygene_hits(memories_genes)
        write_genelist_id_csv(memories_genes, memories_genes_id, outpath='entrez_id_2018scMCA_TFonly.csv')
    else:
        path_to_compare_targets_to = DATADIR + os.sep + 'misc' + os.sep + 'genelist_entrezids' + os.sep + 'entrez_id_2018scMCA_pruned_TFonly.csv'
        memories_genes_id = read_gene_list_csv(path_to_compare_targets_to, aliases=True)

    # prep target csv
    targetgenes_id_dir = DATADIR + os.sep + 'misc' + os.sep + 'genelist_entrezids'
    if pipeline == 'TF':
        target_names = MOUSE_TFS
    else:
        assert pipeline == 'mir_21'
        target_names = MIR_21_TARGETS

    # write target csv to compare gene list to target database
    if write_targets_id:
        assert pipeline == 'mir_21'
        targetgenes_dir = DATADIR + os.sep + 'misc' + os.sep + 'mir21_targets'
        target_dict = {name: {'gene_path': targetgenes_dir + os.sep + '%s.csv' % name} for name in target_names}
        for name in target_names:
            genes = read_gene_list_csv(target_dict[name]['gene_path'])
            target_dict[name]['genes'] = genes
            gene_hits, hitcounts = collect_mygene_hits(genes)
            write_genelist_id_csv(genes, gene_hits, outpath=targetgenes_dir + os.sep + 'entrez_id_%s.csv' % name)

    # write target csv to compare gene list to target database
    if find_matches:
        for name in target_names:
            target_genes_id = read_gene_list_csv(targetgenes_id_dir + os.sep + 'entrez_id_%s.csv' % name, aliases=True)
            # read target csv to compare gene list to target database
            matches = check_target_in_gene_id_dict(memories_genes_id, target_genes_id, outpath='genes_to_keep_%s.txt' % name)
            print("MATCHES for %s" % name)
            for idx, match in enumerate(matches):
                print(match, memories_genes_id[match[0]], target_genes_id[match[1]])
