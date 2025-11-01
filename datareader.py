import numpy as np
import pandas as pd
from ast import literal_eval

"""
Single genes are removed from the genome in the desired way.
There are 1968 genes, so there are 1969 rows.
Each row is a gene with a -1 indicating the gene is not present and a 1 indicating it is present.
The response is a 1 if the genome grows and a 0 if it does not. 
Returns a 1969 * 1968 matrix and a 1969 binary response vector. 
"""
def read_ko(intergenic, DATA = "./data"):
    if intergenic:
        df = pd.read_csv(DATA + 'singles/full_essential.csv')
        index_id = df['0-index'].tolist()
        index_id = [ i + 1 for i in index_id ]
    else:
        df = pd.read_csv(DATA + 'singles/essential_entergenic.csv')
        index_id = df['Oralgen Gene ID'].tolist()
        
    singletons = np.ones((n_genes + 1, n_genes), dtype=np.int64)
    singletons[1:] -= 2 * np.eye(n_genes, dtype=np.int64)
    response = np.ones(n_genes + 1)
    response[index_id] = 0
    return singletons, response

"""
Entire operons are downregulated in the genome.
Since some operons are single genes, we ignore those since the true list of knockouts is present in the matrix above.
Downregulated operons with multiple genes are treated as an entire operon knockout, with genes in the operon having a -1 in the row's.
Response is binary with 1 if the genome grows and a 0 if not.
Returns a 368 x 1968 matrix and a 368 binary response vector. 
"""
def read_kd(intergenic, essential_singles=[], DATA = "./data"):
    essential_singles = set(essential_singles)
    
    df = pd.read_csv(DATA + 'opmod/opmod_growth.csv')
    df['Operon Map'] = [literal_eval(df['Operon Map'][i]) for i in range(len(df))]
    df = df[[len(mapping) > 1 for mapping in df['Operon Map']]]
    df = df[df['Payload_Name'].str.contains('KD')].reset_index(drop=True)

    kd_df = df[(df['colony_growth'] != 'Y') | (df['liquid_growth'] != 'Y')]
    kd_df.to_csv(DATA + 'opmod/knockdown_multiple.csv', index=False)
    oralgen_id = kd_df['Operon Map'].tolist()
    masks = df['Operon Map'].tolist()
    
    genomes = np.ones((len(df), n_genes), dtype=np.int64)
    response = .75 * np.ones(len(df))
    response[kd_df.index] = .25
    for i, mask in enumerate(masks):
        mask = [j-1 for j in mask]
        genomes[i][mask] = -1
        response[i] -= .25 * (1 - np.pow(2., -len(set(mask).intersection(essential_singles))))
        pass

    return genomes, response
