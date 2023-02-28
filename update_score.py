import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
from molecules.properties import docking, docking_gpx4, tpsa
from molecules.conversion import (
    mols_from_smiles, mol_to_smiles, mol_from_smiles, mols_to_smiles)

def add_property(dataset, name, n_jobs):
    # fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr}[name]
    fn = {"CA9": docking, "tpsa": tpsa, "GPX4": docking_gpx4}[name]
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=0)
    prop = pjob(delayed(fn)(mol) for mol in mols)
    new_data = pd.DataFrame(prop, columns=[name])
    return pd.concat([dataset, new_data], axis=1, sort=False)

if __name__ == "__main__":
    #df1 = pd.read_csv('train.smi')
    df2 = pd.read_csv('train_old_score.smi')
    #df2['score'] = [0] * df2.shape[0]
    #mols = []

    #for index, row in df2.iterrows():
        #exists = df1.loc[df1['smiles'] == row['smiles']]
        #if exists.shape[0] != 0:
            #df2.loc[index, 'score'] = exists.iloc[0]['score']
        #else:
            #score = docking(mol_from_smiles(row['smiles']))
            #df2.loc[index, 'score'] = score

    # df2 = df2.rename(columns={'score': 'CA9'})
    #df2 = df2.iloc[227945:]
    df2.drop('score', axis=1, inplace=True)
    #df2 = df2.reset_index(drop=True)
    df2 = add_property(df2, 'CA9', 16)  # Utilizing 8 CPUs, adjust accordingly
    #df = pd.concat([df1, df2], axis=0)
    #df = df.reset_index(drop=True)
    #df2.drop('fragments', axis=1, inplace=True)
    #df2.drop('n_fragments', axis=1, inplace=True)
    df2.drop(df2.columns[[0]], axis=1, inplace=True)
    df2.to_csv('train_ca9.smi')
    print(df2)




