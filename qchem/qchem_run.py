import pandas as pd

from .qchem import smi2density
from .qchem_setup import setup


def qchem_run(params=None):
    params = setup(params)
    data_df = pd.read_csv(params.data_csv)
    for smiles, idc in zip(data_df['smiles'].values, data_df['idc'].values):
        smi2density(smiles, idc, params)


if __name__ == '__main__':
    qchem_run()
