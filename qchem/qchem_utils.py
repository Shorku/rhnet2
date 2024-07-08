import os
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from zipfile import ZipFile, ZIP_LZMA


def smiles_to_orca(smi, conf_thresh, nconf, opt_thresh):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    param = Chem.rdDistGeom.ETKDGv2()
    param.pruneRmsThresh = conf_thresh
    param.enforceChirality = True
    rdDistGeom.EmbedMultipleConfs(mol, nconf, param)
    conv = AllChem.MMFFOptimizeMoleculeConfs(mol)
    e_mmff = [i[1] for i in conv]
    e_min = min(e_mmff)
    filtered_confs = []
    for i in range(Chem.rdchem.Mol.GetNumConformers(mol)):
        if e_mmff[i] - e_min < opt_thresh:
            filtered_confs.append(Chem.rdmolfiles.MolToXYZBlock(mol, i))
    return filtered_confs


def orca_dft_inps(cas, confs, dft_inp, work_dir):
    with open(dft_inp) as f:
        template = f.read()
    inputs = []
    for i, conf in enumerate(confs):
        geom = conf.split('\n')[2:]
        geom = '\n'.join(geom)
        inp = template.format(geom)
        with open(os.path.join(work_dir, f'{cas}_{i}_dft.inp'), 'w') as f:
            f.write(inp)
        inputs.append(f'{cas}_{i}_dft')
    return inputs


def orca_runner(inputs, work_dir, orca_path):
    cwd = os.getcwd()
    os.chdir(work_dir)
    for inp in inputs:
        with open(f'{inp}.out', 'w') as out:
            subprocess.run([orca_path, f'{inp}.inp'], stdout=out)
    os.chdir(cwd)


def orca_projection_inps(cas, confs, inputs, basis, dmt_inp, work_dir):
    with open(dmt_inp) as f:
        template = f.read()
    with open(basis) as f:
        basis = f.read()
    elements = set()
    for line in confs[0].split('\n')[2:]:
        if len(line) > 0:
            elements.add(line.strip().split()[0])
    basis_block = ''
    for element in elements:
        basis_block += f'NewGTO {element}\n{basis}end\n'
    inputs_projection = []
    for i, conf in enumerate(confs):
        geom = conf.split('\n')[2:]
        geom = '\n'.join(geom)
        inp = template.format(inputs[i], basis_block, geom)
        with open(os.path.join(work_dir, f'{cas}_{i}.inp'), 'w') as f:
            f.write(inp)
        inputs_projection.append(f'{cas}_{i}')
    return inputs_projection


def orca_cleanup(inputs_dft, inputs_dmt, work_dir):
    cwd = os.getcwd()
    os.chdir(work_dir)
    for input in inputs_dft:
        with ZipFile(f'{input}.zip', 'w', compression=ZIP_LZMA) as zf:
            zf.write(f'{input}.gbw')
            zf.write(f'{input}.inp')
            zf.write(f'{input}.out')
        for ext in ['inp', 'prop', 'gbw', 'out']:
            os.remove(f'{input}.{ext}')
    for input in inputs_dmt:
        with ZipFile(f'{input}.zip', 'w', compression=ZIP_LZMA) as zf:
            zf.write(f'{input}.out')
        for ext in ['inp', 'prop', 'gbw', 'out']:
            os.remove(f'{input}.{ext}')
    os.chdir(cwd)


def smi2density(smi, cas, options):
    confs = smiles_to_orca(smi, options.conf_thresh,
                           options.nconf, options.opt_thresh)
    inputs_dft = orca_dft_inps(cas, confs, options.dft_inp, options.work_dir)
    orca_runner(inputs_dft, options.work_dir, options.orca_path)
    inputs_dmt = orca_projection_inps(cas, confs, inputs_dft, options.basis,
                                      options.dmt_inp, options.work_dir)
    orca_runner(inputs_dmt, options.work_dir, options.orca_path)
    orca_cleanup(inputs_dft, inputs_dmt, options.work_dir)
