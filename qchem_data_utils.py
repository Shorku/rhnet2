import io
import os
import re
import math
import pathlib
import zipfile

import numpy as np

from scipy.spatial.transform import Rotation


###############################################################################
# Basic geometry manipulation utilities
###############################################################################
def split_to_atoms(geom: str):
    atoms = []
    coords = []
    for line in geom.strip().split('\n'):
        atom = line.split()[0]
        coord = [float(coord) for coord in line.split()[1:]]
        atoms.append(atom)
        coords.append(coord)

    return atoms, coords


def rotate_geom(atoms: list, coords: list):
    rot_coords = Rotation.random().apply(np.array(coords))
    rot_geom = \
        '\n'.join([atom + ' ' + ' '.join([str(i) for i in list(coord)])
                   for atom, coord in zip(atoms, rot_coords)])

    return rot_geom


###############################################################################
# ORCA outputs parsing utilities
###############################################################################
def orca_out_not_ok(out_file: str):
    if not os.path.isfile(out_file):
        return True
    if out_file.endswith('.zip'):
        z = zipfile.ZipFile(out_file)
        fname = f'{pathlib.Path(out_file).stem}.out'
        if fname not in z.namelist():
            z.close()
            return True
        f = io.TextIOWrapper(z.open(fname), encoding='utf-8')
    else:
        f = open(out_file)
    contents = f.readlines()
    f.close()
    if out_file.endswith('.zip'):
        z.close()
    if '****ORCA TERMINATED NORMALLY****' in contents[-2]:
        return False
    else:
        return True


def read_geom_from_out(out_file: str):
    geom = ''
    if out_file.endswith('.zip'):
        z = zipfile.ZipFile(out_file)
        f = io.TextIOWrapper(z.open(f'{pathlib.Path(out_file).stem}.out'),
                             encoding='utf-8')
    else:
        f = open(out_file)
    for line in f:
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            f.__next__()
            line = f.__next__()
            while len(line) > 1:
                geom += line
                line = f.__next__()
            break
    f.close()
    if out_file.endswith('.zip'):
        z.close()

    return geom.strip()


def dipole_from_orca_out(out_file: str):
    if out_file.endswith('.zip'):
        z = zipfile.ZipFile(out_file)
        f = io.TextIOWrapper(z.open(f'{pathlib.Path(out_file).stem}.out'),
                             encoding='utf-8')
    else:
        f = open(out_file)
    for line in f:
        if 'Magnitude (a.u.)' in line:
            dipole = float(line.split()[-1])
            break
    else:
        raise ValueError('No dipole magnetization')
    f.close()
    if out_file.endswith('.zip'):
        z.close()

    return dipole


def gepol_xyzr(geom: str):
    # from https://doi.org/10.1021/jp8111556
    vwrad = {'O': 1.52,
             'N': 1.55,
             'F': 1.47,
             'S': 1.80,
             'C': 1.70,
             'H': 1.10,
             'Cl': 1.75}
    geom_lines = geom.split("\n")
    new_geom = f'{len(geom_lines):>8}\n'
    for i, line in enumerate(geom_lines):
        new_geom += f'{f"{float(line.split()[1]):.5f}":>10}'
        new_geom += f'{f"{float(line.split()[2]):.5f}":>10}'
        new_geom += f'{f"{float(line.split()[3]):.5f}":>10}'
        new_geom += f'{f"{vwrad[line.split()[0]]:.5f}":>10}'
        new_geom += f'{i + 1:>8} '
        new_geom += f'{line.split()[0]:<2}'
        new_geom += '1       1 AAA   1 MAIN\n'

    return new_geom


def vol_surf_from_geom(geom: str, gepol_path: str):
    temp_inp = os.path.join(gepol_path, 'TEMP.INP')
    temp_out = os.path.join(gepol_path, 'TEMP.OUT')
    temp_xyz = os.path.join(gepol_path, 'TEMP.XYZR')
    with open(temp_inp, 'w') as f:
        f.write('COOF=TEMP.XYZR')
    gepol_geom = gepol_xyzr(geom)
    with open(temp_xyz, 'w') as f:
        f.write(gepol_geom)
    cwd = os.getcwd()
    os.chdir(gepol_path)
    os.system('./GEPOL93 < TEMP.INP > TEMP.OUT')
    os.chdir(cwd)
    with open(temp_out, 'r') as f:
        for line in f:
            if 'Area' in line:
                surf = float(line.split()[-2])
            if 'Volume' in line:
                vol = float(line.split()[-2])

    return vol, surf


def vol_from_orca_out(out_file: str, gepol_path: str = '',
                      dummy: bool = False):
    if dummy:
        return 1.0
    geom = read_geom_from_out(out_file)
    vol = vol_surf_from_geom(geom, gepol_path)[0]

    return vol


def surf_from_orca_out(out_file: str, gepol_path: str = '',
                       dummy: bool = False):
    if dummy:
        return 1.0
    geom = read_geom_from_out(out_file)
    surf = vol_surf_from_geom(geom, gepol_path)[1]

    return surf


def matrices_from_orca(out_file: str, dummy: bool = False):
    if dummy:
        dm = np.array([[1., 1., 1., 1.],
                       [1., 1., 1., 1.],
                       [1., 1., 1., 1.],
                       [1., 1., 1., 1.]])
        ovrlp = np.array([[0.1, 0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.1]])
        atoms = [[0.09], [0.09]]
        coords = [[0., 0., 0.], [0., 0., 1.]]
        nbas_tot = 4
        natoms = 2
        return dm, ovrlp, atoms, coords, nbas_tot, natoms

    supported_orca_versions = \
        ['2.9.1', '5.0.0', '5.0.1', '5.0.2', '5.0.3', '5.0.4']
    charg = {'H':   1.0,
             'C':   6.0,
             'N':   7.0,
             'O':   8.0,
             'F':   9.0,
             'S':  16.0,
             'Cl': 17.0}
    nbas_tot = 0
    natoms = 0
    basis_pattern = \
        (r'Group +1 +Type +[A-Z][a-z]? +: '
         r'\d+s\d+p\d*[d]?\d*[f]? contracted to '
         r'(\d+)s(\d+)p(\d*)[d]?(\d*)[f]? pattern')
    shells = [1, 3, 5, 7]
    geom = ''
    if out_file.endswith('.zip'):
        z = zipfile.ZipFile(out_file)
        f = io.TextIOWrapper(z.open(f'{pathlib.Path(out_file).stem}.out'),
                             encoding='utf-8')
    else:
        f = open(out_file)
    for line in f:
        if 'Program Version' in line:
            orca_version = line.split()[2]
            if orca_version not in supported_orca_versions:
                raise NotImplementedError(f'Orca version {orca_version}'
                                          f' outputs are not supported')
        if line == 'CARTESIAN COORDINATES (ANGSTROEM)\n':
            f.__next__()
            line = f.__next__()
            while len(line) > 1:
                geom += line
                line = f.__next__()
        if line == 'BASIS SET INFORMATION\n':
            for i in range(4):
                line = f.__next__()
            m = re.search(basis_pattern, line)
            for i in range(len(shells)):
                shells[i] = \
                    int(m.groups()[i]) * shells[i] if m.groups()[i] else 0
        if 'Basis Dimension' in line and nbas_tot == 0:
            nbas_tot = int(line.split()[-1])
            dm = np.zeros((nbas_tot, nbas_tot))
            ovrlp = np.zeros((nbas_tot, nbas_tot))
            natoms = nbas_tot // sum(shells)
        if line == 'DENSITY\n':
            f.__next__()
            for block in range(math.ceil(nbas_tot / 6)):
                f.__next__()
                for row in range(nbas_tot):
                    line = f.__next__()
                    cols = len(line.split()) - 1
                    column = 0
                    while column < cols:
                        dm[row, column + 6 * block] = \
                            float(line.split()[column + 1])
                        column += 1
        if line == 'OVERLAP MATRIX\n':
            f.__next__()
            for block in range(math.ceil(nbas_tot / 6)):
                f.__next__()
                for row in range(nbas_tot):
                    line = f.__next__()
                    cols = len(line.split()) - 1
                    column = 0
                    while column < cols:
                        ovrlp[row, column + 6 * block] = \
                            float(line.split()[column + 1])
                        column += 1
    f.close()
    if out_file.endswith('.zip'):
        z.close()
    atoms, coords = split_to_atoms(geom)
    atoms = [[charg[atom] / 100.0] for atom in atoms]

    return dm, ovrlp, atoms, coords, nbas_tot, natoms


def blocks_from_orca(out_file: str, overlap_thresh: float,
                     dummy: bool = False):
    dm, ovrlp, atoms, coords, nbas_tot, natoms = \
        matrices_from_orca(out_file, dummy=dummy)
    nbas = nbas_tot // natoms
    diagonal_densities = []
    off_diagonal_densities = []
    off_diagonal_overlaps = []
    adjacency_atom2link_sources = []
    adjacency_atom2link_targets = []
    adjacency_link2atom_sources = []
    adjacency_link2atom_targets = []
    nlinks = 0
    for atom in range(natoms):
        diagonal_density = \
            np.hstack([dm[atom * nbas + bas: atom * nbas + bas + 1,
                       atom * nbas + bas: (atom + 1) * nbas]
                       for bas in range(nbas)])
        diagonal_densities.append(diagonal_density.flatten().tolist())
        for other_atom in range(natoms):
            if atom == other_atom:
                continue
            off_diagonal_overlap = \
                ovrlp[atom * nbas: (atom + 1) * nbas,
                      other_atom * nbas: (other_atom + 1) * nbas]
            if np.abs(off_diagonal_overlap).max() < overlap_thresh:
                continue
            off_diagonal_overlaps.append(
                off_diagonal_overlap.flatten().tolist())
            adjacency_atom2link_sources.append(atom)
            adjacency_atom2link_targets.append(nlinks)
            adjacency_link2atom_sources.append(nlinks)
            adjacency_link2atom_targets.append(other_atom)
            nlinks += 1
            off_diagonal_density = \
                dm[atom * nbas: (atom + 1) * nbas,
                   other_atom * nbas: (other_atom + 1) * nbas]
            off_diagonal_densities.append(
                off_diagonal_density.flatten().tolist())

    return (diagonal_densities,
            off_diagonal_densities,
            off_diagonal_overlaps,
            adjacency_atom2link_sources,
            adjacency_atom2link_targets,
            adjacency_link2atom_sources,
            adjacency_link2atom_targets,
            atoms, natoms, nlinks, nbas)
