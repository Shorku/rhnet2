import re
import math
import zipfile

import numpy as np

from pathlib import Path
from io import TextIOWrapper


def matrices_from_orca(out_file):
    supported_orca_versions = \
        ['2.9.1', '5.0.0', '5.0.1', '5.0.2', '5.0.3', '5.0.4']
    nbas_tot = 0
    natoms = 0
    basis_pattern = \
        (r'Group +1 +Type +[A-Z][a-z]? +: '
         r'\d+s\d+p\d*[d]?\d*[f]? contracted to '
         r'(\d+)s(\d+)p(\d*)[d]?(\d*)[f]? pattern')
    shells = [1, 3, 5, 7]

    with zipfile.ZipFile(out_file) as z:
        with TextIOWrapper(z.open(Path(out_file).stem + '.out'),
                           encoding='utf-8') as f:
            for line in f:
                if 'Program Version' in line:
                    orca_version = line.split()[2]
                    if orca_version not in supported_orca_versions:
                        raise NotImplementedError(f'Orca version '
                                                  f'{orca_version} outputs '
                                                  f'are not supported')
                if line == 'BASIS SET INFORMATION\n':
                    for i in range(4):
                        line = f.__next__()
                    m = re.search(basis_pattern, line)
                    for i in range(len(shells)):
                        shells[i] = \
                            int(m.groups()[i]) * shells[i] \
                            if m.groups()[i] else 0
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
    return dm, ovrlp, nbas_tot, natoms


def blocks_from_orca(out_file, overlap_thresh):
    dm, ovrlp, nbas_tot, natoms = matrices_from_orca(out_file)

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

    return (diagonal_densities, off_diagonal_densities, off_diagonal_overlaps,
            adjacency_atom2link_sources, adjacency_atom2link_targets,
            adjacency_link2atom_sources, adjacency_link2atom_targets,
            natoms, nlinks, nbas)
