import copy

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree


def bbox_to_graph(output, idx_to_labels, bond_labels):
    # calculate atoms mask (pred classes that are atoms/bonds)
    atoms_mask = np.array([True if ins not in bond_labels else False for ins in output['pred_classes']])

    # get atom list
    atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]
    atoms_list = pd.DataFrame({'atom': atoms_list,
                               'x':    output['bbox_centers'][atoms_mask, 0],
                               'y':    output['bbox_centers'][atoms_mask, 1]})

    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            if row.atom[-2] != '-':
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bonds_list = []

    # get bonds
    for bbox, bond_type, score in zip(output['bbox'][np.logical_not(atoms_mask)],
                                      output['pred_classes'][np.logical_not(atoms_mask)],
                                      output['scores'][np.logical_not(atoms_mask)]):

        if idx_to_labels[bond_type] == 'SINGLE':
            _margin = 5
        else:
            _margin = 8

        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], score))

    return atoms_list.atom.values.tolist(), bonds_list


def mol_from_graph(atoms, bonds):
    """ construct RDKIT mol object from atoms, bonds and bond types
    atoms: list of atom symbols+fc. ex: ['C0, 'C0', 'O-1', 'N1']
    bonds: list of lists of the born [atom_idx1, atom_idx2, bond_type, score]
    """

    # create and empty molecular graph to add atoms and bonds
    mol = Chem.RWMol()
    nodes_idx = {}
    bond_types = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
                  'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
                  'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
                  'AROMATIC': Chem.rdchem.BondType.AROMATIC}

    # add nodes
    for idx, node in enumerate(atoms):
        # neutral formal charge
        if ('0' in node) or ('1' in node):
            a = node[:-1]
            fc = int(node[-1])
        if '-1' in node:
            a = node[:-2]
            fc = -1
        # create atom object
        a = Chem.Atom(a)
        a.SetFormalCharge(fc)

        # add atom to molecular graph (return the idx in object)
        atom_idx = mol.AddAtom(a)
        nodes_idx[idx] = atom_idx

    # add bonds
    existing_bonds = set()
    for idx_1, idx_2, bond_type, score in bonds:
        if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
            if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                try:
                    mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                except:
                    continue
        existing_bonds.add((idx_1, idx_2))
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
            # save safe structure
            prev_mol = copy.deepcopy(mol)
        # check if last addition broke the molecule
        else:
            # load last structure
            mol = copy.deepcopy(prev_mol)

    mol = mol.GetMol()
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return Chem.MolToSmiles(mol)
