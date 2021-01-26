import json
import multiprocessing
from collections import Counter, defaultdict
from xml.dom import minidom

import numpy as np
import pandas as pd
from detectron2.structures import BoxMode
from pqdm.processes import pqdm
from rdkit.Chem import Draw
from scipy.spatial.ckdtree import cKDTree

from utils import *


def _get_unique_atom_smiles_and_rarity(smiles):
    """ HELPER FUNCTION - DONT CALL DIRECTLY
    Get the compound unique atom smiles in the format [AtomType+FormalCharge] and a dictionary
    of the metrics taken into account for rarity measures.
    eg: OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N ---> {'C0', 'N1', 'N0', 'O0', 'S0'}
    :param smiles: SMILES. (string)
    :return: set of atom smiles(strings).
    """

    mol = Chem.MolFromSmiles(smiles)
    assert mol, f'INVALID SMILES STRING: {smiles}'

    doc = _get_svg_doc(mol)

    # get atom positions in order to oversample hard cases
    atoms_pos = np.array([[int(round(float(path.getAttribute('drawing-x')), 0)),
                           int(round(float(path.getAttribute('drawing-y')), 0))] for path in
                          doc.getElementsByTagName('rdkit:atom')])

    # calculat the minimum distance between atoms in the molecule
    sampling_weights = {}
    xys = atoms_pos
    kdt = cKDTree(xys)
    dists, neighbours = kdt.query(xys, k=2)
    nearest_dist = dists[:, 1]

    # min distance
    sampling_weights['global_minimum_dist'] = 1 / (np.min(nearest_dist) + 1e-12)
    # number of atoms closer than half of the average distance
    sampling_weights['n_close_atoms'] = np.sum(nearest_dist < np.mean(nearest_dist) * 0.5)
    # average atom degree
    sampling_weights['average_degree'] = np.array([a.GetDegree() for a in mol.GetAtoms()]).mean()
    # number of triple bonds
    sampling_weights['triple_bonds'] = sum([1 for b in mol.GetBonds() if b.GetBondType().name == 'TRIPLE'])

    return [''.join([a.GetSymbol(), str(a.GetFormalCharge())]) for a in mol.GetAtoms()], sampling_weights


def _get_svg_doc(mol):
    """
    Draws molecule a generates SVG string.
    :param mol:
    :return:
    """
    dm = Draw.PrepareMolForDrawing(mol)
    d2d = Draw.MolDraw2DSVG(300, 300)
    d2d.DrawMolecule(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    doc = minidom.parseString(svg)
    return doc


def create_unique_ins_labels(data, overwrite=False, base_path='.'):
    """
    Create a dictionary with the count of each existent atom-smiles in the
    train dataset and a dataframe with the atom-smiles in each compound.
    eg: SMILES dataframe

    :param data: Pandas data frame with columns ['file_name', 'SMILES']. [Pandas DF]
    :param overwrite: overwrite existing JSON file at base_path + '/data/unique_atoms_smiles.json. [bool]
    :param base_path: base path of the environment. [str]
    :return: A dict of counts[dict] and DataFrame of unique atom-smiles per compound.
    """
    smiles_list = data.SMILES.to_list()

    # check if file exists
    output_counts_path = base_path + '/data/unique_atom_smiles_counts.json'
    output_unique_atoms = base_path + '/data/unique_atoms_per_molecule.csv'
    output_mol_rarity = base_path + '/data/mol_rarity_train.csv'

    if all([os.path.exists(p) for p in [output_counts_path, output_unique_atoms]]):
        if overwrite:
            print(f'{color.BLUE}Output files exists, but overwriting.{color.BLUE}')
        else:
            print(f'{color.BOLD}labels JSON {color.END} already exists, skipping process and reading file.\n',
                  f'{color.BLUE}Counts file readed from:{color.END} {output_counts_path}\n',
                  f'{color.BLUE}Unique atoms file readed from:{color.END} {output_unique_atoms}\n'
                  f'{color.BLUE}Mol rarity file readed from:{color.END} {output_counts_path}\n',
                  f'if you want to {color.BOLD} overwrite previous file {color.END}, '
                  f'call function with {color.BOLD}overwrite=True{color.END}')

            return json.load(open(output_counts_path, 'r')), \
                   pd.read_csv(output_unique_atoms)

    assert type(smiles_list) == list, 'Input smiles data type must be a LIST'

    n_jobs = multiprocessing.cpu_count() - 1

    # get unique atom-smiles in each compound and count for sampling later.
    result = pqdm(smiles_list, _get_unique_atom_smiles_and_rarity,
                  n_jobs=n_jobs, desc='Calculating unique atom-smiles and rarity')
    result, sample_weights = list(map(list, zip(*result)))
    counts = Counter(x for xs in result for x in xs)

    # save counts
    with open(output_counts_path, 'w') as fout:
        json.dump(counts, fout)

    # save sample weights
    sample_weights = pd.DataFrame.from_dict(sample_weights)
    sample_weights.insert(0, "file_name", data.file_name)
    sample_weights.to_csv(output_mol_rarity, index=False)

    # save unique atoms in each molecule to oversample less represented classes later
    unique_atoms_per_molecule = pd.DataFrame({'SMILES': smiles_list, 'unique_atoms': [set(r) for r in result]})
    unique_atoms_per_molecule.to_csv(output_unique_atoms, index=False)

    print(f'{color.BLUE}Counts file saved at:{color.END} {output_counts_path}\n' +
          f'{color.BLUE}Unique atoms file saved at:{color.END} {output_unique_atoms}')

    return counts, unique_atoms_per_molecule


def sample_balanced_datasets(data, counts, unique_atoms_per_molecule, datapoints_per_label=2000):
    """
    Construct a balanced dataset by sampling every label uniformly.
    Returns train and val data [Pandas DF].

    :param data: DataFrame with SMILES data. [Pandas DF]
    :param counts: Count of each label in the dataset. [dict]
    :param unique_atoms_per_molecule: DataFrame with unique atom-smiles[str] in each compound. [set]
    :param datapoints_per_label: Molecules to sample per label. [int]
    :return: Balanced train and val dataset. [Pandas DF]
    """

    # merge data with the respective set of unique atoms contained.
    data = pd.merge(data, unique_atoms_per_molecule, left_on='SMILES', right_on='SMILES')

    # create DF to save balanced train data
    balanced_train_data = pd.DataFrame(data=None, columns=data.columns)
    balanced_val_data = pd.DataFrame(data=None, columns=data.columns)

    # sample datapoints per unique label type and append to datasets
    print(f'{color.BLUE}Sampling {datapoints_per_label} points per label type{color.END}')

    for k in counts.keys():

        if k == 'N1':
            sampled_train_data = data[data.unique_atoms.apply(lambda x: k in x)].sample(5 * datapoints_per_label,
                                                                                        replace=True)
        else:
            sampled_train_data = data[data.unique_atoms.apply(lambda x: k in x)].sample(datapoints_per_label,
                                                                                        replace=True)
        sampled_val_data = data[data.unique_atoms.apply(lambda x: k in x)].sample(datapoints_per_label // 100,
                                                                                  replace=True)

        balanced_train_data = balanced_train_data.append(sampled_train_data)
        balanced_val_data = balanced_val_data.append(sampled_val_data)

    balanced_train_data.drop('unique_atoms', axis=1, inplace=True)
    balanced_val_data.drop('unique_atoms', axis=1, inplace=True)

    return balanced_train_data, balanced_val_data


def sample_images(mol_weights, n=10000):
    """
     Sample compounds depending on complexity.
    :param mol_weights: DataFrame with img_n
    :param n: number of molecules to sample[int]
    :return: Sampled dataset. [Pandas DF]
    """
    img_names_sampled = pd.DataFrame.sample(mol_weights, n=n, weights=mol_weights, replace=True)
    return img_names_sampled.index.to_list()


def get_mol_sample_weight(data, data_mode='train', p=1000, base_path='.'):
    """
    Creating sampling weights to oversample hard cases based on bond, atoms, overlaps and rings.
    :param data: DataFrame with train data(SMILES). [Pandas DF]
    :param data_mode: Train or val. [str]
    :param p: Rarity weight. [int]
    :param base_path: base path of the environment. [str]
    :return:
    """
    # load rarity file
    mol_rarity_path = base_path + f'/data/mol_rarity_{data_mode}.csv'
    assert os.path.exists(mol_rarity_path), 'No mol_rarity.csv. Create first and then call function'
    mol_rarity = pd.read_csv(mol_rarity_path)

    # filter by given list, calculate normalized weight value per image
    mol_rarity = pd.merge(mol_rarity, data, left_on='file_name', right_on='file_name')
    mol_rarity.drop(['SMILES'], axis=1, inplace=True)
    mol_rarity.set_index('file_name', inplace=True)

    # sort each column, after filtering, then assign weight values
    for column in mol_rarity.columns:
        mol_rarity_col = mol_rarity[column].values.astype(np.float64)
        mol_rarity_col_sort_idx = np.argsort(mol_rarity_col)
        ranking_values = np.linspace(1.0 / len(mol_rarity_col), 1.0, num=len(mol_rarity_col))
        ranking_values = ranking_values ** p
        mol_rarity_col[mol_rarity_col_sort_idx] = ranking_values
        mol_rarity[column] = mol_rarity_col
    # normalized weights per img
    mol_weights = pd.DataFrame.sum(mol_rarity, axis=1)
    mol_weights /= pd.DataFrame.sum(mol_weights, axis=0) + 1e-12
    return mol_weights


def get_bbox(smiles, unique_labels, atom_margin=12, bond_margin=10):
    """
    Get list of dics with atom-smiles and bounding box [x, y, width, height].
    :param smiles: STR
    :param unique_labels: dic with labels and idx for training.
    :param atom_margin: margin for bbox of atoms.
    :param bond_margin: margin for bbox of bonds.

    :return:
    """
    # replace unique labels to decide with kind of labels to look for
    labels = defaultdict(int)
    for k, v in unique_labels.items():
        labels[k] = v

    mol = Chem.MolFromSmiles(smiles)

    doc = _get_svg_doc(mol)

    # Get X and Y from drawing and type is generated
    # from mol Object, concatenating symbol + formal charge
    atoms_data = [{'x':    int(round(float(path.getAttribute('drawing-x')), 0)),
                   'y':    int(round(float(path.getAttribute('drawing-y')), 0)),
                   'type': ''.join([a.GetSymbol(), str(a.GetFormalCharge())])} for path, a in
                  zip(doc.getElementsByTagName('rdkit:atom'), mol.GetAtoms())]

    annotations = []
    # anotating bonds
    for path in doc.getElementsByTagName('rdkit:bond'):

        # Set all '\' or '/' as single bonds
        ins_type = path.getAttribute('bond-smiles')
        if (ins_type == '\\') or (ins_type == '/'):
            ins_type = '-'

        # make bigger margin for bigger bonds (double and triple)
        _margin = bond_margin
        if (ins_type == '=') or (ins_type == '#'):
            _margin *= 1.5

        # creating bbox coordinates as XYWH.
        begin_atom_idx = int(path.getAttribute('begin-atom-idx')) - 1
        end_atom_idx = int(path.getAttribute('end-atom-idx')) - 1
        x = min(atoms_data[begin_atom_idx]['x'], atoms_data[end_atom_idx]['x']) - _margin // 2  # left-most pos
        y = min(atoms_data[begin_atom_idx]['y'], atoms_data[end_atom_idx]['y']) - _margin // 2  # up-most pos
        width = abs(atoms_data[begin_atom_idx]['x'] - atoms_data[end_atom_idx]['x']) + _margin
        height = abs(atoms_data[begin_atom_idx]['y'] - atoms_data[end_atom_idx]['y']) + _margin

        annotation = {'bbox':        [x, y, width, height],
                      'bbox_mode':   BoxMode.XYWH_ABS,
                      'category_id': labels[ins_type]}
        annotations.append(annotation)

    # annotating atoms
    for atom in atoms_data:
        _margin = atom_margin

        # better to predict close carbons (2 close instances affected by NMS)
        if atom['type'] == 'C0':
            _margin /= 2

        # Because of the hydrogens normally the + sign falls out of the box
        if atom['type'] == 'N1':
            _margin *= 2

        annotation = {'bbox':        [atom['x'] - _margin,
                                      atom['y'] - _margin,
                                      _margin * 2,
                                      _margin * 2],
                      'bbox_mode':   BoxMode.XYWH_ABS,
                      'category_id': labels[atom['type']]}
        annotations.append(annotation)

    return annotations


def plot_bbox(smiles, labels):
    """
    Plot bounding boxes for smiles in opencv, close window with any letter in pycharm.

    :param smiles: SMILES string. [str]
    :param labels: Predicted bounding boxes. [dict]
    :return:
    """
    # create mol image and create np array
    mol = Chem.MolFromSmiles(smiles)
    img = np.array(Draw.MolToImage(mol))

    # draw rects
    for ins in get_bbox(smiles, labels):
        ins_type = ins['category_id']
        x, y, width, height = ins['bbox']

        cv2.rectangle(img, (x, y), (x + width, y + height), np.random.rand(3, ), 2)

    cv2.namedWindow(smiles, cv2.WINDOW_NORMAL)
    cv2.imshow(smiles, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def create_COCO_json(smiles, file_name, mode, labels, base_path='.'):
    """
    Create COCO style dataset. If there is not image for the smile
    it creates it.
    :param labels:
    :param smiles: SMILES. [str]
    :param file_name: Name of the image file. [str] eg. 'train_123412.png'
    :param mode: train or val. [str]
    :param labels: dic with labels and idx for training.
    :param base_path: base path of the environment. [str]
    :return:
    """
    if not os.path.exists(base_path + f'/data/images/{mode}/{file_name}'):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Draw.MolToImageFile(mol, base_path + f'/data/images/{mode}/{file_name}')

    return {'file_name':   base_path + f'/data/images/{mode}/{file_name}',
            'height':      300,
            'width':       300,
            'image_id':    file_name,
            'annotations': get_bbox(smiles, labels)}
