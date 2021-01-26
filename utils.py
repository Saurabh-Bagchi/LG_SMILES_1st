import os

import cv2
import detectron2
import matplotlib.pyplot as plt
from matplotlib import patches
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw


class color:
    """
    Colors for printing.
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def show_mol(smiles):
    """
    Plot molecule structure.
    :param smiles: SMILES string. [str]
    :return:
    """
    mol = Chem.MolFromSmiles(smiles)
    imageB = Draw.MolToImage(mol)
    plt.imshow(imageB)
    plt.axis('off')
    plt.show()
    return


def compare_predicition_with_test(smiles, file_name):
    """
    Plot prediction and original structure image side
    by size for comparison.
    :param smiles: SMILES string. [str]
    :param file_name: Name of the image. assumed to be in './data/images/test/'. [str].
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    assert os.path.exists('./data/images/test'), f'Test folder do not exist("./data/images/test")'
    imageA = cv2.imread(f'./data/images/test/{file_name}')

    # get MOL from smiles (not the one in the file)
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        print("INVALID MOL")
        ax1.imshow(imageA)
        plt.title(file_name, fontsize=10)
        plt.show()
        return

    imageB = Draw.MolToImage(mol)
    plt.title(f'{file_name}: {smiles}', fontsize=8)
    ax1.imshow(imageA)
    ax2.imshow(imageB)
    ax1.title.set_text('Label')
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    return


def filter_per_instance_class(outputs, instance_id):
    """ HELPER FUNCTION - DONT CALL DIRECTLY
    Filter label types to plot bboxes.
    :param outputs: outputs of the model. [dict]
    :param instance_id: id of the label. [int]
    """
    if instance_id is None:
        return outputs['instances']
    mask = outputs['instances'].pred_classes == instance_id
    obj = detectron2.structures.Instances(image_size=outputs['instances'].image_size)
    obj.set('pred_boxes', outputs['instances'].pred_boxes[mask])
    obj.set('pred_classes', outputs['instances'].pred_classes[mask])
    obj.set('scores', outputs['instances'].scores[mask])
    return obj


def get_similarity(smiles_1, smiles_2):
    """
    Fingerprint similarity
    :param smiles_1: SMILES string. [str]
    :param smiles_2: SMILES string. [str]
    :return:
    """
    mol_1 = Chem.MolFromSmiles(smiles_1)
    mol_2 = Chem.MolFromSmiles(smiles_2)
    fps = [Chem.RDKFingerprint(mol) for mol in [mol_1, mol_2]]
    return DataStructs.FingerprintSimilarity(fps[0], fps[1])


def _plot_bbox_inside_atoms(cpos, points, chosen, margin, file_name):
    """ HELPER FUNCTION, DONT CALL DIRECTLY
    Helper function to visualize cases where there is more than
    2 atoms inside a bond bounding box.
    :param cpos: DF (LU, RD, LD, RU)
    :param points: all atoms (Type, X, Y)
    :param chosen:  chosen atoms (Type, X, Y)
    :param margin: margin taken from real corners
    :param file_name: image file name.
    :return:
    """
    fig, ax = plt.subplots()

    img = plt.imread(f"./data/images/test/{file_name}")
    ax.imshow(img)

    cpos[0] -= margin
    cpos[1] += margin

    W = cpos[1][0] - cpos[0][0]
    H = cpos[1][1] - cpos[0][1]
    ax.scatter(chosen['x'], chosen['y'].values, c='r', s=100)

    for i, (idx, row) in enumerate(points.iterrows()):
        ax.text(row['x'],
                row['y'] + 5,
                str(i),
                fontsize=20,
                color='white',
                bbox=dict(facecolor='blue', alpha=0.5))

    ax.add_patch(
            patches.Rectangle(
                    cpos[0],
                    W,
                    H,
                    edgecolor='red',
                    facecolor='red',
                    fill=False
            ))

    plt.gca().invert_xaxis()
    plt.title(file_name)
    plt.xlim(cpos[0][0] - margin,
             cpos[1][0] + margin)
    plt.ylim(cpos[0][1] - margin,
             cpos[1][1] + margin)
    plt.show()
    return
