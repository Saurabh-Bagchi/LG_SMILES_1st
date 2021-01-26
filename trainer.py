import glob
import pickle

import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

import inference as inference
from labels_generation import *
from utils import *


class Trainer:
    """
    Main class of the project. Preprocess a list of SMILES and directly
    creates images and bounding box labels for atoms and bonds in the images.
    Trains a Faster RCNN model to predict bounding boxes and classify each of
    them as bonds [SINGLE, DOUBLE, TRIPLE] and atoms in the format
    "Symbol+FormalCharge" ['C0', 'N1', 'O-1',...].
    After box predictions, molecular graphs are constructed directly from the
    atoms and bonds. Atom connectivity is calculated by finding closest atoms
    to the corners of the bonds bounding boxes. Finally, a molecular graph object
    is created and the SMILES string generated. Molecular graphs are sanitized
    before generating SMILES strings.

    :param base_path: Root path of the environment. [str]
    :param min_points_threshold: Minimum number of instances of an atom to be considered in labels. [int]
    :param n_jobs: Number of processes for paralell versions. [int]
    :param overwrite: Override existing label files. [bool]
    :param n_sample_hard: Number of hard cases to sample depending on strucute complexity. [int]
    :param n_sample_per_label: Number of cases to sample per atom label to have a balanced train set. [int]
    :param input_format: "RGB" or "BGR", depeding on the order of color channels of the input data. Important! [str]


    """

    def __init__(self, params):
        self.base_path = params['base_path']
        self.min_points_threshold = params['min_points_threshold']
        self.n_jobs = params['n_jobs']
        self.overwrite = params['overwrite']
        self.n_sample_hard = params['n_sample_hard']
        self.n_sample_per_label = params['n_sample_per_label']
        self.input_format = params['input_format']

        assert os.path.exists(self.base_path + '/data/train.csv'), f"No train CSV file in data folder."
        self.data = pd.read_csv('./data/train.csv')

        self.preprocess()

        # load label and assigned idx
        self.unique_labels = json.load(open(self.base_path + f'/data/labels.json', 'r'))
        self.unique_labels['other'] = 0
        self.labels = list(self.unique_labels.keys())
        self.labels.insert(0, self.labels.pop())  # need "other" first in the list

        # idx to labels for inference
        self.bond_labels = [self.unique_labels[b] for b in ['-', '=', '#']]
        self.idx_to_labels = {v: k for k, v in self.unique_labels.items()}
        for l, b in zip(self.bond_labels, ['SINGLE', 'DOUBLE', 'TRIPLE']):
            self.idx_to_labels[l] = b

        # preparing datasets for training
        for mode in ["train", "val"]:
            dataset_name = f"smilesdetect_{mode}"
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
            DatasetCatalog.register(dataset_name, lambda mode=mode: self.get_metadata(mode))
            MetadataCatalog.get(dataset_name).set(thing_classes=self.labels)
        self.smiles_metadata = MetadataCatalog.get("smilesdetect_val")

        self.cfg = self.create_cfg()
        self.predictor = None

        self.inference_metadata = MetadataCatalog.get("smilesdetect_val")

    def preprocess(self):
        """
        Creates COCO-style object annotations directly from a list of SMILES strings. If images are not
        the data folder, they are created automatically.
        :return:
        """
        assert os.path.exists(self.base_path + '/data/train.csv'), f"No train CSV file in data folder."

        if not all([os.path.exists(self.base_path + f'/data/annotations_{mode}.pkl') for mode in ['train', 'val']]):
            print(f"{color.BLUE}Creating COCO-style annotations for both sampled datasets [train, val]{color.BLUE}")

            # Get counts and unique atoms per molecules to construct datasets.
            counts, unique_atoms_per_molecule = create_unique_ins_labels(self.data,
                                                                         overwrite=self.overwrite,
                                                                         base_path=self.base_path)

            # bonds SMARTS
            unique_bonds = ['-', '=', '#']

            # Choose labels depending on a minimum count.
            counts = {k: v for k, v in counts.items() if v > self.min_points_threshold}
            labels = list(counts.keys()) + unique_bonds
            unique_labels = {u: idx + 1 for idx, u in zip(range(len(labels)), labels)}

            # Sample uniform datasets among labels
            train_balanced, val_balanced = sample_balanced_datasets(self.data,
                                                                    counts,
                                                                    unique_atoms_per_molecule,
                                                                    datapoints_per_label=self.n_sample_per_label)

            # sample hard cases
            sampled_train = sample_images(get_mol_sample_weight(self.data, base_path=self.base_path),
                                          n=self.n_sample_hard, )
            sampled_val = sample_images(get_mol_sample_weight(self.data, base_path=self.base_path),
                                        n=self.n_sample_hard // 100),

            # create splits with sampled data
            self.data.set_index('file_name', inplace=True)
            data_train = self.data.loc[sampled_train].reset_index()
            data_val = self.data.loc[sampled_val].reset_index()

            # concatenate both datasets
            data_train = pd.concat([data_train, train_balanced])
            data_val = pd.concat([data_val, val_balanced]).drop_duplicates()

            # create COCO annotations
            for data_split, mode in zip([data_train, data_val], ['train', 'val']):
                if os.path.exists(self.base_path + f'/data/annotations_{mode}.pkl'):
                    f"{color.BLUE}{mode.capitalize()} already exists, skipping...{color.END}"
                    continue
                params = [[row.SMILES,
                           row.file_name,
                           'train',
                           unique_labels,
                           self.base_path] for _, row in data_split.iterrows()]
                result = pqdm(params,
                              create_COCO_json,
                              n_jobs=self.n_jobs,
                              argument_type='args',
                              desc=f'{color.BLUE}Creating COCO-style {mode} annotations{color.END}')

                # clean any corrupted annotation
                result = [annotation for annotation in result if type(annotation) == dict]
                print(f'{color.PURPLE}Saving COCO annotations - {mode}{color.END}')
                with open(self.base_path + f'/data/annotations_{mode}.pkl', 'wb') as fout:
                    pickle.dump(result, fout)

            print(f'{color.BLUE}Saving training labels{color.END}')
            with open(self.base_path + f'/data/labels.json', 'w') as fout:
                json.dump(unique_labels, fout)

            return
        else:
            print(f"{color.BLUE}Preprocessed files already exist. Loading annotations... [train, val]{color.END}")
            return

    def create_cfg(self):
        """
        Creates configuration file for the model.
        :return:
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        # Passing the Train and Validation sets
        cfg.DATASETS.TRAIN = ("smilesdetect_train",)
        cfg.DATASETS.TEST = ("smilesdetect_val",)
        cfg.OUTPUT_DIR = self.base_path + '/trained_models'
        cfg.INPUT.FORMAT = self.input_format
        # Number of data loading threads
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.unique_labels)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        return cfg

    def train(self, train_params):
        """
        Train model with the parameters passed in train_params.
        :param train_params: Training parameters. [dict]
        :return:
        """
        self.cfg.SOLVER.IMS_PER_BATCH = train_params['images_per_batch']
        self.cfg.SOLVER.BASE_LR = train_params['learning_rate']
        self.cfg.SOLVER.MAX_ITER = train_params['maximum_iterations']
        self.cfg.SOLVER.CHECKPOINT_PERIOD = train_params['checkpoint_save_interval']
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = train_params['ROI_batch_per_image']
        self.cfg.TEST.EVAL_PERIOD = train_params['evaluation_interval']
        self.cfg.DATALOADER.NUM_WORKERS = train_params['num_workers']
        trainer = CocoTrainer(self.cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()
        self.load_model("model_final.pth")
        return

    def load_model(self, model_name, NMS_THRESH=0.6, SCORE_THRESH=0.4, device=None):
        """
        Load model from the folder "trained_models".
        :param model_name: model name. [str]
        :param NMS_THRESH: Non-maximum supression threshold, [float]
        :param SCORE_THRESH: Minimum label score accepted as output in a bbox. [float]
        :param device: 'cuda' or 'cpu'. [str]
        :return:
        """
        assert os.path.exists(self.base_path + '/trained_models'), "'trained_models' folder do not exist in root folder"
        print(f"{color.BLUE}Loading model{color.END}")
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, model_name)
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_THRESH
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
        if device is not None:
            self.cfg.MODEL.DEVICE = device
        self.predictor = CustomBatchPredictor(self.cfg)
        return

    def get_metadata(self, mode):
        """ HELPER FUNCTION - DONT CALL DIRECTLY
        Metadata loader to register dataset.
        :param mode: 'train' or 'val'. [str]
        :return: list of [dicts]. [list]
        """
        dataset_dicts = pickle.load(open(self.base_path + f'/data/annotations_{mode}.pkl', 'rb'))
        return dataset_dicts

    def predict(self, image_path):
        """
        Predict SMILES on a single image.
        :param image_path: Path to the image. [str]
        :return: SMILES. [str]
        """
        assert self.predictor, 'Model not loaded, first load model.'
        img = cv2.imread(image_path)[:, :, ::-1]
        output = self.predictor([img])[0]['instances'].to('cpu')

        return self.output_to_smiles(output)

    def output_to_smiles(self, output):
        """
        Generate SMILES from atoms list (['C0', 'C0', 'N1'...]) and bonds list ([0, 1, 'SINGLE), (1, 2, 'DOUBLE')]).
        :param output: output dictionaries from the model. [dict]
        :return: SMILES. [str]
        """
        output = {'bbox':         output.pred_boxes.tensor.numpy(),
                  'bbox_centers': output.pred_boxes.get_centers().numpy(),
                  'scores':       output.scores.numpy(),
                  'pred_classes': output.pred_classes.numpy()}

        atoms_list, bonds_list = inference.bbox_to_graph(output,
                                                         idx_to_labels=self.idx_to_labels,
                                                         bond_labels=self.bond_labels)
        return inference.mol_from_graph(atoms_list, bonds_list)

    def predict_batch(self, images_path='/data/images/test', batch_size=12):
        """
        Predict smiles for a batch of images in a path. Default path contains
        the images of the DACON competition.
        :param images_path: images path. [str]
        :param batch_size: batch size for predictions, 8GB GPU - 12 /. [int]
        :return: Pandas dataframe, columns = ['file_name', 'SMILES']. [Pandas DF]
        """
        # get image paths
        images_paths = glob.glob(self.base_path + f'{images_path}/*.png')

        print(f'{color.BLUE}Predicting bounding boxes - {len(images_paths) // batch_size} batches{color.END}')
        outputs = []
        for i in tqdm(range(0, len(images_paths), batch_size)):
            # input format for the model list[{"image"}: ...,], image: Tensor, image in (C, H, W) format.
            imgs = [cv2.imread(path)[:, :, ::-1] for path in images_paths[i:i + batch_size]]

            with torch.no_grad():
                # predict batch, move to cpu and add to outputs.
                outputs.extend([pred['instances'].to('cpu') for pred in self.predictor(imgs)])

        print(f'{color.BLUE}Generating molecular graphs from detected atoms and bonds{color.END}')

        res = []
        for i in tqdm(range(len(outputs))):
            res.append(self.output_to_smiles(outputs[i]))

        return pd.DataFrame({'file_name': [path.split('/')[-1].strip() for path in images_paths],
                             'SMILES':    res})

    def show_bboxes(self, image_path, split_per_label_type=False, default_font_size=8):
        """
        Show detected bounding boxes for atoms and bonds.
        :param image_path: path to the image. [str]
        :param split_per_label_type: Plot every label type separately. [bool]
        :param default_font_size: font size for the labels. [int]
        :return:
        """
        assert self.predictor, 'Model not loaded, first load model.'

        # Prediction
        img = cv2.imread(image_path)[:, :, ::-1]
        outputs = self.predictor([img])[0]

        if split_per_label_type:
            for i in outputs['instances'].to('cpu').pred_classes.unique():
                v = Visualizer(img, metadata=self.inference_metadata, scale=2)
                v._default_font_size = default_font_size
                out = v.draw_instance_predictions(filter_per_instance_class(outputs, i).to("cpu"))

                fig = plt.figure(figsize=(12, 12))
                plt.title(f"{image_path.split('/')[-1]} - {self.idx_to_labels[int(i)]}", fontsize=12)
                plt.imshow(out.get_image())
                plt.show()
                plt.close(fig)
        else:
            v = Visualizer(img, metadata=self.inference_metadata, scale=2)
            v._default_font_size = default_font_size
            out = v.draw_instance_predictions(filter_per_instance_class(outputs, None).to("cpu"))
            plt.title(f"{image_path.split('/')[-1]}")
            plt.figure(figsize=(12, 12))
            plt.imshow(out.get_image())
            plt.show()
        return


class CocoTrainer(DefaultTrainer):
    """
    COCO trainer for training verbose during training.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


class CustomBatchPredictor:
    """
    End-to-end predictor adapted to run on batches.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, images_batch):
        """
        :param images_batch (np.ndarray): an image of shape (H, W, C) (in BGR order).
        :return: bbox predictions. list[(dict),...]:
        """

        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                images_batch = [img[:, :, ::-1] for img in images_batch]
            height, width = images_batch[0].shape[:2]
            inputs = []
            for image in images_batch:
                image = self.aug.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width})
            predictions = self.model(inputs)
            return predictions
