import torch
import trimesh
import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
#import chardet

from typing import List, Dict, Union
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as TF

from trimesh.exchange.binvox import load_binvox#parse_binvox, voxel_from_binvox

from torch.utils.data import Dataset

from torch_geometric.data import Data
#from torch_geometric.io import read_obj
from torch_geometric.transforms import FaceToEdge
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import to_trimesh, from_trimesh

from agents.base_model import BaseModel
from geometry.voxelize import voxelize

# Synset to Label mapping (for ShapeNet core classes)
synset_to_label = {
    '04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
    '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
    '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
    '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
    '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
    '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
    '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
    '04554684': 'washer', '04090263': 'rifle', '02946921': 'can'
}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}


class ShapeDataset(Dataset):

    def __init__(self, path: str, items_per_category: Dict[str, int], 
                 partition: str='RFC', train_split: float=.9, imit_split: float=.03,
                 voxel_grid_side: int=64):
        """
        @param partition: string in {'RFC', 'IMIT', 'TEST'} (resp. reinforcement, imitation, testing).
        @param train_split: portion of data used for training. 1-@train_split = portion of data used for testing.
        @param imit_split: portion of *training* data used for imitation learning. 
            @train_split-@imit_split = portion of data used for reinforcement learning.
        """
        self.root = Path(path)
        self.paths = []
        self.synset_idxs = []
        self.voxel_grid_side = voxel_grid_side
        synset_names = list(items_per_category.keys())
        self.synsets = self.__convert_categories(synset_names)
        self.labels = [synset_to_label[s] for s in self.synsets]
        #self.edge_transform = FaceToEdge(remove_faces=False)

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError(
                    'Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target))
                )

            # find all objects in the class
            models = sorted(class_target.glob('*'))
            models = [x for x in models if os.path.exists(x / 'models/model_render.png')]
            models = models[:items_per_category[synset_names[i]]]
            stop = int(len(models) * train_split)
            if partition == 'RFC' or partition == 'IMIT':
                models = models[:stop]
                stop = int(len(models) * imit_split)
                models = models[:stop] if partition == 'IMIT' else models[stop:]
            else:
                models = models[stop:]
            self.paths += models
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # return 3D shape (mesh) and reference

        obj_location = self.paths[idx] / 'models/model_normalized.solid.binvox'#model_normalized.obj'
        render = self.paths[idx] / 'models/model_render.png'

        model = load_binvox(open(obj_location, 'rb'))
        #model.show()
        #print(len(model.points))
        voxels = torch.from_numpy(model.points)#.to(os.environ['DEVICE'])
        voxels = voxelize(voxels, 64)
        #print(voxels.sum())

        image = Image.open(render)
        reference = TF.to_tensor(TF.resize(TF.to_grayscale(image), size=(120, 120)))
        #plt.imshow(reference.squeeze())
        #plt.show()

        return {'target': model, 'mesh': voxels.bool(), 'reference': reference}

    def __convert_categories(self, categories):
        assert categories is not None, 'List of categories empty'
        synsets = [label_to_synset[c] for c in categories if c in label_to_synset.keys()]
        if len(synsets) < len(categories):
            warnings.warn('Selected unavailable categories - skipped')
        return synsets

    def read_obj(self, in_file):
        vertices = []
        faces = []

        for k, v in self.__yield_file(in_file):
            if k == 'v':
                vertices.append(v)
            elif k == 'f':
                faces.append(v)

        if not len(faces) or not len(vertices):
            return None

        pos = torch.tensor(vertices, dtype=torch.float)
        face = torch.tensor(faces, dtype=torch.long).t().contiguous()

        data = Data(pos=pos, face=face)

        return data

    def __yield_file(self, in_file):
        f = open(in_file, 'r')
        #f
        buf = f.read()
        #print(chardet.detect(buf))
        f.close()
        for b in buf.split('\n'):
            if b.startswith('v '):
                yield ['v', [float(x) for x in b.split(" ")[1:]]]
            elif b.startswith('f '):
                triangles = b.split(' ')[2:]
                # -1 as .obj is base 1 but the Data class expects base 0 indices
                yield ['f', [int(t.split("/")[0]) - 1 for t in triangles]]
            else:
                yield ['', ""]
