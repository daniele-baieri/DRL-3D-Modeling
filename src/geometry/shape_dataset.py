import torch
import trimesh
import warnings
import numpy as np
import os
#import chardet

from typing import List, Dict, Union
from pathlib import Path

from trimesh.exchange.binvox import load_binvox#parse_binvox, voxel_from_binvox

from torch.utils.data import Dataset

from torch_geometric.data import Data
#from torch_geometric.io import read_obj
from torch_geometric.transforms import FaceToEdge
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import to_trimesh, from_trimesh


from agents.base_model import BaseModel

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

    def __init__(self, path: str, categories: List[str]=['chair'], 
                 train: bool=True, split: float=.8, voxel_grid_side: int=64):
        self.root = Path(path)
        self.paths = []
        self.synset_idxs = []
        self.voxel_grid_side = voxel_grid_side
        self.synsets = self.__convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]
        self.edge_transform = FaceToEdge(remove_faces=False)

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
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.paths += models
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # return 3D shape (mesh) and reference
        # label = self.labels[self.synset_idxs[idx]]
        obj_location = self.paths[idx] / 'models/model_normalized.solid.binvox'#model_normalized.obj'

        #mesh = self.read_obj(obj_location)
        #to_trimesh(mesh).show()
        model = load_binvox(open(obj_location, 'rb'))
        #voxels.show()
        print(len(model.points))
        voxels = torch.from_numpy(model.points).to(os.environ['DEVICE'])
        #print(voxels.shape)
        # mesh = self.edge_transform(mesh)
    
        # NOTE: this should officially work. spoiler: it didn't work

        #mesh = to_trimesh(mesh)
        #voxels = mesh.voxelized(pitch)
        #voxels.show()

        max_comp = torch.max(voxels)#torch.max(mesh.pos, dim=0)[0]
        min_comp = torch.min(voxels)#torch.min(mesh.pos, dim=0)[0]
        #print(max_comp, min_comp)
        #widest = torch.argmax(max_comp - min_comp, dim=0).item()
        pitch = (max_comp - min_comp) / (self.voxel_grid_side)#(max_comp[widest] - min_comp[widest]) / self.voxel_grid_side
        voxels = torch.unique(
            voxel_grid(
                voxels, torch.zeros(voxels.shape[0]), 
                #torch.from_numpy(mesh.pos).float(), 
                pitch, min_comp, max_comp#min_comp[widest], max_comp[widest] 
            )
        )
        print(voxels.shape)
        return {'target': model, 'mesh': voxels, 'reference': torch.rand((120, 120))}

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


    '''
    def __as_mesh(self, scene_or_mesh: Union[Trimesh, Scene]) -> Trimesh:
        """
        Convert a possible scene to a mesh.

        If conversion occurs, the returned mesh has only vertex and face data.
        """
        if isinstance(scene_or_mesh, Scene):
            if len(scene_or_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()))
        else:
            assert(isinstance(mesh, trimesh.Trimesh))
            mesh = scene_or_mesh
        return mesh
    '''
    '''
    def __align_and_resize(self, voxels: VoxelGrid) -> None:
        S = torch.cat((torch.from_numpy(voxels.scale.copy()).float(), torch.tensor([1.0])), 0) 
        #print(S)
        T = torch.diag(torch.pow(S, -1))
        voxels.apply_transform(T)
        T = torch.eye(4)
        T[[0,1,2], -1] = -1 * torch.from_numpy(voxels.translation.copy()).float()
        voxels.apply_transform(T)
        #print(T)
    '''
