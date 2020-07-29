# DRL based 3D Modeling tool

## Computer Science Master Degree @ La Sapienza Università di Roma, 2020
## Course project for the Deep Learning & Applied AI course

### Author: Daniele Baieri

## Abstract

This work is based on [this paper](https://arxiv.org/abs/2003.12397) by Lin C., Fan T., Wang W., and Nießner, M. My proposal is to replicate, and possibly improve, their approach to learning policies for the modeling of 3D objects based on deep reinforcement learning. 

Adding to their work, I examined the possibility to include non-Euclidean data (i.e. 3D shapes) in the Q-network itself.

I also implemented a small, but fully functional and reusable DRL framework based on PyTorch. Check the `src/agents/` and `src/agents/imitation` directories if you're interested.

## Dependencies

The following Python modules are required:

* [PyTorch](https://pytorch.org/)

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

* [TriMesh](https://github.com/mikedh/trimesh)

The following additional software is required:

* [OpenSCAD](https://www.openscad.org/) (for boolean mesh operations)

## Data

I used the following datasets to train different versions of my models:

* [ShapeNet Core V2](https://www.shapenet.org/) (rigid objects)

I did not include data in the repository for size and copyright issues. If you wish to try out the training process, follow the links and download it. You will find the required directory structure in `./data/info.txt`.
