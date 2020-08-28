import torch


def voxelize(x: torch.FloatTensor, grid_size: int, x_max: float=None, x_min: float=None, device: str='cpu') -> torch.LongTensor:
    """
    Efficient point cloud voxelization. 
    """

    min_comp = x.min() if x_min is None else x_min
    max_comp = x.max() if x_max is None else x_max
    pitch = (max_comp - min_comp) / grid_size
    
    G = torch.zeros(grid_size, grid_size, grid_size, dtype=torch.long, device=device)
    # Given a vertex (x,y,z), transform it to an index of G (a triple of ints in [0,64))
    VOX = torch.floor((x - min_comp) / pitch).long()
    VOX[VOX >= grid_size] = grid_size-1
    G[VOX[:,0], VOX[:,1], VOX[:,2]] = 1
    return G.flatten()