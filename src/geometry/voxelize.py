import torch


def voxelize(x: torch.FloatTensor, grid_size: int) -> torch.LongTensor:
    x_min, x_max = x.min(), x.max()
    pitch = (x_max - x_min) / grid_size
    
    G = torch.zeros(grid_size, grid_size, grid_size, dtype=torch.long)
    # Given a vertex (x,y,z), transform it to an index of G (a triple of ints in [0,64))
    VOX = torch.floor((x - x_min) / pitch).long()
    VOX[VOX >= grid_size] = grid_size-1
    G[VOX[:,0], VOX[:,1], VOX[:,2]] = 1
    return G.flatten()