from pathlib import Path
import os, sys
from tqdm import tqdm


root = os.getcwd()

for material in tqdm(Path(root).rglob('*models/model_normalized.mtl')):
    
    home = material.parents[0]
    os.mkdir(home / 'mtl')
    os.rename(material, home / 'mtl' / material.name)
