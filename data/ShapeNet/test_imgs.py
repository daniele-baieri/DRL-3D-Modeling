from pathlib import Path
import os, sys
from tqdm import tqdm

from PIL import Image
import torchvision.transforms.functional as TF


root = os.getcwd() + '/03948459'

for render in tqdm(Path(root).rglob('*models/model_render.png')):
    try:
        image = Image.open(render)
        reference = TF.to_tensor(TF.resize(TF.to_grayscale(image), size=(120, 120)))
    except:
        print(render)