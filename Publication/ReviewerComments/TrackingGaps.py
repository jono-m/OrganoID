from PIL import Image
from Core.ImageHandling import SavePILImageStack, LoadPILImages, GetFrames
from pathlib import Path
import numpy as np

images = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\Timelapse.tif"))
frames = [np.asarray(x) for x in GetFrames(images[0])]
toSave = [Image.fromarray(x) for i, x in enumerate(frames) if i % 6 == 0]
SavePILImageStack(toSave, Path(r"Publication\ReviewerComments\Timelapse12hr.tiff"))
