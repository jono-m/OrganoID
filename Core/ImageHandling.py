from typing import List, Tuple, Union
import pathlib
import re
from PIL import Image, UnidentifiedImageError, ImageFont, ImageDraw
import numpy as np
import skimage.color
import skimage.filters
import skimage.measure
import math
from Core.HelperFunctions import printRep


def ComputeOutline(image: np.ndarray):
    return skimage.filters.sobel(image) > 0


def ImagesToHeatmaps(images: np.ndarray):
    print("Preparing heatmaps...", end="", flush=True)
    heatmaps = np.zeros(list(images.shape) + [3], dtype=np.uint8)
    for i in range(images.shape[0]):
        printRep(str(i) + "/" + str(images.shape[0]))
        image = images[i]
        minimum = np.min(image)
        maximum = np.max(image)
        hue = 44.8 / 360
        h = np.ones_like(image) * hue
        s = np.minimum(1, 2 - 2 * (image - minimum) / (maximum - minimum))
        v = np.minimum(1, 2 * (image - minimum) / (maximum - minimum))
        concat = np.stack([h, s, v], -1)
        converted = skimage.color.hsv2rgb(concat)
        heatmaps[i] = (converted * 255)
    printRep(None)
    print("Done.")
    return heatmaps


def LabeledImagesToColoredImages(images: np.ndarray, colors=None):
    if colors is None:
        colors = [(255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 255),
                  (255, 255, 0),
                  (255, 0, 255),
                  (0, 255, 255),
                  (255, 255, 255)]
    cycles = math.ceil(float(np.max(images)) / len(colors))
    colorMap = np.asarray([(0, 0, 0)] + colors * cycles, dtype=np.uint8)
    colorized = colorMap[images]
    return colorized


def NumFrames(image: Image):
    return getattr(image, "n_frames", 1)


def GetFrames(image: Image.Image):
    for i in range(NumFrames(image)):
        image.seek(i)
        yield image
    image.seek(0)


def PILImageForFrameInList(i, images: List[Image.Image]):
    for image in images:
        if i < NumFrames(image):
            return image
        i -= NumFrames(image)


def ConvertImagesToStacks(images: np.ndarray, originalImages: List[Image.Image]):
    stacks = []
    i = 0
    for originalImage in originalImages:
        start = i
        end = i + NumFrames(originalImage)
        stacks.append(images[start:end])
        i = end
    return stacks


def ConvertImagesToPILImageStacks(images: np.ndarray, originalImages: List[Image.Image],
                                  resize=True):
    stacks = ConvertImagesToStacks(images, originalImages)
    if resize:
        return [
            [Image.fromarray(d).resize(o.size, resample=Image.Resampling.NEAREST) for d in stack]
            for o, stack in
            zip(originalImages, stacks)]
    else:
        return [[Image.fromarray(d) for d in stack] for stack in stacks]


def SavePILImageStack(stack: List[Image.Image], path: pathlib.Path):
    if len(stack) == 1:
        stack[0].save(path)
    else:
        stack[0].save(path, save_all=True, append_images=stack[1:], compression=None)


def SaveAsGIF(images: np.ndarray, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).save(path, save_all=True,
                                    append_images=[Image.fromarray(im) for im in images[1:]],
                                    loop=0)


def LoadPILImages(source: Union[pathlib.Path, List[pathlib.Path]]) -> List[Image.Image]:
    def OpenAndSkipErrors(path: List[pathlib.Path]):
        for p in path:
            try:
                i = Image.open(p)
                yield i
            except UnidentifiedImageError:
                pass

    if isinstance(source, list):
        return sum([LoadPILImages(i) for i in source], [])
    if source.is_dir():
        # Load directory
        matches = sort_paths_nicely([path for path in source.iterdir() if path.is_file()])
        if len(matches) == 0:
            raise Exception(
                "Could not find any images in directory '" + str(source.absolute()) + "'.")
        return list(OpenAndSkipErrors(matches))
    if source.is_file():
        return [Image.open(source)]

    # Handle regular expression paths
    matches = sort_paths_nicely(
        [path for path in source.parent.glob(source.name) if path.is_file()])
    if len(matches) == 0:
        raise Exception("Could not find any images matching '" + str(source.absolute()) + "'.")
    return list(OpenAndSkipErrors(matches))


# Overlay a set of organoid regions on a list of base images.
def DrawRegionsOnImages(labeledImages: np.ndarray, images: np.ndarray,
                        textColor: Tuple[int, int, int],
                        fontSize: int, overlayColor: Tuple[int, int, int]):
    font = ImageFont.truetype("arial.ttf", fontSize)
    images = np.repeat(images[:, :, :, None], 3, axis=-1)
    outlined = np.zeros(images.shape[:-1], dtype=bool)
    for i in range(images.shape[0]):
        outlined[i] = ComputeOutline(labeledImages[i])
    drawnImages = np.where(outlined[:, :, :, None], overlayColor, images).astype(np.uint8)
    for i in range(images.shape[0]):
        image = Image.fromarray(drawnImages[i])
        drawer = ImageDraw.Draw(image)
        for rp in skimage.measure.regionprops(labeledImages[i]):
            x, y = reversed(rp.centroid)
            drawer.text((x, y), str(rp.label), anchor="ms", fill=textColor, font=font)
        drawnImages[i] = np.asarray(image)
    return drawnImages


def sort_paths_nicely(paths: List[pathlib.Path]):
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(paths, key=lambda x: alphanum_key(x.name))
