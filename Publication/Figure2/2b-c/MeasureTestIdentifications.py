import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadLiteModel, Detect, PrepareImagesForModel
from Core.Identification import SeparateContours, Cleanup, DetectEdges
from Core.ImageHandling import LoadPILImages, LabeledImagesToColoredImages, SavePILImageStack
from Core.Tracking import MatchOrganoidsInImages, Inverse, Overlap
from Core.HelperFunctions import printRep
from Core.Config import ARIAL_FONT_PATH
from PIL import Image, ImageDraw, ImageFont
import skimage.measure

model = LoadLiteModel(Path(r"OptimizedModel"))
pilImages = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\images"))
segmentationsRaw = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\segmentations"))

# mouseImages = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\images")) + \
#               LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\training\pre_augmented\images")) + \
#               LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\validation\images"))
# mouseSegmentations = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\segmentations")) + \
#                 LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\training\pre_augmented\segmentations")) + \
#                 LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\validation\segmentations"))
#
# pilImages += mouseImages
# segmentationsRaw += mouseSegmentations

images = PrepareImagesForModel(pilImages, model)
segmentations = []
for segmentation in segmentationsRaw:
    s = skimage.measure.label(np.asarray(segmentation.convert(mode="1")))
    s = Image.fromarray(s)
    s = s.resize([512, 512], resample=Image.Resampling.NEAREST)
    segmentations.append(np.asarray(s))
segmentations = np.stack(segmentations)

detections = Detect(model, images)
organoID_labeled = Cleanup(
    SeparateContours(detections, DetectEdges(detections, 2, 0.005, 0.05, 0.5), 0.5, 2),
    200, True, False)

countFile = open(r"Publication\Figure2\2b-c\CountComparison.csv", "w+")
organoidFile = open(r"Publication\Figure2\2b-c\OrganoidComparison.csv", "w+")
outputPath = Path(r"Publication\Figure2\2b-c\ExampleImages")
outputPath.mkdir(exist_ok=True)
countFile.write("Filename,Manual,Automated")
organoidFile.write("Filename,ID,Feature,Manual,Automated")

segmentations = Cleanup(segmentations, 200, True, False)
for i in range(segmentations.shape[0]):
    filename = Path(pilImages[i].filename).stem
    printRep(str(i + 1) + "/" + str(segmentations.shape[0]))
    manualOrganoids = skimage.measure.regionprops(segmentations[i])
    organoIDOrganoids = skimage.measure.regionprops(organoID_labeled[i])
    countFile.write("\n%s,%d,%d" % (filename, len(manualOrganoids), len(organoIDOrganoids)))

    matches = MatchOrganoidsInImages(manualOrganoids, organoIDOrganoids,
                                     costFunction=Inverse(Overlap),
                                     costOfNonAssignment=1)
    oID = 1

    manualImage = np.zeros([512, 512], dtype=np.uint8)
    automatedImage = np.zeros([512, 512], dtype=np.uint8)
    for manualOrganoid, automatedOrganoid in matches:
        if manualOrganoid is None or automatedOrganoid is None:
            continue

        manualImage[tuple(manualOrganoid.coords.T)] = oID
        automatedImage[tuple(automatedOrganoid.coords.T)] = oID


        def circularity(rp):
            return (4 * np.pi * rp.area) / (rp.perimeter * rp.perimeter)


        data = (("Area", manualOrganoid.area, automatedOrganoid.area),
                ("Perimeter", manualOrganoid.perimeter, automatedOrganoid.perimeter),
                ("Solidity", manualOrganoid.solidity, automatedOrganoid.solidity),
                ("Eccentricity", manualOrganoid.eccentricity, automatedOrganoid.eccentricity),
                ("Circularity", circularity(manualOrganoid), circularity(automatedOrganoid)),
                ("X", manualOrganoid.centroid[1], automatedOrganoid.centroid[1]),
                ("Y", manualOrganoid.centroid[0], automatedOrganoid.centroid[0]))
        for name, manual, automated in data:
            organoidFile.write("\n%s,%d,%s,%f,%f" %
                               (filename, oID, name, manual, automated))
        oID += 1

    manualImageCol = Image.fromarray(LabeledImagesToColoredImages(manualImage))
    automatedImageCol = Image.fromarray(LabeledImagesToColoredImages(automatedImage))
    m = ImageDraw.Draw(manualImageCol)
    a = ImageDraw.Draw(automatedImageCol)
    font = ImageFont.truetype(ARIAL_FONT_PATH, 20)
    oID = 1
    for manualOrganoid, automatedOrganoid in matches:
        if manualOrganoid is None or automatedOrganoid is None:
            continue
        y, x = list(manualOrganoid.centroid)
        m.text((x, y), str(oID), anchor="ms",
               fill=(255, 255, 255), font=font)

        y, x = list(automatedOrganoid.centroid)
        a.text((x, y), str(oID), anchor="ms",
               fill=(255, 255, 255), font=font)
        oID += 1
    SavePILImageStack([manualImageCol, automatedImageCol], outputPath / (filename + ".tif"))

printRep(None)
countFile.close()
organoidFile.close()
