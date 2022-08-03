import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadLiteModel, Detect, PrepareImagesForModel, PrepareSegmentationsForModel, \
    LoadModel
from Core.Identification import SeparateContours, Cleanup, DetectEdges, Label
from Core.ImageHandling import LoadPILImages
from Core.Tracking import MatchOrganoidsInImages, Inverse, Overlap
from Core.HelperFunctions import printRep
from PIL import Image
import skimage.measure

model = LoadModel(Path(r"Publication\ReviewerComments\MouseOrganoids\RetrainedModel"))
pilImages = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\images"))
images = PrepareImagesForModel(pilImages, model)
segmentationsRaw = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\segmentations"))
segmentations = []
for segmentation in segmentationsRaw:
    s = skimage.measure.label(np.asarray(segmentation.convert(mode="1")))
    s = Image.fromarray(s)
    s = s.resize([512, 512], resample=Image.Resampling.NEAREST)
    segmentations.append(np.asarray(s))
segmentations = np.stack(segmentations)
detections = Detect(model, images)
organoID_labeled = Cleanup(Label(detections, 0.5), 200, True, False)
segmentations = Cleanup(segmentations, 200, True, False)

countFile = open(r"Publication\ReviewerComments\MouseOrganoids\CountComparison.csv", "w+")
areaFile = open(r"Publication\ReviewerComments\MouseOrganoids\AreaComparison.csv", "w+")
countFile.write("Filename, Manual count, OrganoID count")
areaFile.write("Manual area, OrganoID area")

for i in range(segmentations.shape[0]):
    printRep(str(i + 1) + "/" + str(segmentations.shape[0]))
    manualOrganoids = skimage.measure.regionprops(segmentations[i])
    organoIDOrganoids = skimage.measure.regionprops(organoID_labeled[i])
    countFile.write("\n%s, %d, %d" % (
        Path(pilImages[i].filename).name, len(manualOrganoids), len(organoIDOrganoids)))

    matches = MatchOrganoidsInImages(manualOrganoids, organoIDOrganoids,
                                     costFunction=Inverse(Overlap),
                                     costOfNonAssignment=1)
    for manualOrganoid, organoIDOrganoid in matches:
        if manualOrganoid is None or organoIDOrganoid is None:
            continue
        areaFile.write("\n%d, %d" % (manualOrganoid.area,
                                     organoIDOrganoid.area))
printRep(None)
countFile.close()
areaFile.close()
