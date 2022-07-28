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
import skimage.measure

model = LoadModel(Path(r"Publication\Figure4\RetrainedModel"))
pilImages = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\images"))
images = PrepareImagesForModel(pilImages, model)
segmentations = PrepareSegmentationsForModel(
    LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\segmentations")), model)

detections = Detect(model, images)
organoID_labeled = Cleanup(Label(detections, 0.5), 100, False, False)

countFile = open(r"Publication\Figure4\4c\CountComparison.csv", "w+")
areaFile = open(r"Publication\Figure4\4c\AreaComparison.csv", "w+")
countFile.write("Filename, Manual count, OrganoID count")
areaFile.write("Manual area, OrganoID area")

segmentations = np.stack([skimage.measure.label(x) for x in segmentations])
segmentations = Cleanup(segmentations, 100, False, False)
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
