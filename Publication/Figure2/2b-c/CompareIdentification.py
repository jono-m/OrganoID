import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadLiteModel, Detect, PrepareImagesForModel, PrepareSegmentationsForModel
from Core.Identification import SeparateContours, Cleanup, DetectEdges
from Core.ImageHandling import LoadPILImages
from Core.Tracking import MatchOrganoidsInImages, Inverse, Overlap
from Core.HelperFunctions import printRep
import skimage.measure

model = LoadLiteModel(Path(r"OptimizedModel"))
pilImages = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\images"))
images = PrepareImagesForModel(pilImages, model)
segmentations = PrepareSegmentationsForModel(
    LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\segmentations")), model)

detections = Detect(model, images)
organoID_labeled = Cleanup(SeparateContours(detections, DetectEdges(detections, 2, 0.005, 0.05, 0.5), 0.5, 2),
                           200, False, False)

squareMicronsPerSquarePixel = 6.8644

countFile = open(r"Publication\Figure2\2b-c\CountComparison.csv", "w+")
areaFile = open(r"Publication\Figure2\2b-c\AreaComparison.csv", "w+")
countFile.write("Filename, Manual count, OrganoID count")
areaFile.write("Manual area, OrganoID area")

segmentations = np.stack([skimage.measure.label(x) for x in segmentations])
segmentations = Cleanup(segmentations, 200, False, False)
print(segmentations.shape)
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
        areaFile.write("\n%d, %d" % (manualOrganoid.area * squareMicronsPerSquarePixel,
                                     organoIDOrganoid.area * squareMicronsPerSquarePixel))
printRep(None)
countFile.close()
areaFile.close()
