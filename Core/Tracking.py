# Tracker.py -- tracks organoids in sequences of labeled images

from typing import List, Optional, Callable
import numpy as np
from skimage.measure._regionprops import regionprops, RegionProperties
from scipy.optimize import linear_sum_assignment
from Core.HelperFunctions import printRep


class OrganoidTrack:
    # Collection of data points for a single identified organoid
    def __init__(self):
        self.id = 0
        self.regionPropsPerFrame: List[Optional[RegionProperties]] = []


CostFunctionType = Callable[[RegionProperties, RegionProperties], float]


# Cost functions
def Overlap(a: RegionProperties, b: RegionProperties):
    return np.size(IntersectCoords(a.coords, b.coords))


def Negative(f: CostFunctionType) -> CostFunctionType:
    return lambda x, y: -f(x, y)


def Inverse(f: CostFunctionType) -> CostFunctionType:
    def i_f(x, y):
        v = f(x, y)
        if v == 0:
            return np.inf
        return 1 / v

    return i_f


def PercentOverlap(a: RegionProperties, b: RegionProperties):
    larger = max(a.area, b.area)
    return float(np.size(IntersectCoords(a.coords, b.coords))) / larger


def IntersectCoords(a: np.ndarray, b: np.ndarray):
    imageWidth = max(np.max(a[:, 1]), np.max(b[:, 1]))
    aIndices = a[:, 0] * imageWidth + a[:, 1]
    bIndices = b[:, 0] * imageWidth + b[:, 1]
    return np.intersect1d(aIndices, bIndices)


def LastDetection(track: OrganoidTrack):
    return next((x for x in reversed(track.regionPropsPerFrame) if x is not None), None)


def Track(images: np.ndarray, costOfNonAssignment: float, costFunction: CostFunctionType,
          trackLostCutoff=10):
    tracks = []
    relabeledImages = np.zeros_like(images)
    print("Tracking images...", end="", flush=True)
    for i in range(images.shape[0]):
        printRep(str(i) + "/" + str(images.shape[0]))
        mapping = np.asarray(UpdateTracks(tracks, images[i], costOfNonAssignment, trackLostCutoff,
                                          costFunction))
        m2 = np.zeros(np.max(mapping[:, 0]) + 1)
        m2[mapping[:, 0]] = mapping[:, 1]
        mappedImage = m2[images[i]]
        relabeledImages[i] = mappedImage

    printRep("Done.")
    printRep(None)
    return relabeledImages


def IsTrackAvailable(track: OrganoidTrack, numFrames: int, trackLostCutoff: int):
    if trackLostCutoff is None or numFrames < trackLostCutoff:
        return True
    return any(track.regionPropsPerFrame[-trackLostCutoff:])


def UpdateTracks(currentTracks: List[OrganoidTrack], nextImage: np.ndarray,
                 costOfNonAssignment, trackLostCutoff, costFunction):
    mappingForThisImage = []

    nextID = max([x.id for x in currentTracks], default=0) + 1
    numFrames = len(currentTracks[0].regionPropsPerFrame) if currentTracks else 0

    # Morphologically analyze labled regions in the image
    detectedOrganoids = regionprops(nextImage)

    availableTracks = [t for t in currentTracks if IsTrackAvailable(t, numFrames, trackLostCutoff)]

    lastDetectedOrganoids = [LastDetection(availableTrack) for availableTrack in availableTracks]

    assignments = MatchOrganoidsInImages(detectedOrganoids, lastDetectedOrganoids, costFunction,
                                         costOfNonAssignment)

    for detectedOrganoid, lastDetectedOrganoid in assignments:
        if not lastDetectedOrganoid:
            # New track
            track = OrganoidTrack()
            track.id = nextID
            track.regionPropsPerFrame = [None] * numFrames
            currentTracks.append(track)
            nextID += 1
        else:
            track = next(t for t in availableTracks if LastDetection(t) is lastDetectedOrganoid)
        track.regionPropsPerFrame.append(detectedOrganoid)
        if detectedOrganoid is not None:
            mappingForThisImage.append((detectedOrganoid.label, track.id))

    return mappingForThisImage


def MatchOrganoidsInImages(organoidsA: List[RegionProperties], organoidsB: List[RegionProperties],
                           costFunction: CostFunctionType, costOfNonAssignment):
    fullSize = len(organoidsA) + len(organoidsB)
    costMatrix = np.zeros([fullSize, fullSize], dtype=float)

    costNonA = np.full([len(organoidsA), len(organoidsA)], np.inf)
    costNonB = np.full([len(organoidsB), len(organoidsB)], np.inf)
    np.fill_diagonal(costNonA, costOfNonAssignment)
    np.fill_diagonal(costNonB, costOfNonAssignment)
    costMatrix[:len(organoidsA), len(organoidsB):] = costNonA
    costMatrix[len(organoidsA):, :len(organoidsB)] = costNonB
    for i, a in enumerate(organoidsA):
        for j, b in enumerate(organoidsB):
            costMatrix[i, j] = costFunction(a, b)
    assignment = linear_sum_assignment(costMatrix)
    return [(organoidsA[i] if i < len(organoidsA) else None,
             organoidsB[j] if j < len(organoidsB) else None)
            for i, j in zip(assignment[0], assignment[1])
            if i < len(organoidsA) or j < len(organoidsB)]
