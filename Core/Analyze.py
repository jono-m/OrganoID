from pathlib import Path
import numpy as np
import skimage.measure
import pandas as pd


def AnalyzeAndExport(images: np.ndarray, path: Path):
    with pd.ExcelWriter(str(path.absolute())) as writer:
        propertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                         'eccentricity', 'equivalent_diameter_area', 'euler_number',
                         'extent', 'feret_diameter_max', 'orientation',
                         'perimeter', 'perimeter_crofton', 'solidity']

        size = (np.max(images)+1, images.shape[0])
        data = {propertyName: pd.DataFrame(np.ndarray(size, dtype=str)) for propertyName in propertyNames}

        for t in range(images.shape[0]):
            regions = skimage.measure.regionprops(images[t])
            for propertyName in propertyNames:
                for region in regions:
                    value = getattr(region, propertyName)
                    label = region.label
                    data[propertyName].iloc[label, t] = str(value)

        for propertyName in propertyNames:
            data[propertyName].to_excel(writer, sheet_name=propertyName)
g