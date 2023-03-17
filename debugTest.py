from Core.Model import GroundTruth, LoadGroundTruths, LoadFullModel
from pathlib import Path

model = LoadFullModel(Path("TrainableModel"))
trainingData = GroundTruth(Path(r"C:\Users\jonoj\Repositories\OrganoID\Publication\Dataset\MouseOrganoids\training\images\0ab74149-12d9-4ee2-b532-dfd4c33286c0.png"),
                           Path(r"C:\Users\jonoj\Repositories\OrganoID\Publication\Dataset\MouseOrganoids\training\segmentations\0ab74149-12d9-4ee2-b532-dfd4c33286c0.png"))
trainingData2 = GroundTruth(Path(r"C:\Users\jonoj\Downloads\t\aug1.jpeg"),
                           Path(r"C:\Users\jonoj\Downloads\aug1.jpeg"))

td = LoadGroundTruths([trainingData], model)
td2 = LoadGroundTruths([trainingData2], model)
print(td)
