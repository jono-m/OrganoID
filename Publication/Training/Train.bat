:: Optimized model
python OrganoID.py train Publication\Dataset\OriginalData Publication\Training\OriginalModelEpochs model --saveall --lite

:: Trainable model
python OrganoID.py train Publication\Dataset\OriginalData TrainableModel TrainableModel --saveall
