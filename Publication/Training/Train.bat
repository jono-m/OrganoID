:: Optimized model
python OrganoID.py train Publication\Dataset\OriginalData Publication\Training\OriginalModelEpochs model -B 2 -P 10 -E 400 -DR 0.125 -LR 0.001 -S 512 512 --lite

:: Trainable model
python OrganoID.py train Publication\Dataset\OriginalData TrainableModel TrainableModel -B 2 -P 10 -E 400 -DR 0.125 -LR 0.001 -S 512 512
