:: Commented out commands are long-running. Check these separately

::Publication\Training\Split.bat
::Publication\Training\Augment.bat
::Publication\Training\Train.bat
python Publication\Figure1\1b\ExampleImage.py
python Publication\Figure1\1c\ExampleImages.py
python Publication\Figure1\1c\ComputeTestingIOUS.py
python Publication\Figure2\2a\ExampleImage.py
python Publication\Figure2\2b-c\CompareIdentification.py
python Publication\Figure2\2b-c\Plot.py
::Publication\Figure2\2d\TrackingExample.bat
python Publication\Figure2\2d\TrackingFigure.py
::Publication\Figure3\RunModel.bat
python Publication\Figure3\FluorescenceAnalysis.py
python Publication\Figure3\3a\CreateFluorescenceOverlayImages.py
python Publication\Figure3\3b\Plot.py
python Publication\Figure3\3c\Plot.py
::python Publication\Figure4\SplitRetrain.bat
::python Publication\Figure4\AugmentRetrain.bat
::python Publication\Figure4\Retrain.bat
python Publication\Figure4\4b\ComputeIOU.py
python Publication\Figure4\4c\CompareIdentification.py
python Publication\Figure4\4c\Plot.py
::python Publication\Supplement\S1C\MeasureTrainingLosses.py
python Publication\Supplement\S1C\Plot.py
python Publication\Supplement\S2\ComputeTestingIOUS.py
python Publication\Supplement\S2\Plot.py
python Publication\Supplement\S4\Plot.py