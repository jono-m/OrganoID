REQUIREMENTS:

OrganoID was run with the following software configuration:

- Windows 10 64-bit
- Python 3.9

---

INSTALLATION:

To set up OrganoID source dependencies, create an empty Conda environment (e.g., with miniconda) and
install all packages listed in <i><b>requirements.txt</b></i>.

NOTE: OrganoID uses TensorFlow for neural network predictions. It is recommended that this be
installed through the Conda package manager. Additionally, TensorFlow will automatically run on your
GPU if compatible libraries are installed for your graphics card (e.g. NVIDIA CUDA).

INSTRUCTIONS:

OrganoID is currently available as a command line tool. To see instructions for use, run:

> python OrganoID.py run -h

The OrganoID distribution comes with an optimized TensorFlow Lite model, <b>OptimizedModel</b>. This
model can be used for most applications:

> python OrganoID.py run OptimizedModel /path/to/images /path/to/outputFolder

If you would like to tune model performance for particular applications, the included model
<b>TrainableModel</b> can be re-trained through this tool. Run the following command to view
training instructions:

> python OrganoID.py train -h

Such as:

> python OrganoID.py train /path/to/trainingData /path/to/outputFolder NewModelName -M TrainableModel

DATASET:

The dataset for model training and all validation/testing from the OrganoID publication is openly available here:
https://osf.io/xmes4/
