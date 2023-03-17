<h1>REQUIREMENTS</h1>

OrganoID was run with the following software configuration:

- Windows 10 64-bit
- Python 3.9

<h1>INSTALLATION</h1>

<i>Overview: to set up OrganoID source dependencies, create an empty Conda environment (e.g., with miniconda) and
install all packages listed in <b>requirements.txt</b>.

NOTE: OrganoID uses TensorFlow for neural network predictions. TensorFlow will automatically run on your
GPU if compatible libraries are installed for your graphics card (e.g. NVIDIA CUDA). See tensorflow.org/install for guidance.</i>

1) Install Anaconda (https://www.anaconda.com/products/distribution).
2) Open <i>Anaconda Prompt</i> and create a new environment:
   ```
   >> conda create -n OrganoID python=3.9
   >> activate OrganoID
   ```
3) Download OrganoID and extract it to a directory of your choosing (https://github.com/jono-m/OrganoID/archive/refs/heads/master.zip). You may also clone the repository instead.
4) In <i>Anaconda Prompt</i>, navigate to the OrganoID root directory (which contains <i>OrganoID.py</i>):
   ```
   >> cd path/to/OrganoID/directory
   ```
5) If you would like to run TensorFlow on your GPU (which may be faster for batch processing), go to https://www.tensorflow.org/install/pip and follow the relevant instructions for your operating system, if GPU-mode is supported. (e.x. Step 5 for Windows Native). Skip this step otherwise.
6) Install all OrganoID requirements:
   ```
   pip install -r requirements.txt
   ```

<h1>USAGE</h1>

The OrganoID distribution comes with an optimized TensorFlow Lite model, <b>OptimizedModel</b>. This
model can be used for most applications. Here is an example of usage (run in <i>Anaconda Prompt</i> from the directory that contains <i>OrganoID.py</i>):

> python OrganoID.py run OptimizedModel /path/to/images /path/to/outputFolder

This command goes through each image in the <i>/path/to/images</i> folder and produces a labeled grayscale image, where the intensity at each pixel is the organoid "ID". These images are saved in the <i>/path/to/outputFolder</i> directory. You can also output other versions of this image with command options, such as <i>--binary</i>, <i>--belief</i>, or <i>--colorize</i> to generate black-and-white masks, detection belief images, or color-labeled images, respectively. To see all options with instructions, run the following command:

> python OrganoID.py run -h

If you would like to tune model performance for particular applications, the included model
<b>TrainableModel</b> can be re-trained through this tool. Run the following command to view
training instructions:

> python OrganoID.py train -h

Such as:

> python OrganoID.py train /path/to/trainingData /path/to/outputFolder NewModelName -M TrainableModel


<h1>USER INTERFACE</h1>

OrganoID now includes a user interface. To start the interface, run:

> python OrganoID_UI.py

The parameters in the interface correspond to those in the command-line tool.

<h1>DATASET</h1>

The dataset for model training and all validation/testing from the OrganoID publication is openly available here:
https://osf.io/xmes4/
