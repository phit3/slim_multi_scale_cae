# Slim multi-scale CAE
Reproduction package for the Slim multi-scale CAE paper.

Before starting any model training, please make sure that the required libraries are installed, the desired dataset is provided, and the config.yaml was setup correctly.

To install the required libraries, run:
```bash
pip3 install -r requirements.txt
```

The data used in the corresponding paper can be retrieved from Dataverse at https://doi.org/10.7910/DVN/2M8RKD. Download the data via your favorite browser.

You can also use a custom snapshot data set. Ensure that the training, validation and test subsets are saved as numpy arrays in numpy files (.npy) with the shape (snapshot, 1, height, width) and the respective suffix (train, valid, test). Create a checkpoints and a data folder in the root directory of the project:

```bash
mkdir /path/to/project/checkpoints
mkdir /path/to/project/data
```

Move the dataset into the data folder (example 9o4E5_w64x64):
```bash
mv /path/to/9o4E5_w64x64_train.npy /path/to/project/data
mv /path/to/9o4E5_w64x64_valid.npy /path/to/project/data
mv /path/to/9o4E5_w64x64_test.npy /path/to/project/data
```

Change the working directory to the project directory:

```bash
cd /path/to/project
```

Modify the config file (config.yaml) as needed. It contains an example config, which should be straight forward to adapt. At least the file name base (data_fname) of your dataset and the checkpoint file name (cp_fname) should be set accordingly.

```yaml
data_params:
  ...
  data_fname: 9o4E5_w64x64
  ...
cae_params:
  ...
  cp_fname: test_cp
  ...
```

Exectue the CAE training procedure by running the main.py script.

```bash
python3 main.py
```

After the training converges, the model will process the test data. Subsequently, the NMSS is calculated and printed.
