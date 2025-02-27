### Installation
LandBench_Multi works in Python3.9.13, the following site-packages are required:

pytorch 1.13.1
pandas 1.4.4
numpy 1.22.0
scikit-learn 1.0.2
scipy 1.7.3
matplotlib 3.5.2
xarray 2023.1.0
netCDF4 1.6.2


### Data
The data is hosted https://doi.org/10.11888/Atmos.tpdc.300294

### Config File
Usually, we use the config file in model training, testing and detailed analyzing.

The config file contains all necessary information, such as path,data,model, etc.

### Process data and train model

Run the following command in the directory of `LandBench_Multi` to process data and start training.

```
python main.py 
```

### Detailed analyzing

Run the following command in the directory of `LandBench_Multi` to get detailed analyzing.

```
python postprocess.py 
python post_test.py
python post_test_2.py 
```
