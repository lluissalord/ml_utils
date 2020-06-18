# Machine Learning Utils

Library for functions useful for daily usage on Data Science and Data Analytics fields.

This library includes several kind of utilities which are split by:

- DICOM utilities for processing of health images from DICOM files.
- Deep Neural Network utilities for general purposes in this field.
- Feature Selection utilities for reducing dimensionality of the input data wisely.
- Image utilities for general preprocessing on images.
- Plot utilities to facilitate generation of useful plots.
- Time Series utilities for time transformation and seasonality data extraction.
- Miscelaneas utilities.

See API documentation [here](https://lluissalord.github.io/ml_utils/)

## Instalation

This library requires some dependencies which are mandatory to be able to import the modules. However, some other dependencies are only required for specific functions from the modules, hence, only dependencies in use need to be installed.

### Required dependencies

```
conda install numpy matplotlib pandas tqdm
```

### Optional dependencies

```
conda install pydicom
```
```
conda install scipy
```
```
conda install scikit-learn
```
```
conda install tensorflow=2
```
```
conda install seaborn
```
```
conda install catboost
```
```
conda install -c menpo opencv
```
