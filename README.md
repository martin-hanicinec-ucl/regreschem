## Introduction:

This is a repository accompanying part II. of my PhD thesis:

*Hanicinec, M. 2021, Towards Automatic Generation of Chemistry Sets for Plasma Modeling Applications, University College London, United Kingdom.*

Please, refer to the thesis for the full documentation and explanation, this README file only provides some basic explanation and description of the files in the repository.
The repository contains a pre-trained machine learning-based regression model for approximation of reaction rate coefficients for plasma reactions at room temperature.

## Installation:
The code in this repository is written in Python, and `pipenv` is used to handle the virtual environment management.
When `pipenv` is installed, on Linux operating systems one might create the virtual environment containing all the necessary python packages by running 
```
pipenv sync
```
from the directory of the cloned project, where the `Pipfile.lock` is located.
This should result in the *completely identical* environment like the one used for the model development.
The environment requires Python 3.9, which needs to be installed already.
Once created, the environment can be activated by running 
```
pipenv shell
```
from its root directory (containing `Pipenv` file).

## Description of the repository
This repository is not a python package, but rather a collection of all the source files and code accompanying my development of the Regression model for estimating rate coefficients of binary heavy-species plasma reactions only involving ground-state species at the room temperature.
Please refer to the thesis for the full explanation, following is a short description of all the files present in the repository:

* `utils.py` defines the `get_final_regression_pipeline()` function, which will return the final trained `scikit-learn` regressor.
  The model itself is stored as `final_regression_pipeline.joblib` file, but it uses some custom defined transformers which need to be imported, so it is best to use the function.
  Consult `scikit-learn` documentation for how to use trained regression models.
* `sample_input.csv` is an example input for the regression model. The model accepts `pandas.DataFrame` instance, which can be built from the `csv` file.
  The order of the columns in the DataFrame does not matter, but their names need to be exacly as in the header of the `sample_input.csv` file.
  The index of the rows in the input DataFrame is not taken into account by the model, it is treated just as unique identifiers of reactions being estimated.
* `example_use_snippet.ipynb` as a Jupyter Notebook showing an example usage of the trained regression model.
  It shows it's import from the `utils.py` module, wrapping the `sample_input.csv` data into a `pandas.DataFrame` instance and predicting the rate coefficients for the ten reactions represented by the rows in the input DataFrame.
  The notebook can be launched from the active virtual environment by running 
  ```
  jupyter notebook example_use_snippet.ipynb
  ```
* `kinetic_regression_model.ipynb` is the main file in this repository.
  It contains a Jupyter notebook detailing the full ML process, including data analysis, training, hyperparameters optimization, estimation of the generalization error, discussions of results etc.
* Finally, `dataset_raw.csv` is the raw file used by the main Jupyter notebook to build the full dataset input file, and it is built from the `data_final.yaml` using the `build_raw_dataset.py` script.
  `data_final.yaml` file contains some useful info related to the reactions of the training datase, which are not used by the model, such as reaction IDs specific to their source databases, sources cited by the databases, where available, etc.

As the final remark, I would like to re-iterate that this documentation was not written to make sense on it's own, and should be read in conjunction with the PhD thesis.

