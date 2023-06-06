## A hands-on tutorial on explainable methods for machine learning with Python: applications to gender bias

This repository contains the material for the tutorial held at the [EuADS Summer School](https://www.euads.org/fjkdlasjdiglsmdgkcxjhvckh/euads-summer-school-913/) on Data Science for Explainable and Trustworthy AI (6-9 June 2023).

### Repository organisation

This repository contains the following folders:

- [code](https://github.com/aurorarq/euads-genderbias/tree/main/code): Python notebooks and scripts to run the examples in local.
- [data](https://github.com/aurorarq/euads-genderbias/tree/main/data): Datasets in CSV format.
- [gcolab](https://github.com/aurorarq/euads-genderbias/tree/main/gcolab): All-in-one notebooks to be used in Google Colab.

You can see the [slides](https://github.com/aurorarq/euads-genderbias/blob/main/EuADS-SummerSchool-GenderBias.pdf) used in the tutorial session too.

### Dependencies

The code has been developed with Python 3.10.2 using Visual Code Studio with Jupyter extension. The Jupyter Notebook Renderers extension is required to visualise the interactive plots generated by [dalex](https://github.com/ModelOriented/DALEX/). Machine learning algorithms are built with [sklearn](https://github.com/scikit-learn/scikit-learn) and [fairlearn](https://github.com/fairlearn/fairlearn).

If you want to run the examples on your machine, follow these steps:

1. Clone/zip this repository
2. Create a virtual environment using python env/conda. For python env: `python -m venv <your-venv-path>`
3. Activate your virtual environment (activate script on /bin or /Scripts depending on your OS)
4. Install the dependencies from the requirements file. For pip: `pip install -r requirements.txt`
5. Go to the code folder

### Datasets

The examples of this tutorial use two datasets:

- Example 1: employee promotion, the original file is available on [kaggle](https://www.kaggle.com/datasets/arashnic/hr-ana).
- Example 2: dutch census, a preprocessed file is available on [github](https://github.com/alku7660/counterfactual-fairness/blob/main/Datasets/dutch/preprocessed_dutch.csv).

### Notebooks

For each example, two notebooks are available:

Two notebooks are available for each example:

- Example 1: employee promotion
  - Data analysis: exploration of the features of the dataset
  - Machine learning + XAI: comparison of classifiers using different subsets of data with XAI techniques (model inspection, local explanations and counterfactuals).
- Example 2: Dutch census
  - Data analysis: exploration of the features of the dataset.
  - Fair machine learning + XAI: comparison of classifiers using bias mitigation methods with XAI techniques (model inspection, local explanations and counterfactuals).

### Funding

This tutorial is part of the [GENIA project](https://github.com/aurorarq/genia), funded by the Annual Research Plan (2022) of the University of Córdoba (Spain).
