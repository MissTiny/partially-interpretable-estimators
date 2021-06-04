
# Partially Interpretable Estimators

This repository is the official implementation of [Partially Interpretable Estimators](https://arxiv.org/abs/2030.12345). 

<!-- 
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
-->
## Requirements

To install requirements for baselines:

```setup
Rscript baseline_requirements.R
```

<!-- 
>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
-->
## Lasso_Gam_Xgboost
This folder contains the baseline coding functions for both regression and classification. Also an example of CASP dataset and the baseline running code is included in this file. 
In the paper, we uses Lasso, GAM, XGBoost as baselines

Before runing baseline code, please remember to the associated "load_function.R". The "CASP_baseline_code.R" is an example of Regression baseline, and the "adult_baseline_code.R" is an example of Classification baseline.

### NAM
This folder contains the modified code of NAM model. The original code is from "https://github.com/nickfrosst/neural_additive_models".

An example of NAM training code is presented in "nam_train_test.py" file with "CASP" dataset
Note: This baseline is done with Python

> Rishabh Agarwal, Nicholas Frosst, Xuezhou Zhang, Rich Caruana, and Geoffrey E Hinton. Neural additive models: Interpretable machine learning with neural nets. arXiv preprint arXiv:2004.13912, 2020.

### EBM
This folder contains the EBM code from github "https://github.com/interpretml/interpret".

An example of EBM training code is presented in "EBM.py" file with "glucose" dataset. The csv file is also provided in the folder.
Note: This baseline is done with Python

> Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, and Noemie Elhadad. Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1721–1730. ACM, 2015.

## Transfer Learning
This folder contains the implemented algorithm on the following article.

The "transferArgon.R" contains example code with dataset adult.

Note: This algorithm is only applied to classification dataset.

> Amit Dhurandhar, Karthikeyan Shanmugam, and Ronny Luss. Enhancing simple models by exploiting what they already know. In International Conference on Machine Learning, pages 2525–2534. PMLR, 2020.



## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
