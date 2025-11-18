# 🧬 Evaluating the Efficacy of Several Machine Learning Algorithms on Enhancers Annotation

## Description
This repository contains the full codebase, notebooks, and scripts used in the thesis: _"Evaluating the Efficacy of Several Machine Learning Algorithms on Enhancers Annotation"_ by Roxana-Andreea Bosnea (2023). 

The project explores the application of classical machine learning models and modern depe learning architecutres to classify genomic sequences as enhancer or non-enhancer regions. Experiments were performed on datasets of varying sizes and complexities, including whole-genome scale evaluation.

## Project Overview
Enhancers are key regulatory DNA elements responsible for controlling gene transcription. Identifying them computationally is challenging due to:
* Sequence variability
* Cell-type specifity
* Limited experimental labels

This work evaluates a range of machine learning models to determine which models perform best for enhancer annotation across different dataset sizes (7 features, 31 features) and at whole-genome scale.

## Models Evaluated
Classical ML: Logistic Regression, Random Forest, LightGBM, XGBoost
Deep Learning: DeepSTARR (CNN-based architecture)

Each model was trained and evaluated under:
* 7-feature dataset
* 31-feature dataset
* Whole genome dataset

## Datasets
Due to their size and licensing restrictions, the datasets are not included in the repository. However, dataset paths and usage isntructions are provided in each script.

## Results Summary
**Performance on 31-Feature Set**
Logistic Regression achieved the best overall performance obtaining:
* Accuracy: 0.81
* AUC: 0.83 (highest among all models)

More complex models (CNN, LightGBM, XGBoost) performed moderately but did not surpass Logistic Regression, suggesting that additional features introduced complexity and noise that non-linear models struggled to generalize from.

Random Forest showed the weakest performance (AUC 0.62), likely due to overfitting and difficulty handling high-dimensional genomic data.

**Performance on 7-Feature Set**
Using only the essential seven genomic features improved or maintained performance across most models, indicating the reduced feature set was highly informative. Top performers were:
* XGBoost: Accuracy 0.85, AUC 0.74
* Logistic Regression: Accuracy 0.84, AUC 0.80
* CNN: Accuracy 0.82, AUC 0.81

Random Forest again performed the worst (AUC 0.74), though CNN and XGBoost benefited from reduced noise and lower dimensionality.

Overall, the smaller feature set led to more stable and consistent performance.

**Whole Genome Testing**
Whole-genome evaluation was performed to assess scalability and model robustness on extremely large datasets.

**Full 31 Features**
* Logistic Regression remained the top performer with AUC 0.84
* CNN and XGBoost followed closely (AUC ~0.77)
* Random Forest performed poorly (AUC 0.62)

**Subset of 7 Features**
* Logistic Regression again ranked highest (AUC 0.81)
* CNN, XGBoost, and LightGBM showed consistent performance (AUC 0.75-0.77)
* Random FOrest significantly declined (AUC 0.55)

## Key Findings
* Logistic Regression is the most stable and consistently high-performing model across all datasets and feature sets.
* Reduced feature sets (7 features) often improved performance by reducing noise and computational burden.
* Non-linear models (CNN, XGBoost, LightGBM) performed well on smaller feature sets but struggled with high-dimensional data.
* Random Forest was the least effective model, especially with large or complex feature sets.
