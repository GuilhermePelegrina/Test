# Shapley value-based approaches to explain the robustness of classifiers in machine learning

### Authors: *Guilherme Dean Pelegrina*, *Sajid Siraj*.

## Introduction

This work proposes the use of Shapley values to explain the contribution of features towards the model's robustness, measured in terms of Receiver-operating Characteristics (ROC) curve and the Area under the ROC curve (AUC). For imbalanced datasets, the use of Precision-Recall Curve (PRC) is considered more appropriate, therefore we also demonstrate how to explain the PRCs with the help of Shapley values. 

To cite this work: Pelegrina, G. D. & Siraj, Sajid. (2022). Shapley value-based approaches to explain the robustness of classifiers in machine learning. ArXiv preprint, arXiv:2209.04254. Available at: https://arxiv.org/abs/2209.04254

## Execution Steps

All the files in this repository are in .py format, so it is necessary to execute them in Python. 

1) Choose what to explain (ROC curve, AUC, PR curve or AUPRC)
2) Clone the repository
3) Customize the dataset and the machine learning model (and other parameters, if it is the case)
4) Execute the file 
