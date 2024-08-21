# Exploring Neural Joint Activity in Spiking Neural Networks for Fraud Detection

Source code of the paper accepted at CIARP 2024, the 27th IberoAmerican Congress on Pattern Recognition. 

## Paper Abstract
Spiking Neural Networks (SNNs), inspired by the real brain's behavior, offer an energy-efficient alternative to traditional artificial neural networks coupled with their neural joint activity, also referred to as population coding. This population coding is replicated in SNNs by attributing more than one neuron to each class in the output layer. This study leverages SNNs for fraud detection through real-world datasets, namely the Bank Account Fraud dataset suite, addressing the fairness and bias issues inherent in conventional machine learning algorithms. Different configurations of time steps and population sizes were compared within a 1D-Convolutional Spiking Neural Network, whose hyperparameters were optimized through a Bayesian optimization process.
Our proposed SNN approach with neural joint activity enables the classification of fraudulent opening of bank accounts more accurately and fairly than standard SNNs. The results highlight the potential of SNNs to surpass non-population coding baselines by achieving an average of 47.08% of recall at a business constraint of 5% of false positive rate, offering a robust solution for fraud detection. Moreover, the proposed approach attains comparable results to gradient-boosting machine models while maintaining predictive equality towards sensitive attributes above 90%.

**Keywords:** Spiking Neural Networks $\cdot$ Population Coding $\cdot$ Fraud Detection $\cdot$ Energy Efficiency $\cdot$ Responsible AI $\cdot$ Fair ML

## Installation

To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```
Download the six Variant of the Bank Account Fraud (BAF) Dataset and extract the parquet files to the data folder.

## Dataset

The Bank Account Fraud (BAF) dataset is a synthetic dataset based on real-world data that simulates the applications for bank account opening. The dataset contains 6 parquet files, each representing a different variant of the dataset (Base, Variant I, Variant II, Variant III, Variant IV, and Variant V). The dataset contains 30 features and a binary target variable indicating whether the application is fraudulent or not.

## Repository Structure

TBD

## Bibtex

To cite this work, use the following bibtex entry:
```bibtex
TBD
```

