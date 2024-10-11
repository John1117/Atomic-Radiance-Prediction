# Atomic Radiance Prediction by Convolutional Neural Network and Integrated Gradients

This project is accomplished when I was a research assistant in [Dr. Hsiang-Hua Jen](https://sites.google.com/view/hsianghuajen/home)'s lab, Institute of Atomic and Molecular Sciences (IAMS), Academia Sinica. We implemented a convolutional neural network (CNN) model to predict the radiance of a one-dimensional atomic array. Additionally, an explainable AI technique, Integrated Gradients, was applied to extract physics insights from the trained CNN model. For more informaiton, please refer to [our published paper](https://iopscience.iop.org/article/10.1088/1361-6455/ac6f33).

## Table of Contents
- [Introduction](#atomic-radiance-prediction-by-convolutional-neural-network-and-integrated-gradients)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Start](#getting-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
- [References](#references)

## Features

- **Radiance Prediction**: CNN model for predicting the radiance of 1D atomic array.
- **Explainable AI**: Integrated Gradients used to interpret the CNN's predictions and extract physical insights.
- **Physics-Based Insight**: Use of Integrated Gradients to highlight relevant atomic interaction range.

## Project Structure

- `main.ipynb`: Jupyter Notebook for training and testing the CNN model.
- `data.py`: Data generating and loading functions.
- `attribution.py`: Implementation of Integrated Gradients for model interpretation.
- `plot.py`: Visualization of the model’s predictions and attribution results from IG calculation.

## Getting Start

### Prerequisites
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `tensorflow` (for building the CNN and IG)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/John1117/Atomic-Radiance-Prediction.git
    ```

2. Install the dependencies using:
    ```bash
    cd Atomic-Radiance-Prediction
    pip install -r requirements.txt
    ```

### Usage
If you would like to train your own CNN model, please refer to `main.ipynb` and tailor your model in `Network Configuration` section.

## References
- [Interpretable machine-learning identification of the crossover from subradiance to superradiance in an atomic array (our published paper)](https://iopscience.iop.org/article/10.1088/1361-6455/ac6f33)
- [Axiomatic Attribution for Deep Networks (the paper of Integrated Gradients)](https://arxiv.org/pdf/1703.01365)
- [TensorFlow Tutorial of Integrated Gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
