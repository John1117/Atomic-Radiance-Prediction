# Atomic Radiance Prediction by Convolutional Neural Network and Integrated Gradients

This project is accomplished when I was a research assistant of [Dr. Hsiang-Hua Jen](https://sites.google.com/view/hsianghuajen/home) in Institute of Atomic and Molecular Sciences (IAMS), Academia Sinica. We implemented a convolutional neural network (CNN) model to predict the radiance of a one-dimensional atomic array. Additionally, an explainable AI technique, Integrated Gradients (IG), was applied to extract physics insights from the trained CNN model. For more informaiton, please refer to our published paper [here](https://iopscience.iop.org/article/10.1088/1361-6455/ac6f33).

## Features

- **Radiance Prediction**: CNN model for predicting the radiance of 1D atomic array.
- **Explainable AI**: Integrated Gradients (IG) used to interpret the CNN's predictions and extract physical insights.
- **Physics-Based Insight**: Use of IG to highlight relevant atomic interaction range.

## Project Structure

- `training_and_testing.py`: Script for training and testing the CNN model.
- `data.py`: Data generating and loading functions.
- `attribution.py`: Implementation of Integrated Gradients for model interpretation.
- `plot.py`: Visualization of the model’s predictions and IG results.

## Prerequisites
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `keras`
- `tensorflow` (for building the CNN and IG)

## References
- [Interpretable machine-learning identification of the crossover from subradiance to superradiance in an atomic array (our published paper)](https://iopscience.iop.org/article/10.1088/1361-6455/ac6f33)
- [Axiomatic Attribution for Deep Networks (the paper of Integrated Gradients)](https://arxiv.org/pdf/1703.01365)
- [TensorFlow tutorial of Integrated Gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
