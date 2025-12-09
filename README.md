# stat362-final-project-CorrKimGonzalezFlores

### Stat 362 Fall Quarter 2025 Final Project

#### Team Members
- Caroline Corr
- Hailee Kim
- Leslie Gonzalez-Flores

#### Project Title: CNN Dog Breed Classification

For our final project, our goal was to construct a deep learning convolutional neural network (CNN) model that can effectively classify 20 dog breeds with over 90% final testing accuracy. This work has many practical applications such as identifying stray/lost dogs and expediting animal shelter intake processes. To build such a model, we are utilizing the Stanford Dogs dataset, sourced from TensorFlow. This data set includes a total of 20,580 images and 120 classes of various dog breeds. Given the large size of the dataset, it is imperative to utilize modeling techniques that can withhold the complexity of the dataset. The types of models our project focuses on are building CNN image classifications models that allow for our model to learn and make predictions from images. To limit model complexity given our computational restraints, we pared down the dataset to the 20 most frequently occurring breeds.

We used three different pre-trained models throughout this experiment: Keras DenseNet121, Keras ResNet50, and HuggingFace Vision Transformer VIT model. Using these models as a baseline, we explored different augmentation and callback techniques as well as the introduction of transfer learning where early layers were frozen in order to fine tune the deeper ones.

#### Results
Hailee's best HuggingFace VIT model provided the best final testing accuracy of 0.9612 and loss of 0.1511. Caroline's best DenseNet121 model performed the same after transfer learning as it did without, with a final testing accuracy of 0.9363 and a loss of 0.1775. Leslie's best ResNet50 model had a final testing accuracy of ___ and loss of ___.

#### Steps to Running Code
- data_import.ipynb: *Optional* - isolated script for data import from TensorFlow. Also included in each notebook, but listed separately for EDA/inspection if needed.
- densenet121_model_CarolineCorr: Caroline's pre-trained model script; only best combinations of augmentation/callbacks/model architecture kept; can be run concurrently with other pre-trained model scripts
- huggingface_VIT_HaileeKim:
- leslie's_notebook_here: 


