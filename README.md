# stat362-final-project-CorrKimGonzalezFlores

### Stat 362 Fall Quarter 2025 Final Project

#### Team Members

- Caroline Corr
- Hailee Kim
- Leslie Gonzalez-Flores

#### Project Title: CNN Dog Breed Classification

- For our final project, our goal was to construct a deep learning convolutional neural network (CNN) model that can effectively classify 20 dog breeds with over 90% final testing accuracy. This work has many practical applications such as identifying stray/lost dogs and expediting animal shelter intake processes. To build such a model, we are utilizing the Stanford Dogs dataset, sourced from TensorFlow. This data set includes a total of 20,580 images and 120 classes of various dog breeds. Given the large size of the dataset, it is imperative to utilize modeling techniques that can withhold the complexity of the dataset. The types of models our project focuses on are building CNN image classifications models that allow for our model to learn and make predictions from images. To limit model complexity given our computational restraints, we pared down the dataset to the 20 most frequently occurring breeds.
- We used three different pre-trained models throughout this experiment: Keras DenseNet121, Keras ResNet50, and HuggingFace Vision Transformer VIT model. Using these models as a baseline, we explored different augmentation and callback techniques as well as the introduction of transfer learning where early layers were frozen in order to fine tune the deeper ones.

#### Results

Hailee's best HuggingFace VIT model provided the best final testing accuracy of 0.9612 and loss of 0.1511. Caroline's best DenseNet121 model performed the same after transfer learning as it did without, with a final testing accuracy of 0.9363 and a loss of 0.1775. Leslie's best ResNet50 model had a final testing accuracy of 0.9208 and loss of 0.2926. Accompanying model visualizations can be seen in relevant notebook files. 

#### Steps to Running Code
##### Relevant notebooks and files: 

- 'Baseline_Model.ipynb': 
- 'Data_Preparation.ipynb': *Optional* - isolated notebook for data import from TensorFlow and necessary data preperation.Also included in each notebook, but listed separately for EDA/inspection if needed.
- 'DenseNet121_model_CarolineCorr.ipynb': Caroline's pre-trained model script; only best combinations of augmentation/callbacks/model architecture kept; can be run concurrently with other pre-trained model scripts
- 'GoogleViT_HaileeKim.ipynb': Hailee's pre-trained model script; incorporates a pre-trained Google Vision Transformer (ViT) model sourced directly from HuggingFace
- 'ResNet50_LeslieGonzalezFlores.ipynb': Leslie's pre-trained model script; attempts made with custom head before data augumentation, with attempted data augmentation, and then transfer learning using best data augmentation techniques can be seen. Each model run has accompanying accuracy and loss learning curves, and confusion matrix for how each of the 20 classes performed.

##### Steps for Setting Up Enviornment: 

1. Clone this repository
 ```
git clone <this-repo-url>
cd <this-project-folder>
```
2. Create a Python Virtual Environment in Current Project Directory 
```
python -m venv venv
```
3. Activate Virtual Enviornment
For Windows 
```
venv\Scripts\activate
```
For macOS\Linux 
```
source venv/bin/activate
```
4. Install packages from requirments.txt file
```
pip install -r requirments.txt
```
5. Deactivate virutal enviornment if necessary (once finished with utilizing enviornment for running notebooks). 
```
deactivate
```
Once virtual enviornment for package dependencies has been completed, notebook files mentioned above can be open and run. Note: notebook files were created and run on Google Colab T4 GPU. Using CPU or differing GPU's may produce slower runtimes when running notebook files. All notebooks have data preperation steps that imports the data from TensorFlow, downloads the data into necessary folders. It is recommended to utilize the data_preperation.ipynb file to download and prepare the data to ensure proper data setup prior to running notebook files. 

#### Data Sources Utilized:

1. Data Source: https://www.tensorflow.org/datasets/catalog/stanford_dogs 
2. Data Source Code Github Builder Page: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/stanford_dogs/stanford_dogs_dataset_builder.py

#### Academic Sources Utilized: 

MadTcTech. (2025, September 8). Create virtual environment and requirements.txt in Python [Video]. 
YouTube. https://youtu.be/h8bt4RvE7zM <br> 

Google. (n.d.). vit-base-patch16-224 [Model]. Hugging Face. Retrieved from https://huggingface.co/google/vit-base-patch16-224 <br> 

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90 <br>

Keras ResNet50 | Image File Handling and Transfer Learning. (2022, October 3). EDUCBA. https://www.educba.com/keras-resnet50/ <br> 

Analytics Vidhya. (2021, October). Understanding transfer learning for deep learning. Retrieved from https://www.analyticsvidhya.com/blog/2021/10/understanding-transfer-learning-for-deep-learning/ <br>  
TensorFlow Datasets. (2023). Stanford Dogs dataset [Data set]. https://www.tensorflow.org/datasets/catalog/stanford_dogs <br> 

TensorFlow. (n.d.). Transfer learning and fine‑tuning [Tutorial]. https://www.tensorflow.org/tutorials/images/transfer_learning <br> 



