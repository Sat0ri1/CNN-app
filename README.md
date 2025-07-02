# Theraphosidae Species Classifier

**Author:** Paweł Grygielski  
**Description:**  
A CNN models for recognizing Theraphosidae (tarantula) species based on images.  
Built with TensorFlow/Keras and image preprocessing to classify 101 species.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Installation and Usage](#installation-and-usage)
- [Running on Streamlit Cloud (recommended - does not require installation)](#running-on-streamlit-cloud)
- [Usage](#usage)  
- [List of Recognized Species](#list-of-recognized-species)  
- [How It Works](#how-it-works)  
- [License](#license)  

---

## Project Overview

This project uses a convolutional neural network (CNN) to classify tarantula species from images.  
There were two models trained on approximately 110-130 images per class, with data augmentation techniques applied to improve generalization.
Both models and database were built by me and are too heavy for github. Online app is linked below (Running on Streamlit Cloud).

Paper is available in pdf (only in polish for now).

---

## Installation and Usage

1. Clone the repository:  
```bash
git clone https://github.com/Sat0ri1/CNN-app
cd CNN-app
```
2. Install required packages
```bash
pip install -r requirements
```
3.  Either run model on your computer without strimlit
```bash
py model_run.py
```
4. Or run model on your computer with streamlit
```bash
streamlit run app.py
```

## Running on Streamlit Cloud
Just click here to open the app:  
[Open Theraphosidae Species Classifier](https://share.streamlit.io/YourUsername/YourRepo/main/app.py)

## List of Recognized Species
The model recognizes the following 101 species and genera, mostly those popular in the pet trade.  
The full list is available in the [Species List file](./species_list.txt) and in Streamlit app linked above.

## How It Works
First model built from scratch is a CNN with convolutional layers and batch normalization for stable training.

Secound model is using transfer learning techniques and is based on InceptionV3 by Google.

Data augmentation techniques such as rotation, shifting, zooming, flipping, and brightness adjustment are used to improve generalization.

During training, metrics on the validation set are monitored using EarlyStopping and ModelCheckpoint callbacks to avoid overfitting.

## License
This project is released under the [MIT License with Commons Clause restriction](./LICENSE) — this means commercial use requires my permission.  
See the LICENSE file for details.
