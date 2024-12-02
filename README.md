# Human-Protein-Atlas---Single-Cell-Classification
This repository contains our solution for the <u>Kaggle Competition: Human Protein Atlas - Single Cell Classification, developed as part of our Big Data Project Spring 2021. The objective was to create models capable of segmenting and classifying individual human cells from microscope images, leveraging advanced data processing and machine learning techniques.

## 📖 Project Overview

The study of individual human cells reveals important insights into protein localization and cellular heterogeneity. Proteins are essential for cellular functions, and their subcellular distribution often determines their role in various cellular processes.

The **Human Protein Atlas** initiative provides an extensive dataset of microscope images labeled with protein localization patterns for entire populations of cells. This project addresses the need for fine-grained analysis by segmenting and classifying individual cells in these images.

This work contributes to:
- **Understanding cellular heterogeneity.**
- **Modeling the spatial organization of proteins within human cells.**
- **Enabling the discovery of mechanisms** that cannot be identified in bulk population studies.

---

## 🧬 Project Details
### Objective
Classify protein localization patterns in individual cells from microscopy images. This involves both:
1. **Cell segmentation:** Identifying individual cells in an image.
2. **Multi-label classification:** Assigning protein localization labels to each segmented cell.


### Dataset
The dataset consists of 4-channel microscopy images (`RGBY` channels) where:
- **Red:** Microtubules
- **Green:** Protein of interest
- **Blue:** Nucleus
- **Yellow:** Endoplasmic reticulum

For each image, multiple labels indicate the subcellular protein localizations. Labels are assigned for the entire image (weakly supervised), but models must classify individual cells.


### Key Labels (19 classes):
- Nucleoplasm
- Nuclear Membrane
- Nucleoli
- Nuclear Speckles
- Centrosome
- Plasma Membrane
- Cytosol, etc.

### Challenge
This is a **weakly-supervised, multi-label classification problem** requiring:
- **Accurate cell segmentation** from weak annotations.
- **Predicting subcellular localization** for individual cells in the image.

---
## 📂 Input Datasets

The datasets used in this project are available on Kaggle:

- [Human Protein Atlas - Single Cell Image Classification](https://www.kaggle.com/competitions/hpa-single-cell-image-classification/data)
- [HPA Cell Segmentator Models](https://www.kaggle.com/datasets/rdizzl3/hpacellsegmentatormodelweights/data)
- [Additional Preprocessed Metadata](https://www.kaggle.com/datasets/dschettler8845/hpa-sample-submission-with-extra-metadata)
Download these datasets and place them in the appropriate directories as required by the scripts.

---

## 🚀 Solution Workflow

### 1. Data Preprocessing
- **Image Decoding:** Loaded and stacked `RGBY` channels into 4-channel arrays for input.
- **RGB Conversion:** Combined `RGBY` channels into visualizable RGB images for data exploration.
- **Data Augmentation:** Applied test-time augmentations (flipping, rotation, brightness adjustments) to improve robustness.

### 2. Cell Segmentation
- **HPA Cell Segmentator:** Used the pre-trained `hpacellseg` tool for cell segmentation, generating bounding boxes and binary masks for each cell in the images.
- **Post-Processing:** Extracted individual cells by cropping and resizing to a standard size (`224x224`).

### 3. Multi-Label Classification
- Trained a **ResNet-50** architecture with a custom dense head for multi-label classification.
- Utilized **focal loss** to handle label imbalance.
- Incorporated test-time augmentations (TTA) for robust predictions.

### 4. Run-Length Encoding (RLE)
- Used RLE encoding for mask compression in submission files.

### 5. Test-Time Ensemble
- Combined predictions from different color channels (`Red`, `Green`, `Blue`, and `Yellow`) to improve classification accuracy.
- Applied weighted averaging to refine confidence scores.

The pipeline effectively segments cells and classifies protein localization with high accuracy. Key metrics include:
- **Accurate cell segmentation** for weakly annotated images.
- **Multi-label classification** with precise predictions for individual cells.

---

## 📦 Dependencies

Key libraries and tools used in this project:
- **TensorFlow:** Deep learning framework for model training and inference.
- **OpenCV:** Image processing (segmentation and bounding box extraction).
- **HPA-CellSegmentator:** Cell segmentation using pre-trained models.
- **NumPy & Pandas:** Data manipulation and analysis.
- **Matplotlib & Seaborn:** Data visualization.

---

## 🏆 Acknowledgments

- **Human Protein Atlas:** For providing the dataset and hosting the competition.
- **Kaggle community contributions:**
  - [Daniel Schettler](https://www.kaggle.com/dschettler8845) for multi-label classification inspiration.
