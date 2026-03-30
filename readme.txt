Beyond Visible Spectrum: AI for Agriculture 2026
Automated Multimodal Crop Disease Diagnosis from multimodal remote sensing imagery 5th


Beyond Visible Spectrum: AI for Agriculture 2026

Submit Prediction
Overview
ICPR 2026 Competition on “Beyond Visible Spectrum: AI for Agriculture”
Abstract
The ICPR 2026 Competition on “Beyond Visible Spectrum: AI for Agriculture” offers a unique opportunity for researchers to advance computer vision techniques in agricultural crop disease monitoring. Building on a successful series of competitions from 2022 to 2025, which attracted over 1000 participants from over 10 countries across Europe, Asia, and North America.

The challenge focuses on developing innovative deep learning algorithms using extensive multi/hyperspectral and satellite remote sensing datasets across two main tasks:

Task 1: Automated Multimodal Crop Disease Diagnosis from Multimodal Remote Sensing Imagery: Participants will be provided with RGB, multispectral (G/R/RE/NIR), and hyperspectral imagery. They will be encouraged to develop multimodal models capable of adapting to remote sensing data of varying spectral and spatial information to classify the data into three categories: 'rust,' 'healthy,' and 'other.'
Task 2: Boosting Automatic Crop Diseases Classification using Sentinel Satellite Data and Self-Supervised Learning (SSL): Addressing the critical issue of limited labelled data in agriculture, this task requires participants to employ self-supervised learning (SSL) to learn meaningful representations from large, unlabelled Sentinel satellite datasets before fine-tuning on limited labelled data.
The primary goal is to enhance the accuracy and efficiency of identifying crop diseases, contributing to significant advancements in precision farming and sustainable agricultural practices to support global food security.

Background and Task
Remote sensing technology offers continuous global monitoring of vast agricultural areas in real-time, enabling early identification and mitigation of threats. Advances in digital imaging have led to various image-based methods for automated plant management, showing great potential for crop disease diagnosis using RGB, thermal, multi-spectral, and hyperspectral sensors. Hyperspectral imagery (HSI), with its numerous narrow spectral bands, provides detailed spectral–spatial information, offering enhanced diagnostic accuracy for disease infestation. However, the high data dimensionality of HSI can also present challenges, leading to research into new methods like super-resolution and automated detection models.

Deep learning methods have shown promising results in remote sensing imagery analysis. Identifying relevant spectral features and extracting meaningful representations from remote sensing data are crucial for effective deep learning. However, the high number of spectral bands and complex relationships in remote sensing images pose significant challenges for feature selection and extraction, necessitating the development of new deep learning algorithms.

Building on our research in precision agriculture and extensive experience, we are hosting The ICPR 2026 Competition on “Beyond Visible Spectrum: AI for Agriculture.” This is our fifth consecutive year running this competition. We have seen a remarkable and sustained increase in community engagement, demonstrating the critical importance of this research area.

Competition Schedule
Launch Date: December 15, 2025
Submission Opening: December 30, 2025
Deadline for Competition Participants: March 1st, 2026
Winner Announcement: March 15, 2026
Initial Submission of Competition Reports Deadline: April 1st, 2026
Presentation at ICPR Conference: August 17–21, 2026
Start

2 months ago
Close
16 days to go
Description
Task 1: Automated Multimodal Crop Disease Diagnosis from Multimodal Remote Sensing Imagery

Aims to detect rust areas by leveraging joint spectral and spatial information from a provided collection of RGB, multispectral, and hyperspectral imagery.
Encourages participants to develop multimodal models capable of adapting to remote sensing data of varying types and resolutions.
Prizes
Total Prizes Available: 3

1st Place - £100
2nd Place - £60
3rd Place - £ 40
Organizers
Prof. Liangxiu Han, Manchester Metropolitan University, United Kingdom
Prof. Wenjiang Huang, Aerospace Information Research Institute, Chinese Academy of Sciences
Technical Committee:

Dr. Xin Zhang, Manchester Metropolitan University, UK
Dr. Yue Shi, Manchester Metropolitan University, UK
Dr. Yingying Dong, Aerospace Information Research Institute, Chinese Academy of Sciences
Citation
robeson. Beyond Visible Spectrum: AI for Agriculture 2026. https://kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026, 2025. Kaggle.


Cite
Competition Host
robeson

Prizes & Awards
$200

Does not award Points or Medals

Participation
379 Entrants

216 Participants

183 Teams

1,330 Submissions

Tags
Accuracy Score
Table of Contents


Beyond Visible Spectrum: AI for Agriculture 2026
Automated Multimodal Crop Disease Diagnosis from multimodal remote sensing imagery 5th


Dataset Description
Wheat Disease Multimodal Classification Dataset
Overview
This dataset contains multimodal UAV imagery for the classification of wheat diseases. The data was collected to identify the spread of downy mildew and rust at critical growth stages. The dataset has been pre-processed into three modalities: RGB, Multispectral (MS), and Hyperspectral (HS), allowing for the development of multimodal deep learning models.

Data Acquisition
Dates: May 3, 2019 (Pre-grouting stage) and May 8, 2019 (Middle grouting stage).
Equipment: DJI M600 Pro UAV with an S185 snapshot hyperspectral sensor.
Flight Altitude: 60 meters (Spatial resolution ~4cm/pixel).
Spectral Range: 450-950nm (Visible to Near-Infrared).
Spectral Resolution: 4nm.
Data Modalities
For each sample, three aligned data types are provided:

RGB Images (/RGB)

Format: .png
Description: True-color images generated from the hyperspectral bands (Red: ~650nm, Green: ~550nm, Blue: ~480nm).
Multispectral Data (/MS)

Format: .tif (GeoTIFF)
Bands: 5 bands critical for vegetation health analysis:
Blue (~480nm)
Green (~550nm)
Red (~650nm)
Red Edge (740nm)
NIR (833nm)
Hyperspectral Data (/HS)

Format: .tif (GeoTIFF)
Bands: 125 bands (450-950nm).
Note: While the raw data contains 125 bands, the spectral ends (first ~10 and last ~14 bands) may contain sensor noise.
Dataset Structure
The dataset is organized into Training and Validation sets.

Training Set (/train)
Organized by class folders:

Health: Healthy wheat samples.
Rust: Samples infected with Rust.
Other: Other conditions or background.
Validation Set (/val)
Contains samples with randomized filenames to mask the class labels.

Files: Located in val/RGB, val/MS, and val/HS.
Labels: The ground truth for the validation set is provided in result.csv.
Task
Multimodal Classification: The goal is to classify each image patch into one of three categories using one or more of the provided modalities (RGB, MS, HS). 

Classes:

Health
Rust
Other
Submission Format
A CSV file with the following columns:

Id: The filename (e.g., val_a1b2c3d4.tif)
Category: The predicted class (Health, Rust, or Other)
Files
2700 files

Size
276.72 MB

Type
tif, png

License
MIT

Kaggle_Prepared(2 directories)
train

3 directories
val

3 directories
Data Explorer
276.72 MB

Kaggle_Prepared

train

val

Summary
2700 files


Download All
kaggle competitions download -c beyond-visible-spectrum-ai-for-agriculture-2026
Download data

Metadata
License
MIT


