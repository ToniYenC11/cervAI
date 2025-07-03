# cerv.AI

This document serves as a comprehensive documentation on the deep learning models trained for the cerv.AI project.


## Team
**Ms. Roxanne S. Aviñante** - Project Supervisor 

**Mr. Jeffrey A. Aborot** - Technical Supervisor


**Toni Yenisei Czar S. Castañares** - Artificial Intelligence Engineer

**Rafael D. Ronquillo** - Artificial Intelligence Engineer

**Ken B. Horlador** - Artificial Intelligence Engineer

# Table of Contents

1. [About cerv.AI](#about-cervai)
2. [Theory on Models Trained](#theory-on-models-trained)
3. [Training and Validation Results](#training-and-validation-results)
4. [Visualization of Models](#visualization-of-models)
5. [Recommendations for Improvement](#recommendations-for-improvement)

---

## About cerv.AI

cerv.AI is a project under the Department of Science and Technology - Advanced Science and Technology Institute (DOST-ASTI). The aim of the project is to develop a screening system that will utilize deep learning models to increase cervical cancer detection in the Philippines, by up to **70%** of the female population.

### Cervical Cancer in the Philippines
Cervical cancer is the [fourth](https://www.who.int/news-room/fact-sheets/detail/cervical-cancer) most prominent cancer in female worldwide. Moreover, it is the [second](https://www.philhealth.gov.ph/news/2023/cervical_coverage.pdf) most prominent in the Philippines. In partnership with [cerviQ](https://endcervicalcancerph.com/), the DOST-ASTI developed the cerv.AI project to increase the inexpensive screening methods of cervical cancer, to reduce the cost for the more expensive screening methods, and to continuously increase the vaccination with the female population for the Human papillomavirus (HPV). 

### Goal of the project (2030)
- **90%** of girls fully vaccinated by HPV (There are only 23% vaccinated for the first dose, and 5% for the second dose).
- **70%** high-performance test and screening (less than 1% are currently screened nationwide)
- **90%** of positive cases receive treatment (Approximately 50-60% have received treatment from the screened cases)

---

## Theory on Models Trained

### 📌 RetinaNet

**RetinaNet** is a one-stage object detection model known for introducing **Focal Loss** to handle class imbalance during training. It combines the speed of one-stage detectors with the accuracy of two-stage detectors. RetinaNet uses a backbone (e.g., ResNet) with a **Feature Pyramid Network (FPN)** to detect objects at multiple scales and applies **anchor boxes** at each pyramid level to predict class and bounding box offsets.

> 🔍 Key Innovation: **Focal Loss** down-weights easy negatives and focuses training on hard, misclassified examples.

[Link to RetinaNet Paper](https://arxiv.org/abs/1708.02002)

---

### 📌 DETR (DEtection TRansformer)

**DETR** reformulates object detection as a **direct set prediction problem** using transformers. It eliminates traditional components like anchor boxes, non-maximum suppression (NMS), and region proposal networks. DETR uses a CNN backbone (like ResNet) to extract features and a **transformer encoder-decoder** to model global relationships. It predicts a **fixed number of objects** via bipartite matching using **Hungarian loss**.

> 🔍 Key Innovation: Integrates **transformer-based attention** directly into object detection.

[Link to DETR Paper](https://arxiv.org/abs/2005.12872)

---

### 📌 RT-DETR (Real-Time DETR)

**RT-DETR** is a real-time variant of DETR optimized for speed and deployment. It maintains the end-to-end set prediction formulation of DETR but uses a **lightweight backbone** and **efficient transformer design**. RT-DETR introduces techniques like **query selection**, **faster decoding**, and hardware-friendly architectures to achieve fast inference suitable for edge devices and real-time applications.

> 🔍 Key Innovation: Balances **transformer-based accuracy** with **real-time performance constraints**.

[Link to RT-DETR Paper]([INSERT_LINK_HERE](https://docs.ultralytics.com/models/rtdetr/))

---

## Training and Validation Results



---

## Visualization of Models



---

## Recommendations for Improvement

- 
