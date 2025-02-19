# ShareAD

ShareAD is the first framework to perform model pre-training on **multivariate** time series and provide **anomaly detection alignment** support for software engineering operation scenarios.

* Extended Vesion Paper: ''On the Practicability of Deep Learning based Anomaly Detection for Modern Online Software Systems: A Pre-Train-and-Align Framework'', to appear in ACM Transactions on Software Engineering and Methodology (TOSEM).
* Conference Version Paper: ``Share or Not Share? Towards the Practicability of Deep Models for Unsupervised Anomaly Detection in Modern Online Systems'' at the 33rd IEEE International Symposium on Software Reliability Engineering (ISSRE 2022), which is also honored to get the best paper award in the research track.

This repo contains the official code of ShareAD.

## Introduction

![image](https://github.com/IntelligentDDS/ShareAD/blob/main/img/workflow_of_anomaly_detection_ours.png)


Operation and maintenance are critical activities in the whole life cycle of modern online software systems, and anomaly detection is a crucial step of these activities. Due to the difficulty in modelling the complex patterns of the telemetry data from online software systems to detect anomalies, recent studies mainly develop deep learning techniques to complete this task. Notably, even though the proposed techniques have achieved promising results in experimental evaluations, there are still several challenges for them to be successfully applied in a real-world modern online software system. Such challenges stem from some significant properties of modern online software systems, e.g., large scale, dynamics and complexity. This study investigates how these properties affect the adoption of deep anomaly detectors for the maintenance of modern online software systems, and summarizes three practicality gaps that need to be bridged, including the scalability gap, the availability gap and the alignment gap. To bridge these gaps, we propose an anomaly detection framework, namely ShareAD, based on a  pre-train-and-align paradigm. Specifically, we argue that pre-training a shared model for anomaly detection is an effective way to bridge the scalability gap and the availability gap. To support this argument, we systematically study the necessity and feasibility of model sharing for anomaly detection. We further propose a novel model as the backbone for model sharing, which works well for anomaly detection pre-training. Based upon Transformer encoder layers and Base layers, the proposed ShareAD pre-trained model can effectively model diverse patterns of different monitor entities, and further perform anomaly detection accurately. Besides, it can accept inputs with variant cardinalities, which is a required property for a model that needs to be shared. Then, to bridge the alignment gap, we propose ShareAD alignment to align the pre-trained model with operator preference by jointly considering the local observation context and sensitivity of each monitor entity. Extensive experiments on two real-world large-scale datasets from modern online software systems demonstrate the effectiveness and practicality of ShareAD, with a relative PR-AUC improvement of 6.66\%~325.30\% for anomaly detection pre-training, and a relative F1-score improvement of 15.73\%~175.04\% for anomaly detection alignment. Moreover, ShareAD is lightweight and efficient since its per-observation anomaly detection time is within 100$\mu$s using GPU and within 1ms using CPU.

## Project Structure

The project structure is as follows:

```
.
├── demo: code examples for pre-training and alignment
│   ├── check_points: sharead pre-trained model checkpoint
│   │   └── sharead_checkpoint.pt
│   ├── example_alignment.ipynb: example of alignment
│   └── example_pretrain.ipynb: example of pre-training
├── img
│   └── workflow_of_anomaly_detection_ours.png
├── issre2022_uniad.pdf: our conference version paper
├── README.md
├── requirements.txt: python dependencies
└── sharead
    ├── algorithm_utils.py: algorithm utils
    ├── datasets_for_JSer.py: dataset loader for JumpStarter Data
    ├── datasets_for_SMD.py: dataset loader for SMD Data
    ├── datasets_for_TencentData.py: dataset loader for Tencent Data
    ├── datasets.py: dataset loader for CTF Data
    ├── __init__.py: init file
    ├── soft_dtw_cuda.py: soft dtw cuda implementation, a choice of loss function that can be used in the model
    ├── thresholding.py: thresholding utils
    └── transformer_ad.py: our sharead model for anomaly detection, including pre-training and alignment
```

## Datasets

You can get the datasets from:

* CTF_data: <https://github.com/NetManAIOps/CTF_data>
* TC_data <https://drive.google.com/file/d/1U8DdiM9uCIJHT7_HiQH70F5v57kwWbDL>
* SMD: <https://github.com/NetManAIOps/OmniAnomaly>
* JS_D2+D3: <https://github.com/NetManAIOps/JumpStarter>


Please download the datasets and put them in corresponding folders. For example, put the CTF_data in the `CTF_data` folder.

## Quick Start

### Install dependencies (with python 3.7.6) 

```
pip install -r requirements.txt
```

### Demo Execution of Pre-training

A demo of Pre-training using the CTF_data is provided in the notebook `demo/example_pretrain.ipynb`.


### Demo Execution of Alignment

A demo of Alignment using the CTF_data is provided in the notebook `demo/example_alignment.ipynb`.

### Parameters Setting

The parameters of the model can be set in the `demo/example_pretrain.ipynb` and `demo/example_alignment.ipynb` files, or when you use our library in your code. The parameters mainly include:

#### For Data Loading

- `datafiles`: the data files to be loaded
- `prefix`: the path of the data files
- `sequence_len`: the sliding window sequence length

#### For Pre-training

- `lr`: the learning rate of the model
- `num_epochs`: the number of epochs for pre-training
- `sequence_len`: the sliding window sequence length
- `z_dim`: the dimension of the latent space
- `batch_size`: the batch size of the model

#### For Alignment

- `feedback_labels`: the feedback label of the operator for the alignment, recorded in a dictionary
- `feedback_amount`: the amount of feedback for the alignment, denoted as a percentage of the total data


## Use on Your Own Data

You can use your own data by following the steps below:

- Prepare your data in the format like the CTF_data. 
- Specifically, the data should be stored in a csv/txt file with only the data values separated by ',', and the data of different entities should be stored in different files. In a specific file, the data of each row are metrics from different timestamps, and each column represents different metric of an entity.
- Put the data in the corresponding folder.
- Set the datafiles and prefix parameters as the corresponding files and paths when calling `MonitorEntityDataset` interface to load the data.
- Use the our train and alignment functions to train and align the model on your data.

## Citation
If you find our work useful in your research, please consider citing our paper:

```
@article{10.1145/3712195,
author = {He, Zilong and Chen, Pengfei and Zheng, Zibin},
title = {On the Practicability of Deep Learning based Anomaly Detection for Modern Online Software Systems: A Pre-Train-and-Align Framework},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1049-331X},
url = {https://doi.org/10.1145/3712195},
doi = {10.1145/3712195},
note = {Just Accepted},
journal = {ACM Trans. Softw. Eng. Methodol.},
month = jan,
keywords = {anomaly detection, deep learning, pre-training, preference alignment, online software systems}
}
```

```
@inproceedings{DBLP:conf/issre/HeCH22,
  author       = {Zilong He and
                  Pengfei Chen and
                  Tao Huang},
  title        = {Share or Not Share? Towards the Practicability of Deep Models for
                  Unsupervised Anomaly Detection in Modern Online Systems},
  booktitle    = {{IEEE} 33rd International Symposium on Software Reliability Engineering,
                  {ISSRE} 2022, Charlotte, NC, USA, October 31 - Nov. 3, 2022},
  pages        = {25--35},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/ISSRE55969.2022.00014},
  doi          = {10.1109/ISSRE55969.2022.00014},
  timestamp    = {Sat, 01 Jul 2023 10:38:34 +0200},
  biburl       = {https://dblp.org/rec/conf/issre/HeCH22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact

- Zilong He (hezlong@mail2.sysu.edu.cn)
