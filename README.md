# ShareAD

This repo provides official code for ''On the Practicability of Deep Learning based Anomaly Detection for Modern Online Systems: A Pre-Train-and-Align Framework''.


## Introduction

This work extends upon our previous work ``Share or Not Share? Towards the Practicability of Deep Models for Unsupervised Anomaly Detection in Modern Online Systems'' at the 33rd IEEE International Symposium on Software Reliability Engineering (ISSRE 2022), which is also honored to get the best paper award in the research track.


![image](https://github.com/IntelligentDDS/ShareAD/blob/main/img/workflow_of_anomaly_detection_ours.png)

```
Operation and maintenance are critical activities in the whole life cycle of modern online software systems, and anomaly detection is a crucial step of these activities. Due to the difficulty in modelling the complex patterns of the telemetry data from online software systems to detect anomalies, recent studies mainly develop deep learning techniques to complete this task. Notably, even though the proposed techniques have achieved promising results in experimental evaluations, there are still several challenges for them to be successfully applied in a real-world modern online software system. Such challenges stem from some significant properties of modern online software systems, e.g., large scale, dynamics and complexity. This study investigates how these properties affect the adoption of deep anomaly detectors for the maintenance of modern online software systems, and summarizes three practicality gaps that need to be bridged, including the scalability gap, the availability gap and the alignment gap. To bridge these gaps, we propose an anomaly detection framework, namely ShareAD, based on a  pre-train-and-align paradigm. Specifically, we argue that pre-training a shared model for anomaly detection is an effective way to bridge the scalability gap and the availability gap. To support this argument, we systematically study the necessity and feasibility of model sharing for anomaly detection. We further propose a novel model as the backbone for model sharing, which works well for anomaly detection pre-training. Based upon Transformer encoder layers and Base layers, the proposed ShareAD pre-trained model can effectively model diverse patterns of different monitor entities, and further perform anomaly detection accurately. Besides, it can accept inputs with variant cardinalities, which is a required property for a model that needs to be shared. Then, to bridge the alignment gap, we propose ShareAD alignment to align the pre-trained model with operator preference by jointly considering the local observation context and sensitivity of each monitor entity. Extensive experiments on two real-world large-scale datasets from modern online software systems demonstrate the effectiveness and practicality of ShareAD, with a relative PR-AUC improvement of 6.66\%~325.30\% for anomaly detection pre-training, and a relative F1-score improvement of 15.73\%~175.04\% for anomaly detection alignment. Moreover, ShareAD is lightweight and efficient since its per-observation anomaly detection time is within 100$\mu$s using GPU and within 1ms using CPU.
```

## Datasets

You can get the public datasets from:

* CTF_data: <https://github.com/NetManAIOps/CTF_data>
* SMD: <https://github.com/NetManAIOps/OmniAnomaly>
* JS_D2+D3: <https://github.com/NetManAIOps/JumpStarter>


Please download the datasets and put them in corresponding folders. For example, put the CTF_data in the `CTF_data` folder.

## Quick Start

### Install dependencies (with python 3.7.6) 

```
pip install -r requirements.txt
```

### Example of Pre-training

An example of Pre-training using the CTF_data is provided in the notebook `example_pretrain.ipynb`.


### Example of Alignment

An example of Alignment using the CTF_data is provided in the notebook `example_alignment.ipynb`.

## Project Structure

The project structure is as follows:

```
.
├── examples: code examples for pre-training and alignment
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
    ├── requirements.txt: python dependencies
    ├── soft_dtw_cuda.py: soft dtw cuda implementation, a choice of loss function that can be used in the model
    ├── thresholding.py: thresholding utils
    └── transformer_ad.py: our sharead model for anomaly detection, including pre-training and alignment
```

## Use on Your Own Data

You can use your own data by following the steps below:

- Prepare your data in the format like the CTF_data. 
- Specifically, the data should be stored in a csv/txt file with only the data values separated by ',', and the data of different entities should be stored in different files. In a specific file, the data of each row are metrics from different timestamps, and each column represents different metric of an entity.
- Put the data in the corresponding folder.
- Set the datafiles and prefix parameters as the corresponding files and paths when calling MonitorEntityDataset interface to load the data.
- Use the our train and alignment functions to train and align the model on your data.

## Citation
If you find our work useful in your research, please consider citing our paper:

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