# ShareAD

This repo provides official code for ''On the Practicability of Deep Learning based Anomaly Detection for Modern Online Systems: A Pre-Train-and-Align Framework''.


## Introduction

This work extends upon our previous work ``Share or Not Share? Towards the Practicability of Deep Models for Unsupervised Anomaly Detection in Modern Online Systems'' at the 33rd IEEE International Symposium on Software Reliability Engineering (ISSRE 2022), which is also honored to get the best paper award in the research track.



![image](https://github.com/IntelligentDDS/ShareAD/blob/main/img/workflow_of_anomaly_detection_ours.png)



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

## Use Your Own Data

You can use your own data by following the steps below:

- Prepare your data in the format like the CTF_data. 
- Specifically, the data should be stored in a csv/txt file with only the data values separated by ',', and the data of different entities should be stored in different files.
- Put the data in the corresponding folder.
- Set the datafiles and prefix parameters as the corresponding files and paths when calling MonitorEntityDataset interface to load the data.
- Use the our train and alignment functions to train and align the model on your data.

## Documentation

The documentation of the code is provided in the `docs` folder.