# ShareAD
## On the Practicability of Deep Learning based Anomaly Detection for Modern Online Systems: A Pre-Train-and-Align Framework

![image](https://github.com/IntelligentDDS/ShareAD/blob/main/img/workflow_of_anomaly_detection_ours.png)

This work extends upon our previous work ``Share or Not Share? Towards the Practicability of Deep Models for Unsupervised Anomaly Detection in Modern Online Systems'' at the 33rd IEEE International Symposium on Software Reliability Engineering (ISSRE 2022), which is also honored to get the best paper award in the research track.

## Start

#### Clone the repository

```
git clone https://github.com/IntelligentDDS/ShareAD.git
```

#### Get data

You can get the public datasets from:

* CTF_data: <https://github.com/NetManAIOps/CTF_data>
* SMD: <https://github.com/NetManAIOps/OmniAnomaly>
* JS_D2+D3: <https://github.com/NetManAIOps/JumpStarter>

Here, since we focus on anomaly detection in modern **large-scale** online systems, we may prioritize CTF_data. 

#### Install dependencies (with python 3.7.6) 

```
pip install -r requirements.txt
```

#### Running an Example

An example using the CTF_data is provided in the notebook `example_pretrain.ipynb` and `example_align.ipynb`.