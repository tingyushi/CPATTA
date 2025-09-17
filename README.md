# Active Test-Time Adaptation with Conformal Prediction

### Description

![CPATTA Method](images/schematic_diagram.png)

**C**onformal **P**rediction **A**ctive **TTA** (CPATTA), a framework that integrates CP into ATTA. The core idea is to replace heuristic uncertainty measures with principled, coverage-guaranteed ones, and to adapt them to dynamic test-time environments. Specifically, CPATTA introduces three key components.
* CPATTA employs smoothed conformal scores and a top-K certainty measure to provide fine-grained uncertainty signals, enabling more efficient allocation of scarce human annotations and reliable pseudo-labeling.
* CPATTA develops an online weight-update algorithm that leverages pseudo coverage as feedback to dynamically correct coverage under domain shifts, ensuring that uncertainty estimates remain calibrated to the user-chosen risk level. 
* CPATTA incorporates a domain-shift detector that increases human supervision when a new domain is encountered, preventing error accumulation at the onset of sudden distributional changes. 

CPATTA works as the following:  
* The real-time model $f(\cdot ; \theta)$ makes the predictions right away when an online batch of data arrives. 
* If CPATTA detects that current batch's domain is different from the previous batch's domain($D_t \neq D_{t-1}$), the algorithm selects more data for human annotation to ensure the real-time performance. 
* Two CPs provide uncertainty measures for each sample within the batch; human annotates uncertain samples while the model annotates certain samples. 
* Then, CPATTA updates the weights of two CPs based on the pseudo coverages calculated from model predictions and prediction sets.


### Datasets
* Please prepare PACS, VLCS, and Tiny-ImageNet-C. 
* You can download 3 datasets here [https://pan.baidu.com/s/10krX1S36XwM2Ix2vJdNWoQ?pwd=bvd3](https://pan.baidu.com/s/10krX1S36XwM2Ix2vJdNWoQ?pwd=bvd3).

### Requirements
* Python: 3.11.6
* Pytorch: 2.5.1
* CUDA: 11.8

### Configs and Runner
* ``cpatta_config.yml``: An example config file for CPATTA algorithm
* ``data_config.yml``: An example config file for loading dataset
* ``runner.py``: A driver file for running the algorithm
* The core CPATTA algorithm is [here](./cp_atta_package/atta_alg)
