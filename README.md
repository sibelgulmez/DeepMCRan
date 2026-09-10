# DeepMCRan
Deep MCRan Dataset

This repository contains the four ransomware datasets used in the Deep MCRan study (by Sibel Gulmez, Arzu Gorgulu Kakisim, Ibrahim Sogukpinar). 

If you use this dataset, please cite our paper:
Gulmez, S., Kakisim, A. G., & Sogukpinar, I. (2026). Deep MCRan: Hybrid Analysis-based Ransomware Detection Using Multi-Channel Data Fusion. IEEE Access.
DOI: 10.1109/ACCESS.2026.3727310

## Dataset Overview

| Dataset | Source | Ransomware Samples | Benign Samples | Ransomware Families |
|---------|--------|-------------------:|---------------:|--------------------:|
| RanVS | VirusShare, Windows System Files & Download.com | 9,000 | 9,000 | N/A |
| RanA | Moreira *et al.* | 1,023 | 1,134 | 25 |
| RanB | MalwareBazaar | 4,055 | 0 | 19 |
| RanZero | MalwareBazaar | 157 | 0 | 2 |

* The executable files of the **RanA** dataset were obtained from the previously published dataset introduced in the following paper: Moreira, C. C., Moreira, D. C., & de Sales Jr, C. D. S. (2023). Improving ransomware detection based on portable executable header using xception convolutional neural network. Computers & Security, 130, 103265.
* The executable files of the remaining three datasets were collected from publicly available repositories (VirusShare and MalwareBazaar).

All executable files were analyzed using our the methodology described in our paper, and the resulting datasets are provided in this repository. Please refer to our paper for detailed information about the datasets:
-- To be added --
