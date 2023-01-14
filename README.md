# MULT-MicroExpressionSpot
This repo is the implementation of our paper "Micro-expression Spotting with Multi-scale Local Transformer in Long Videos". The entire pipeline can be divided into five parts. Please use the code by the following content.

## Feature Extraction
### Optical flow calculation
We use TV-L1(opencv) to calculate optical flow, and the optical flow interval was 2. We save the optical flow in x and y directions separately.

### 3D feature extraction by the pretrained model
1) Features are extracted by [I3D] (https://github.com/Finspire13/pytorch-i3d-feature-extraction)
2) Sliding window ground truth information are generated
Reference address: (https://github.com/VividLe/A2Net)
SAMM: The sliding windows contain 256 features. Features are calculated with stride=2, thus one sliding window corresponding to 512 frames.
CAS(ME)^2: The sliding windows contain 128 features. Features are calculated with stride=2, thus one sliding window corresponding to 256 frames.

[CAS(ME)^2](https://pan.baidu.com/s/1z_jB7vkoHBf5MaoQ0Ky1KQ ), password:95lo
[SAMM](https://pan.baidu.com/s/1HzmuhuEQ0PyvIqfZHouzdA), password:d3lb 

## Modifying the configuration file
### experiments/samm(cas).yaml
    - ROOT_DIR
    - FEAT_DIR
    - ANNO_PATH

## Train the model
1) Select options in main.py (CAS(ME)^2 or SAMM)
2) Run main.py

## Evaluation
1) Select options in tools/F1_score.py (CAS(ME)^2 or SAMM)
2) Run tools/F1_score.py

## Accessing the Results 
Accessing existing results: https://pan.baidu.com/s/1f7gi95edkoFJWCXBl87I4g , password:rltx
