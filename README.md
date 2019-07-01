# Evaluations on hand pose estimation with RGB 
[我的博客](https://blog.csdn.net/qq_32896115)  
## Description
This project provides codes to evaluate performances of hand pose estimation on several public RGB datasets, including RHD, Dexter+Object, Stereo hand pose dataset. We collect predicted labels of some prior work which are available online and visualize the performances.
## Datasets
* **Stereo hand pose dataset (SHD)** 
provides frames of 12 stereo video sequenceseachrecordingasinglepersonperformingvarious gestures [66]. It contains total 36,000 frames. Among them, 30,000 frames sampled from 10 videos constitute a training set while the remaining 6,000 frames (from 2 videos) are used for testing.

* **The rendered hand pose dataset (RHD)**  
contains 43,986 synthetically generated images showing 20 differentcharactersperforming39actionswhere41,258images are provided for training while the remaining 2,728 frames are reserved for testing [68]. Both datasets are recorded under varying backgrounds and lighting conditions and they are provided with the ground-truth 2D and 3D skeleton positions of 21 keypoints (1 for palm and 4 for each ﬁnger), on which the accuracy is measured. 
* **The Dexter+Object dataset (DO)** 
contains 3,145 video frames sampled from 6 video sequences recording a single person interacting with an object

## Evaluation metric
There are two types of evaluation metrics that are widely used for hand pose estimation:
* **3D PCK & AUC :** the percentage of correct keypoints of which the Euclidean error distance is below a threshold
* **EPE：** mean end-point-error
## Methods and corresponding predicted labels
### the Rendered Hand Pose Dataset(RHD)
* DHPE[1]:CVPR'19, 0.926 (3D PCK:0~50%)
* Weakly[5]:ECCV'18, 0.887(3D PCK:0~50%)
* Umar[4]:ECCV'18, 3.57(pixel)(2D)  13.41(mm)(3D)(EPE)
* CPM[2]:arXiv'18, 5.223(in pixels)(2D)(EPE)
### Stereo hand pose(SHD)
* DHPE[1]:CVPR'19, 0.995 (3D PCK:0~50%)
* Umar[4]:ECCV'18, 0.994 (3D PCK:0~50%)
* GANerated[3]:CVPR'18, 0.965 (3D PCK:0~50%)
* CPM[2]:arXiv'18, 5.801(in pixels)(2D)(EPE)
* Weakly[5]:ECCV'18, 0.994(3D PCK:0~50%)
### Dexter+Object (DO)
* Umar[4]:ECCV'18, 0.71(3D PCK:100%)
* DHPE[1]:CVPR'19, 0.65 (3D PCK:0~100%)
* GANerated[3]:CVPR'18, 0.64(in pixels) (2D PCK:0~30%)
* CPM[2]:arXiv'18, 14.593(in pixels)(2D)(EPE)
## Results
### Results on RHD dataset
|     Methods     |   3D PCK   |              EPE               |
| :-------------: | :--------: | :----------------------------: |
|     DHPE[1]     | 0.926(50%) |               -                |
|     CPM[2]      |            |   5.223(pixel)(2D)(320,320)    |
| GAN(Mueller)[3] |            |                                |
|     Umar[4]     |            | 3.57(pixel)(2D)  13.41(mm)(3D) |
| Weakly(cai)[5]  | 0.887(50%) |                                |
|     dVAE[6]     | 0.849(50%) |         19.95(mm)(3D)          |
|     Zimm[7]     | 0.675(50%) |                                |
|    Spurr[8]     | 0.849(50%) |         21.15(mm)(3D)          |
|     HAMR[9]     | 0.901(50%) |                                |
|    Graph[10]    | 0.92(50%)  |                                |
### Results on SHD dataset
|     Methods     |   3D PCK   |            EPE            |
| :-------------: | :--------: | :-----------------------: |
|     DHPE[1]     | 0.995(50%) |             -             |
|     CPM[2]      |     -      | 5.801(pixel)(2D)(640,480) |
| GAN(Mueller)[3] | 0.965(50%) |                           |
|     Umar[4]     | 0.994(50%) |                           |
| Weakly(cai)[5]  | 0.994(50%) |                           |
|     dVAE[6]     | 0.991(50%) |       8.66(mm)(3D)        |
|     Zimm[7]     | 0.986(50%) |                           |
|    Spurr[8]     | 0.983(50%) |         9.49(mm)          |
|     HAMR[9]     | 0.995(50%) |                           |
|    Graph[10]    | 0.998(50%) |         6.37(mm)          |
| ADnane[11]      |            | 9.76(mm) |
### Results on DO dataset
| Methods |   3D PCK   |          EPE           |
| :-----: | :--------: | :--------------------: |
| DHPE[1] | 0.65(100%) |           -            |
| CPM[2]  |            | 14.593(pixel)(2D)(640,320) |
| GANerated[3] | 0.56(100%) |                       |
| Umar[4] | 0.71(100%) |                       |
| Zimm[7] |  | 34.75 |
| Spurr[8] |                      | 40.20 |
| HAMR[9] | 0.82(50%)(without O) |                       |
| ADnane[11] |  | 25.53(mm) |
<a href="#evaluations-on-hand-pose-estimation with RGB">[back to top]</a>
## Reference
* [1] Pushing the Envelope for RGB-based Dense 3D Hand Pose Estimation via Neural Rendering, Seungryul Baek, Kwang In Kim, Tae-Kyun Kim, CVPR 2019.
* [2] RGB-based 3D Hand Pose Estimation via Privileged Learning with Depth Images, Shanxin Yuan, Bjorn Stenger, Tae-Kyun Kim, arXiv 2018.
* [3] GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB, Franziska Mueller, Florian Bernard, Oleksandr Sotnychenko, Dushyant Mehta, Srinath Sridhar, Dan Casas, Christian Theobalt, CVPR 2018.
* [4] Hand Pose Estimation via Latent 2.5D Heatmap Regression, Umar Iqbal, Pavlo Molchanov, Thomas Breuel, Juergen Gall, Jan Kautz, ECCV 2018.
* [5] Weakly-supervised 3D Hand Pose Estimation from Monocular RGB Images,Yujun Cai, Liuhao Ge, Jianfei Cai, Junsong Yuan, ECCV 2018.
* [6] Disentangling Latent Hands for Image Synthesis and Pose Estimation,Linlin Yang, Angela Yao, CVPR 2019.
* [7] Learning to Estimate 3D Hand Pose from Single RGB Images,Christian Zimmermann, Thomas Brox, ICCV 2017.
* [8] Cross-modal Deep Variational Hand Pose Estimation,Adrian Spurr, Jie Song, Seonwook Park, Otmar Hilliges, CVPR 2018.
* [9] End-to-end Global to Local CNN Learning for Hand Pose Recovery in Depth data, Meysam Madadi, Sergio Escalera, Xavier Baro, Jordi Gonzalez,arXiv 2019.
* [10] 3D Hand Shape and Pose Estimation from a Single RGB Image, Liuhao Ge, Zhou Ren, Yuncheng Li, Zehao Xue, Yingying Wang, Jianfei Cai, Junsong Yuan,CVPR 2019.
* [11] 3D Hand Shape and Pose from Images in the Wild,Adnane Boukhayma, Rodrigo de Bem, Philip H.S. Torr,CVPR 2019.


