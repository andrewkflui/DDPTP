# Data-driven Pedestrian Trajectory Prediction (DDPTP)

Copyright (C) 2022 - Andrew Kwok-Fai Lui, Yin-Hei Chan
Hong Kong Metropolitan University

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see http://www.gnu.org/licenses/.

## Introduction

This repository contains prototype implementation of the Data-driven Pedestrian Trajectory Prediction (DDPTP) method for training microscopic pedestrian trajectory prediction models. The algorithm is described in the following paper presented at the IEEE BigData 2021 international conference.

> Lui, A.K.F., Chan, Y.H. and Leung, M.F., 2021, December. Modelling of Destinations for Data-driven Pedestrian Trajectory Prediction in Public Buildings. In 2021 IEEE International Conference on Big Data (Big Data) (pp. 1709-1717). IEEE.

The variants discussed in the paper and the baseline models, most notably a re-implementation of the [PoPPL](https://github.com/xuehaouwa/poppl) (Xue et al., 2020) model, are also included in the repository.

DDPTP was designed and implemented by Andrew Kwok-Fai LUI and Yin-Hei Chan

## Installation and Running

### Datasets
The New York Grand Central (NYGC) dataset and the ATC dataset.
> Yi, S., Li, H. and Wang, X., 2015. Understanding pedestrian behaviors from stationary crowd groups. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3488-3496).

> Brščić, D., Kanda, T., Ikeda, T. and Miyashita, T., 2013. Person tracking in large public spaces using 3-D range sensors. IEEE Transactions on Human-Machine Systems, 43(6), pp.522-534.

### Instruction

1. Download data from https://drive.google.com/file/d/121BDgocp2-pO6GERNICXxbowYHjaiO9G/view?usp=sharing
2. Unzip to ./datas
3. Execute the following
```
run model trainer
run results
```
More details to be published later.

### Sample Trajectory Prediction from DDPTP

The NYGC dataset. These two examples show the importance of the speed and direction in the velocity feature. (Left) Without the velocity feature, overshoot at the end of trajecotry can be seen. (Right) With the velocity feature, the model learned to better move with the correct speed and reached the end point (in a red circle).

![Picture 1](https://user-images.githubusercontent.com/8808539/198897421-bd5fdf69-a93c-4191-8fee-ba350a5bd831.png)
![Picture2png](https://user-images.githubusercontent.com/8808539/198897423-5b5040df-af14-4889-9bc1-2f3ae1802833.png)

The ATC dataset. The predicted trajectory using DDPTP-no-attention-ID with 4 destinations and 4 intermediate destinations (only one of them is relevant to this trajectory.
<img width="206" alt="Picture1" src="https://user-images.githubusercontent.com/8808539/198897485-c2b1f70b-ee2e-40e6-8501-95110af24e9c.png">

## Reference
[Alternative repository](https://github.com/YinHei-Chan/DDPTP)

## Acknowledgement
The work described in this paper was fully supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (UGC/FDS16/E12/20).
