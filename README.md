# RSDformer: Learning An Effective Transformer for Remote Sensing Satellite Image Dehazing

[![GoogleDrive](https://img.shields.io/badge/Data-GoogleDrive-brightgreen)](https://drive.google.com/drive/folders/1KRR_L276nviPT9JFPL9zfBiZVKJO6dM1?usp=drive_link)
[![BaiduPan](https://img.shields.io/badge/Data-BaiduPan-brightgreen)](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t)

> **Abstract:** 
The existing remote sensing (RS) image dehazing methods based on deep learning have sought help from the convolutional frameworks.
Nevertheless, the inherent limitations of convolution, {\em i.e.,} local receptive fields and independent input elements, curtail the network to learn the long-range dependencies and non-uniform distributions. 
To this end, we design an effective RS image dehazing Transformer architecture, denoted as RSDformer.
Firstly, given the irregular shapes and non-uniform distributions of haze in RS images, capturing both local and non-local features is crucial for RS image dehazing models.
Hence, we propose a detail-compensated transposed attention to extract the global and local dependencies by across channels.
Secondly, for enhancing the ability to learn degraded features and better guide the restoration process, we develop a dual-frequency adaptive block with dynamic filters.
Finally, a dynamic gated fusion block is designed to achieve fuse and exchange features across different scales effectively.
In this way, the model exhibits robust capabilities to capture dependencies from both global and local areas, resulting in improving image content recovery.
Extensive experiments prove that the proposed method obtains more appealing performances against other competitive methods.
The source codes are available at https://github.com/MingTian99/RSDformer

![RSDformer](figs/arch.png)

## News

- **July 4, 2023:** Paper submitted. 
- **Sep 13, 2023:** The basic version is released, including codes, [pre-trained models on Sate 1k dataset](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t), and [the used dataset](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t).
- **Sep 14, 2023:** [RICE dataset](https://pan.baidu.com/s/1zbTBTys4VqL9CnJI0UFgoQ?pwd=7vj5) updated.
- **Sep 15, 2023:** The [visual results on Sate 1K](https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k) and [real-world dataset RSSD300](https://pan.baidu.com/s/1OZUWj8eo6EmP5Rh8DE1mrA?pwd=8ad5) are updated.


## Preparation

### Install

We test the code on PyTorch 1.9.1 + CUDA 11.1 + cuDNN 8.0.5.

1. Create a new conda environment
```
conda create -n RSDformer python=3.8
conda activate RSDformer 
```

2. Install dependencies
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install matplotlib scikit-image opencv-python numpy einops math natsort tqdm lpips time tensorboardX
```

### Download

You can download the pre-trained models and datasets on [GoogleDrive](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t) or [BaiduPan](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t) (pu2t).

Currently, we only provide the pre-trained model trained on the Sate 1K dataset and the used dataset (Sate 1K, RICE and RRSD300).  The pre-trained models trained on RICE will be updated as quickly as possible.

The final file path should be the same as the following:

```
┬─ pretrained_models
│   ├─ thin_haze.pth
│   ├─ moderate_haze.pth
│   ├─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ Sate_1K
    │├─ Haze1k_thick
    ││   ├─ train
    ││   │   ├─ input
    ││   │   │   └─ ... (image filename)
    ││   │   └─ target
    ││   │       └─ ... (corresponds to the former)
    ││   └─ test
    ││       └─ ...
    │└────  ... (dataset name)
    │
    │
    └─ ... (dataset name)

```
### Training, Testing and Evaluation

### Train
The training code will be released after the paper is accepted.
You should change the path to yours in the `Train.py` file.  Then run the following script to test the trained model:

```sh
python Train.py
```

### Test
You should change the path to yours in the `Test.py` file.  Then run the following script to test the trained model:

```sh
python Test.py
```


### Evaluation
You should change the path to yours in the `Dataload.py` file.  Then run the following script to test the trained model:

```sh
python PSNR_SSIM.py
```
It is recommended that you can download the visual deraining results and retest the quantitative results on your own device and environment.

### Visual Results

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Thin Haze</th>
    <th>Moderate Haze</th>
    <th>Thin Haze</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k">Download</a> </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td>Google Drive</td>
    <td> <a href="https://drive.google.com/drive/folders/16UHn439SMJp0ZnDt_yoYc96ypsY7FN7n?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/16UHn439SMJp0ZnDt_yoYc96ypsY7FN7n?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/16UHn439SMJp0ZnDt_yoYc96ypsY7FN7n?usp=drive_link">Download</a> </td>
  </tr>
</tbody>
</table>

## Notes

1. Send e-mail to songtienyu@163.com if you have critical issues to be addressed.
2. Please note that there exists the slight gap in the final version due to errors caused by different testing devices and environments. 
3. Because the synthetic dataset is not realistic enough, the trained models may not work well on real hazy images.


## Acknowledgment

This code is based on the [Restormer](https://github.com/swz30/Restormer). Thanks for their awesome work.
This real-world dataset RRSD300 is collected from [RSHazeNet](https://github.com/chdwyb/RSHazeNet). Thanks for their awesome work.
