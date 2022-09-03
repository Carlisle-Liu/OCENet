# Online Confidence Estimation - OCENet
<!-- <p align="center"><img src="introduction_figure.png" alt="introduction_figure" width="90%"></p> -->
![alt_text](./Network-Overview_v2.jpg)

The implementation of **Modeling Aleatoric Uncertainty for Camouflaged Object Detection**, Jiawei Liu, Jing Zhang and Nick Barnes, [[Paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Liu_Modeling_Aleatoric_Uncertainty_for_Camouflaged_Object_Detection_WACV_2022_paper.pdf).

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@InProceedings{Liu_2022_WACV,
    author    = {Liu, Jiawei and Zhang, Jing and Barnes, Nick},
    title     = {Modeling Aleatoric Uncertainty for Camouflaged Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1445-1454}
}
```

## Requirements
- Python 3.6
- pytorch 1.3.0, torchvision 0.4.1
- CUDA 10.1
- 2 x 2080Ti GPU


Download COD10K training set through [Google Drive link](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1XL6OjpDF-MVnXOY6-bdaBg) with the fetch code:djq2.


## Usage
1. Training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=3000 train.py
```

2. Inference


Download the weight of CODNet and OCENet from the [[Google Drive]](https://drive.google.com/drive/folders/10wGvx2M_LRrpL_gQu4He9U7kk12S64vM?usp=sharing)
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

3. Evaluation

Please use the [SOD evaluation toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox.git) provided by [jiwei0921](https://github.com/jiwei0921).


## Results
The prediction results can be downloaded from [[Google Drive]](https://drive.google.com/drive/folders/1CEA6GUtV0nwlamntA25LmCNI-KCXg4WL?usp=sharing)
