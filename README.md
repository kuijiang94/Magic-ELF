# Magic-ELF
Magic ELF: Image Deraining Meets Association Learning and Transformer (ACMMM2022)

[Kui Jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl), [Zhongyuan Wang](https://dblp.org/pid/84/6394.html), [Chen Chen](https://scholar.google.com/citations?user=TuEwcZ0AAAAJ&hl=zh-CN), [Zheng Wang](https://scholar.google.com/citations?user=-WHTbpUAAAAJ&hl=zh-CN), [Laizhong Cui](https://scholar.google.com/citations?hl=zh-CN&user=cb8kYbUAAAAJ), and [Chia-Wen Lin](https://scholar.google.com/citations?user=fXN3dl0AAAAJ&hl=zh-CN)

**Paper**: [Magic ELF: Image Deraining Meets Association Learning and Transformer](https://arxiv.org/abs/2207.10455)


## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## Quick Test

To test the pre-trained deraining model on your own images, run 
```
python test.py  
```

## Training and Evaluation

Training and Testing codes for deraining are provided in their respective directories.

## Results
Experiments are performed for different image processing tasks including, image deraining, image dehazing and low-light image enhancement.

## Acknowledgement
Code borrows from [MPRNet](https://github.com/swz30/MPRNet) by [Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en). Thanks for sharing !

## Citation
If you use PCNet, please consider citing:

    @article{jiangpcnet,
        title={Magic ELF: Image Deraining Meets Association Learning and Transformer},
        author={Kui Jiang and Zhongyuan Wang and Chen Chen and Zheng Wang and Laizhong Cui and Chia-Wen Lin},
        journal={ACMMM}, 
        year={2022}
    }

## Contact
Should you have any question, please contact Kui Jiang (kuijiang@whu.edu.cn)
