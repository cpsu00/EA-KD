
## Code for EA-KD: Entropy-based Adaptive Knowledge Distillation

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>).

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
pip3 install -r requirements.txt
python3 setup.py develop
```

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

### Training on CIFAR100
EA-KD
```
python3 tools/train.py --cfg configs/cifar100/EA/kd.yaml
```
EA-ReviewKD
```
python3 tools/train.py --cfg configs/cifar100/EA/reviewkd.yaml
```
EA-DKD
```
python3 tools/train.py --cfg configs/cifar100/EA/dkd.yaml
```
EA-MLD
```
python3 tools/train_mld.py --cfg configs/cifar100/EA/mld.yaml
```
EA-MLD+LS
```
python3 tools/train_mld.py --cfg configs/cifar100/EA/mld+ls.yaml
```

### Training on ImageNet
EA-KD
```
python3 tools/train.py --cfg configs/imagenet/r34_r18/EA/kd.yaml
```
EA-DKD
```
python3 tools/train.py --cfg configs/imagenet/r34_r18/EA/dkd.yaml
```

Code for EA-CTKD and EA-FCFD, along with training scripts for Tiny-ImageNet and LLM distillation, will be made publicly available in the final version.

## Acknowledgement
We extend our sincere thanks to the contributors of [DKD(mdistiller)](<https://github.com/megvii-research/mdistiller>), [MLD](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation.git>), and [LS](<https://github.com/sunshangquan/logit-standardization-KD>) for their invaluable work, which has laid the foundation for our code.