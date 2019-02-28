# Repository for experiments on NSML

NOTE: NFS에서 실행하시는 경우 해당 실험의 로그파일 및 체크포인트들이 ./experiments 안에 저장됩니다.

### Usage
* `--mode`: 실험의 종류를 지정합니다.
  * `train`: training set에 대한 training 및 adversarial training
  * `infer`: validation set에서 validate
  * `attack`: adversarial attack 알고리즘과 defense 알고리즘 성능 측정
  * `print_accuracy`: 다양한 lambda값에 대해 validate
  * `occlusion`: center occlusion에 대한 성능 측정

* `--dataset`: 이미지 데이터셋을 지정합니다. 네트워크에 따라 호환되지 않는 경우도 있습니다.
  * `MNIST`
  * `CIFAR10`
  * `CIFAR100`
  * `TinyImageNet`
  * `ImageNet`
* `--data_dir`: NFS를 이용하시는 경우 데이터셋의 위치를 지정해주어야 합니다. NFS에서 ImageNet의 경우 `data/imagenet/train`.

* `--model`: 실험에 사용할 모델의 종류입니다. submodules/models/\_\_init\_\_.py에 이용가능한 모델들이 적혀있습니다.
  * `ResNet18` (CIFAR용)
  * `resnet18`, `resnet101`
  * `densenet121`
  * `vgg19`, `vgg19_bn`
  * `ace`
  * `ace_cifar`, `ace_cifar_random`
  * `ace_resnet101`, `ace_resnet101_random`
* `--pretrained`: pretrained model을 불러오기 위한 flag입니다. pytorch official model들의 경우 default로 pretrained입니다.
* `--ckpt_dir`: pretrained model을 이용하시는 경우 checkpoint의 위치를 지정해주어야 합니다. NFS에서 ImageNet의 경우 `models/imagenet`.
* `--ckpt_name`: checkpoint 파일의 이름입니다.
* `--fine_tune`: `--mode`가 `train`일 때, 모델이 ace인 경우 뒤에 부착된 base classifier까지 트레이닝하기 위한 flag입니다.

* `--attack`: adversarial attack 알고리즘의 종류입니다. submodules/attacks/\_\_init\_\_.py에 이용가능한 알고리즘들이 적혀있습니다.
  * `fgsm`
  * `onepixel`
  * `jsma`
  * `deepfool_l2`
  * `cwl2`
  * `pgd_l2_10`, `pgd_l2_20`, `pgd_l2_100`, `pgd_l2_1000`
* `--defense`: adversarial defense 알고리즘의 종류입니다. submodules/defenses/\_\_init\_\_.py에 이용가능한 알고리즘들이 적혀있습니다.
  * `pixeldeflection`
  * `randomization`
  * `regiondefense`
* `--source`: `--mode`가 `attack`인 경우에 한해, adversarial attack을 생성할 모델을 지정합니다. 이 경우 attacked image는 `--source`로부터 generate되고, attack에 대한 validation은 `--model`에 대해 이루어집니다.

### Examples

#### CIFAR10

CIFAR10에서 pgd iter 100으로 resnet18을 adversarial training
```
nsml run -d cifar10_ace -a "--dataset CIFAR10 --model ResNet18 --mode train --adv_ratio 0.3 --attack pgd_l2_100"
```

CIFAR10에서 fgsm으로 PRM + resnet18 중 PRM만 adversarial training  
이 경우 resnet18은 pretrained여야 하므로 체크포인트 위치를 지정해주어야 합니다. ckpt_name을 지정해주지 않으면 불러오고자 하는 모델명으로 default입니다. 단 PRM에 사용된 autoencoder의 경우에도 자동으로 체크포인트를 불러오고 있으므로(죄송합니다) 학습이 안 되었더라도 체크포인트 파일이 필요합니다.  
**여러 체크포인트를 동시에 불러와야 하는 경우, 체크포인트 이름을 각 모델명으로 저장해두어야 합니다.**
```
nsml run -d cifar10_ace -a "--dataset CIFAR10 --model ace_cifar --pretrained --ckpt_dir 체크포인트 위치 --mode train --adv_ratio 0.3 --attack fgsm"
```

CIFAR10에서 PRM + resnet18 모두 fine_tune
```
nsml run -d cifar10_ace -a "--dataset CIFAR10 --model ace_cifar --pretrained --fine_tune --ckpt_dir 체크포인트 위치 --mode train"
```

CIFAR10에서 PRM + resnet18에 대해 gray attack scenario
```
nsml run -d cifar10_ace -a "--dataset CIFAR10 --model ace_cifar_random --pretrained --ckpt_dir 체크포인트 위치 --mode attack --attack cwl2 --source ace_cifar --log_step 1"
```

#### ImageNet

ImageNet에서 full prm configuration에 대한 EOT
```
nsml run -d sanghyuk-nfs -a "--dataset ImageNet --data_dir data/imagenet/train --model ace --pretrained --ckpt_dir models/imagenet --mode attack --attack pgd_l2_1000 --batch_size 4 --log_step 1" --nfs-output
```

ImageNet에서 PRM + resnet101에 대해 gray attack scenario
```
nsml run -d sanghyuk-nfs -a "--dataset ImageNet --data_dir data/imagenet/train --model ace_resnet101_random --pretrained --ckpt_dir models/imagenet --source ace_resnet101 --mode attack --attack pgd_l2_1000 --batch_size 1 --log_step 1" --nfs-output
```

ImageNet에서 다른 defense ablation
```
nsml run -d ILSVRC2015 -a "--dataset ImageNet --model resnet101 --pretrained --mode attack --attack fgsm --defense pixeldeflection --log_step 1"
```
  
./scripts에 더 많은 예시가 있습니다.
