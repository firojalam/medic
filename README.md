# Multi-Task Learning for Disaster Image Classification using MEDIC

This repository contains scripts and associate resources to train model using MEDIC dataset. The MEDIC is the largest multi-task learning disaster related dataset, which is an extended version of the crisis image benchmark dataset. It consists data from several data sources such as CrisisMMD, data from AIDR and Damage Multimodal Dataset (DMD). The dataset contains 71,198 images.

## Download
To download the dataset: [https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz](https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz)

More details about the dataset: https://crisisnlp.qcri.org/medic/


## Train
To finetune a ImageNet pretrained resnet18 model run  
`source scripts/train_multitask_resnet.sh`  
Change corresponding parameters to train for other settings.


## Publication:
Please cite the following paper.

1. *Firoj Alam, Tanvirul Alam, Md. Arid Hasan, Abul Hasnat, Muhammad Imran, Ferda Ofli, MEDIC: A Multi-Task Learning Dataset for Disaster Image Classification, 2021. arXiv preprint arXiv:2108.12828. [download](https://arxiv.org/abs/2108.12828).*
2. *Please also cite the papers mentioned in [https://crisisnlp.qcri.org/medic/](https://crisisnlp.qcri.org/medic/) if you use the MEDIC dataset.*

```bib
@article{alam2021medic,
      title={MEDIC: A Multi-Task Learning Dataset for Disaster Image Classification},
      author={Firoj Alam and Tanvirul Alam and Md. Arid Hasan and Abul Hasnat and Muhammad Imran and Ferda Ofli},
      year={2021},
      eprint={2108.12828},
      archivePrefix={arXiv},
      journal={arXiv preprint arXiv:2108.12828},
      primaryClass={cs.CV}
}
```

## Credits

## Licensing
The MEDIC dataset is published under CC BY-NC-SA 4.0 license, which means everyone can use this dataset for non-commercial research purpose: https://creativecommons.org/licenses/by-nc/4.0/.
