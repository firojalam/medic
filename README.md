# Multi-Task Learning for Disaster Image Classification using MEDIC

Recent research in disaster informatics demonstrates a practical and important use case of artificial intelligence to save human lives and sufferings during post-natural disasters based on social media contents (text and images). While notable progress has been made using texts, research on exploiting the images remains relatively under-explored. To advance the image-based approach, we propose MEDIC (available at: [https://crisisnlp.qcri.org/medic/](https://crisisnlp.qcri.org/medic/)), which is the largest social media image classification dataset for humanitarian response consisting of 71,198 images to address four different tasks in a multi-task learning setup. This is the first dataset of its kind: social media image, disaster response, and multi-task learning research. An important property of this dataset is its high potential to contribute research on multi-task learning, which recently receives much interest from the machine learning community and has shown remarkable results in terms of memory, inference speed, performance, and generalization capability. Therefore, the proposed dataset is an important resource for advancing image-based disaster management and multi-task machine learning research.

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
