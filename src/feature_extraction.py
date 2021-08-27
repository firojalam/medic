import os
import argparse

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

import model_utils
import dataset
import train_utils


parser = argparse.ArgumentParser(description='Extract features from pretrained model')
parser.add_argument('--train', default='data/train.tsv', type=str, help='path to train image files/labels')
parser.add_argument('--dev', default='data/dev.tsv', type=str, help='path to dev image files/labels')
parser.add_argument('--test', default='data/test.tsv', type=str, help='path to test image files/labels')
parser.add_argument('--sep', default='\t', type=str, help='column separator used in csv(default: "\t")')
parser.add_argument('--data-dir', default='data/images', type=str, help='root directory of images')
parser.add_argument('--best-state-path', default='models/best.pth', type=str, help='path to best state checkpoint')
parser.add_argument('--out-dir', default='features/image_net', type=str, help='output directory to save features')
parser.add_argument('--arch', default='efficientnet-b1', type=str,
                    help='model architecture, only efficientnet-b1 supported')
parser.add_argument('--num-classes', default=1000, type=int, help='Number of classes in pretrained model')


def extract_feature(model, dataloader, device, file_name):
    model.eval()
    pool = torch.nn.AdaptiveAvgPool2d(1)

    features = None
    targets = None
    idx = 0
    with torch.no_grad():
        for _, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = inputs.to(device)
            outputs = model.extract_features(inputs)
            outputs = pool(outputs)
            outputs = outputs.flatten(start_dim=1)
            outputs = outputs.cpu().numpy()

            if features is None:
                features = np.zeros((len(dataloader), outputs.shape[1]))
                targets = np.zeros(len(dataloader), dtype=np.int)

            features[idx: idx + outputs.shape[0]] = outputs
            targets[idx: idx + outputs.shape[0]] = labels

            idx += outputs.shape[0]
    np.savez(file_name, features=features, labels=targets)


def main():
    use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    print("Using device: {}".format(use_gpu), flush=True)
    args = parser.parse_args()
    print(args, flush=True)

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    sep = args.sep
    data_dir = args.data_dir
    best_state_path = args.best_state_path
    out_dir = args.out_dir

    model_name = args.arch
    num_classes = args.num_classes

    model, resize_image_size, input_size = model_utils.initialize_model(model_name, num_classes, keep_frozen=True,
                                                                        use_pretrained=True)

    data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_image_size, resize_image_size)),
            torchvision.transforms.CenterCrop((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Initializing Datasets and Dataloaders...", flush=True)
    image_datasets = {
        'train': dataset.SingleTaskDataset(train_path, sep, data_dir, data_transform),
        'val': dataset.SingleTaskDataset(dev_path, sep, data_dir, data_transform),
        'test': dataset.SingleTaskDataset(test_path, sep, data_dir, data_transform)}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=2)
                        for x in ['train', 'val', 'test']}

    model = model.to(device)

    model = train_utils.load_model_weight(best_state_path, model, device=device)

    os.makedirs(out_dir, exist_ok=True)
    for split in dataloaders_dict:
        file_name = os.path.join(out_dir, split + '.npz')
        extract_feature(model, dataloaders_dict[split], device, file_name)


if __name__ == "__main__":
    main()
