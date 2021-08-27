import os
import argparse
import json
import copy
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt

import model_utils
import dataset
import train_utils


parser = argparse.ArgumentParser(description='Transfer learning for disaster image classification')
parser.add_argument('--name', default='hum_resnet18', type=str, help='name of the run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--task-name', default='humanitarian', type=str, help='name of the task')
parser.add_argument('--train', default='data/train.tsv', type=str, help='path to train image files/labels')
parser.add_argument('--dev', default='data/dev.tsv', type=str, help='path to dev image files/labels')
parser.add_argument('--test', default='data/test.tsv', type=str, help='path to test image files/labels')
parser.add_argument('--out-file', default='out/results.json', type=str, help='path to output file')
parser.add_argument('--sep', default='\t', type=str, help='column separator used in csv(default: "\t")')
parser.add_argument('--data-dir', default='data/images', type=str, help='root directory of images')
parser.add_argument('--best-state-path', default='models/best.pth', type=str, help='path to best state checkpoint')
parser.add_argument('--fig-dir', default='out/figures', type=str, help='directory path for output figures')
parser.add_argument('--checkpoint-dir', default='out/models', type=str, help='directory for output models/states')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='model architecture [resnet18, resnet50, resnet101, alexnet, vgg, vgg16, squeezenet, densenet,'
                         'inception, efficientnet-b1, efficientnet-b7] (default: resnet18)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--learning-rate', default=1e-5, type=float, help='initial learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--num-epochs', default=50, type=int, help='number of epochs(default: 50)')
parser.add_argument('--use-rand-augment', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='use random augment or not')
parser.add_argument('--keep-frozen', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to keep feature layers frozen (i.e., only update classification layers weight)')
parser.add_argument('--rand-augment-n', default=2, type=int,
                    help='random augment parameter N or number of augmentations applied sequentially')
parser.add_argument('--rand-augment-m', default=9, type=int,
                    help='random augment parameter M or shared magnitude across all augmentation operations')


num_classes_tasks = {
    'damage_severity': 3,
    'informative': 2,
    'humanitarian': 4,
    'disaster_types': 7
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_plot(fig_dir, file_name, losses, accuracies):
    hist_val_acc = [h for h in accuracies["val"]]
    hist_val_loss = [h for h in losses["val"]]
    hist_train_acc = [h for h in accuracies["train"]]
    hist_train_loss = [h for h in losses["train"]]

    num_epochs = len(hist_val_acc)
    x_data = range(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    ax = plt.subplot2grid(loc=(0, 0), shape=(1, 1))

    axes, _ = train_utils.two_scales(ax, None, x_data, hist_train_loss, hist_val_loss, hist_train_acc, hist_val_acc)
    fig.tight_layout()
    plt.draw()
    plt.show()
    os.makedirs(fig_dir, exist_ok=True)
    out_file = os.path.join(fig_dir, file_name + ".png")
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(model, dataloaders, criterion, optimizer, scheduler, device=None, num_epochs=100, curr_epoch=0,
                curr_evals=(None, None), class_names=None, fig_dir="output/figures", checkpoint_dir="output/models",
                run_name="single-task", is_inception=False):

    since = datetime.now().replace(microsecond=0)

    phases = ['train', 'val']
    losses, accuracies = curr_evals
    if not losses:
        losses = {phase: [] for phase in phases}
    if not accuracies:
        accuracies = {phase: [] for phase in phases}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(curr_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]),
                                              desc="[{}/{}] {} Iteration".format(epoch, num_epochs-1, phase.upper())):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # noinspection PyTypeChecker
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()

            accuracies[phase].append(epoch_acc)
            losses[phase].append(epoch_loss)

            # print progress
            learning_rate = optimizer.param_groups[0]["lr"]
            print(' Loss: {:.4f} Acc: {:.4f} lr: {:.4E}'.format(epoch_loss, epoch_acc, learning_rate), flush=True)

            if phase == 'val':
                if isinstance(model, torch.nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                # check if current model yields better accuracy
                is_best = False
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    is_best = True

                # save states dictionary
                state = {
                    "epoch": epoch,
                    "lr": learning_rate,
                    "state_dict": model_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "losses": losses,
                    "accuracies": accuracies,
                    "class_names": class_names
                }

                filename = "{}_{}_{:.3f}_best_state.pth".format(run_name, epoch, epoch_acc)

                # train_utils.save_checkpoint(state, is_best, filename, checkpoint_dir) ## Firoj commented
                train_utils.save_best_checkpoint(state, is_best, filename, checkpoint_dir)
                save_plot(fig_dir, run_name, losses, accuracies)
                train_utils.save_json(fig_dir, run_name, losses, accuracies, None)

                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

        print(flush=True)

    time_elapsed = datetime.now().replace(microsecond=0) - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Training complete in {}'.format(time_elapsed, flush=True))
    print('Best Val Accuracy: {:.4f}'.format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, time_elapsed


def test_model(model, dataloaders, class_names, device=None):
    print("Testing the model on both data splits:", flush=True)
    model.eval()
    phases = ['train', 'val']

    all_phases_result = {}
    for phase in phases:
        preds_ = []
        labels_ = []
        since = datetime.now().replace(microsecond=0)
        with torch.no_grad():
            for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]),
                                              desc="{} Iteration".format(phase.upper())):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                preds_.extend(preds.cpu().numpy().tolist())
                labels_.extend(labels.data.cpu().numpy().tolist())
        time_elapsed = datetime.now().replace(microsecond=0) - since
        print('-' * 10, flush=True)
        print('Performance on {} set:'.format(phase.upper()), flush=True)
        acc = metrics.accuracy_score(labels_, preds_)
        class_report = metrics.classification_report(labels_, preds_, target_names=class_names, digits=3)
        conf_mat = metrics.confusion_matrix(labels_, preds_)
        print('Accuracy: {:.4f}'.format(acc), flush=True)
        print(class_report, flush=True)
        print(conf_mat, flush=True)
        print('-' * 10, flush=True)

        label_y = []
        label_pred = []
        for i in range(len(preds_)):
            label_y.append(class_names[labels_[i]])
            label_pred.append(class_names[preds_[i]])

        acc = metrics.accuracy_score(labels_, preds_)
        precision = metrics.precision_score(labels_, preds_, average="weighted")
        recall = metrics.recall_score(labels_, preds_, average="weighted")
        f1_score = metrics.f1_score(labels_, preds_, average="weighted")

        result = str("{0:.3f}".format(acc)) + "\t" + str(
            "{0:.3f}".format(precision)) + "\t" + str("{0:.3f}".format(recall)) + "\t" + str(
            "{0:.3f}".format(f1_score)) + "\n"

        print(result)
        conf_mat_str = train_utils.format_conf_mat(label_y, label_pred)
        agr_met = train_utils.compute_aggregate_scores(label_y, label_pred, class_names)
        phase_object = {
            "accuracy": acc,
            "results": result,
            "classification_report": class_report,
            "confusion_matrix": conf_mat,
            "conf_mat_str": conf_mat_str,
            "gold": label_y,
            "pred": label_pred,
            "agr_met": agr_met,
            "execution_time": str(time_elapsed)
        }
        all_phases_result[phase] = phase_object

    return all_phases_result


def test_model_save_results(outfile, sep, model, dataloader, test_images, class_names, device=None):
    print("Testing the model and saving the results:", flush=True)

    model.eval()
    probs_ = []
    preds_ = []
    labels_ = []

    running_corrects = 0
    count = 0
    since = datetime.now().replace(microsecond=0)
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, 1)
            probs, preds = torch.max(outputs, 1)
            preds_.extend(preds.cpu().numpy().tolist())
            probs_.extend(probs.cpu().numpy().tolist())
            labels_.extend(labels.data.cpu().numpy().tolist())
            # noinspection PyTypeChecker
            running_corrects += torch.sum(preds == labels.data)
            count += len(labels.data)
    # time_elapsed = time.time() - since
    time_elapsed = datetime.now().replace(microsecond=0) - since
    print('Test complete in {}'.format(time_elapsed, flush=True))
    # print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

    print('Performance results:', flush=True)
    acc = metrics.accuracy_score(labels_, preds_)
    print('Accuracy: {:.4f}'.format(acc), flush=True)
    class_report = metrics.classification_report(labels_, preds_, target_names=class_names, digits=3)
    conf_mat = metrics.confusion_matrix(labels_, preds_)
    print(class_report, flush=True)
    print(conf_mat, flush=True)
    print('-' * 10, flush=True)
    print(flush=True)

    precision = metrics.precision_score(labels_, preds_, average="weighted")
    recall = metrics.recall_score(labels_, preds_, average="weighted")
    f1_score = metrics.f1_score(labels_, preds_, average="weighted")

    result = str("{0:.3f}".format(acc)) + "\t" + str(
        "{0:.3f}".format(precision)) + "\t" + str("{0:.3f}".format(recall)) + "\t" + str(
        "{0:.3f}".format(f1_score)) + "\n"
    print(result)

    test_y = []
    test_pred = []
    dir_name = os.path.dirname(outfile)
    os.makedirs(dir_name, exist_ok=True)
    with open(outfile, 'w') as f:
        for i in range(len(preds_)):
            f.write('{}{}{}{}{:.4f}\n'.format(test_images.samples[i][0], sep, class_names[preds_[i]], sep, probs_[i]))
            test_y.append(class_names[labels_[i]])
            test_pred.append(class_names[preds_[i]])

    agr_met = train_utils.compute_aggregate_scores(test_y, test_pred, class_names)
    conf_mat_str = train_utils.format_conf_mat(test_y, test_pred)
    print(conf_mat_str)
    phase_object = {
        "accuracy": acc,
        "results": result,
        "classification_report": class_report,
        "confusion_matrix": conf_mat,
        "conf_mat_str": conf_mat_str,
        "gold": test_y,
        "pred": test_pred,
        "agr_met": agr_met,
        "execution_time": str(time_elapsed)
    }

    return phase_object


def main():
    use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    print("Using device: {}".format(use_gpu), flush=True)
    args = parser.parse_args()
    print(args, flush=True)

    set_seed(args.seed)
    train_path = args.train
    dev_path = args.dev
    sep = args.sep
    data_dir = args.data_dir
    best_state_path = args.best_state_path
    # target_names_file = args.target_names_file

    fig_dir = os.path.join(args.fig_dir, args.name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = args.arch
    batch_size = args.batch_size
    lr = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    num_classes = num_classes_tasks[args.task_name]

    model, resize_image_size, input_size = model_utils.initialize_model(model_name, num_classes,
                                                                        keep_frozen=args.keep_frozen,
                                                                        use_pretrained=True)

    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_image_size, resize_image_size)),
            torchvision.transforms.RandomCrop((input_size, input_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_image_size, resize_image_size)),
            torchvision.transforms.CenterCrop((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Add RandAugment
    if args.use_rand_augment:
        data_transforms['train'].transforms.insert(0, RandAugment(args.rand_augment_n, args.rand_augment_m))

    print("Initializing Datasets and Dataloaders...", flush=True)
    image_datasets = {
        'train': dataset.SingleTaskDataset(train_path, args.task_name, sep, data_dir, data_transforms['train']),
        'val': dataset.SingleTaskDataset(dev_path, args.task_name, sep, data_dir, data_transforms['val'])}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                                       num_workers=2) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    model, optimizer, scheduler, curr_epoch, curr_evals = \
        train_utils.load_checkpoint(best_state_path, model=model, optimizer=optimizer, scheduler=scheduler,
                                    device=device)

    # check if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()), flush=True)
        model = torch.nn.DataParallel(model)

    model, time_elapsed = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, device=device,
                                      num_epochs=num_epochs, curr_epoch=curr_epoch, curr_evals=curr_evals,
                                      class_names=class_names, fig_dir=fig_dir, checkpoint_dir=checkpoint_dir,
                                      run_name=args.name, is_inception=(model_name == "inception"))

    all_phases_result = test_model(model, dataloaders_dict, class_names=class_names, device=device)
    all_phases_result['train']['execution_time'] = str(time_elapsed)

    if args.test:
        base = os.path.basename(args.out_file)
        base_name = os.path.splitext(base)[0]
        dir_name = os.path.dirname(args.out_file)
        out_file_name = os.path.join(dir_name, base_name + '_labels.txt')
        test_transforms = data_transforms['val']
        test_dataset = dataset.SingleTaskDataset(args.test, args.task_name, sep, data_dir, test_transforms)
        print("There are {} test images.".format(len(test_dataset)))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        results = test_model_save_results(out_file_name, sep, model, test_dataloader, test_dataset, class_names,
                                          device=device)

        all_phases_result['test'] = train_utils.convert_to_json(results)
    dir_name = os.path.dirname(args.out_file)
    os.makedirs(dir_name, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(train_utils.convert_to_json(all_phases_result), f)


if __name__ == "__main__":
    main()
