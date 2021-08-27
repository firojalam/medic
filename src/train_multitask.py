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


parser = argparse.ArgumentParser(description='Multitask learning for disaster image classification')
parser.add_argument('--name', default='multitask_resnet', type=str, help='name of the run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
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
                         'efficientnet-b1, efficientnet-b7] (default: resnet18)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--learning-rate', default=1e-5, type=float, help='initial learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--num-epochs', default=50, type=int, help='number of epochs(default: 50)')
parser.add_argument('--use-rand-augment', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='use random augment or not')
parser.add_argument('--rand-augment-n', default=2, type=int,
                    help='random augment parameter N or number of augmentations applied sequentially')
parser.add_argument('--rand-augment-m', default=9, type=int,
                    help='random augment parameter M or shared magnitude across all augmentation operations')
parser.add_argument('--task-names', default='damage_severity,informative,humanitarian,disaster_types', type=str,
                    help='names of the task separated y comma')


# number of classes and the tasks considered
# task name should be as in the csv files
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


def save_plot(fig_dir, file_name, losses, accuracies, task_names):
    num_tasks = len(losses["train"][0])
    hist_val_acc = accuracies["val"]
    hist_val_loss = losses["val"]
    hist_train_acc = accuracies["train"]
    hist_train_loss = losses["train"]

    num_epochs = len(hist_val_acc)
    x_data = range(1, num_epochs + 1)

    locs = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for idx in range(num_tasks):
        task_train_loss = [x[idx] for x in hist_train_loss]
        task_val_loss = [x[idx] for x in hist_val_loss]
        task_train_acc = [x[idx] for x in hist_train_acc]
        task_val_acc = [x[idx] for x in hist_val_acc]
        ax = plt.subplot2grid(shape=(2, 6), loc=locs[idx], colspan=2)
        axes[idx//3][idx % 3], _ = train_utils.two_scales(ax, task_names[idx], x_data, task_train_loss, task_val_loss,
                                                          task_train_acc, task_val_acc)
    fig.tight_layout()
    plt.draw()
    plt.show()
    os.makedirs(fig_dir, exist_ok=True)
    out_file = os.path.join(fig_dir, file_name + ".png")
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_classes, task_names, device=None,
                num_epochs=100, curr_epoch=0, curr_evals=(None, None), class_names=None, fig_dir="out/figures",
                checkpoint_dir="out/models", run_name="multitask_train", is_inception=False):

    if is_inception:
        raise ValueError("Inception not supported!")

    since = datetime.now().replace(microsecond=0)
    phases = ['train', 'val']
    losses, accuracies = curr_evals
    num_tasks = len(num_classes)
    # 1 + num_tasks => save values for each task and overall average
    if not losses:
        losses = {phase: [] * (1 + num_tasks) for phase in phases}
    if not accuracies:
        accuracies = {phase: [] * (1 + num_tasks) for phase in phases}

    best_model_wts = copy.deepcopy(model.state_dict())
    # best average accuracy across tasks
    best_acc = 0.0
    task_names.append('average')
    for epoch in range(curr_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 20, flush=True)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            print('-' * 5 + phase + '-' * 5, flush=True)
            running_loss = [0.0] * num_tasks  # total loss for each task
            running_total = [0] * num_tasks  # total samples for each task
            running_corrects = [0] * num_tasks

            # Iterate over data.
            for _, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]),
                                            desc="[{}/{}] {} Iteration".format(epoch, num_epochs - 1, phase.upper())):
                inputs = inputs.to(device)
                labels = [labels[i].to(device) for i in range(len(labels))]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = torch.tensor(0., requires_grad=True, dtype=torch.float, device=device)
                    loss.to(device)
                    idx_start = 0
                    for idx in range(num_tasks):
                        output_idx = outputs[:, idx_start:idx_start + num_classes[idx]]
                        label_idx = labels[idx]
                        label_idx.to(device)
                        valid_idx = torch.nonzero(1 * (label_idx != -1)).view(-1)
                        if len(valid_idx) == 0:
                            continue
                        idx_start += num_classes[idx]
                        loss_idx = criterion(output_idx[valid_idx], label_idx[valid_idx])
                        loss_idx.to(device)
                        loss = loss + loss_idx
                        loss.to(device)
                        running_loss[idx] += loss_idx.item() * len(valid_idx)
                        running_total[idx] += len(valid_idx)
                        _, preds = torch.max(output_idx[valid_idx], 1)
                        # noinspection PyTypeChecker
                        running_corrects[idx] += torch.sum(preds == label_idx[valid_idx].data).item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = sum(running_loss) / sum(running_total)
            task_loss = [running_loss[k] / running_total[k] for k in range(num_tasks)]
            task_acc = [running_corrects[k] / running_total[k] for k in range(num_tasks)]
            epoch_acc = sum(task_acc) / len(task_acc)

            task_loss.append(epoch_loss)
            task_acc.append(epoch_acc)

            accuracies[phase].append(task_acc)
            losses[phase].append(task_loss)

            # print progress
            learning_rate = optimizer.param_groups[0]["lr"]
            print(' Loss: {:.4f} Acc: {:.4f} lr: {:.4E}'.format(epoch_loss, epoch_acc, learning_rate),
                  flush=True)

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
                # filename = "{}_{:.3f}_best_state.pth".format(train_file, epoch_acc)

                # train_utils.save_checkpoint(state, is_best, filename, checkpoint_dir)

                train_utils.save_best_checkpoint(state, is_best, filename, checkpoint_dir)
                save_plot(fig_dir, run_name, losses, accuracies, task_names)
                train_utils.save_json(fig_dir, run_name, losses, accuracies, task_names)

                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
    del task_names[-1]

    time_elapsed = datetime.now().replace(microsecond=0) - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Training complete in {}'.format(time_elapsed, flush=True))
    print('Best Val Accuracy: {:.4f}'.format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, time_elapsed


def test_model(model, dataloaders, num_classes, tasks, class_names=None, device=None):
    print("Testing the model on both data splits:", flush=True)
    model.eval()
    phases = ['train', 'val']

    all_phases_result = {}
    for phase in phases:
        num_tasks = len(num_classes)
        preds_ = [[] for _ in range(num_tasks)]
        labels_ = [[] for _ in range(num_tasks)]
        since = datetime.now().replace(microsecond=0)
        with torch.no_grad():
            for _, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]),
                                            desc="{} Iteration".format(phase.upper())):
                inputs = inputs.to(device)
                labels = [labels[i].to(device) for i in range(len(labels))]

                outputs = model(inputs)

                idx_start = 0
                for idx in range(num_tasks):
                    output_idx = outputs[:, idx_start:idx_start + num_classes[idx]]
                    label_idx = labels[idx]
                    valid_idx = torch.nonzero(1 * (label_idx != -1)).view(-1)
                    if len(valid_idx) == 0:
                        continue
                    idx_start += num_classes[idx]
                    _, preds = torch.max(output_idx[valid_idx], 1)

                    preds_[idx].extend(preds.cpu().numpy().tolist())
                    labels_[idx].extend(label_idx[valid_idx].data.cpu().numpy().tolist())
        time_elapsed = datetime.now().replace(microsecond=0) - since
        print('-' * 10, flush=True)
        print('Performance on {} set:'.format(phase.upper()), flush=True)
        phase_object = {}
        # tasks = ['damage_severity', 'informative', 'humanitarian', 'disaster_types']
        for idx in range(num_tasks):
            acc = metrics.accuracy_score(labels_[idx], preds_[idx])
            class_report = metrics.classification_report(labels_[idx], preds_[idx], digits=3)
            conf_mat = metrics.confusion_matrix(labels_[idx], preds_[idx])
            print('Accuracy: {:.4f}'.format(acc), flush=True)
            print(class_report, flush=True)
            print(conf_mat, flush=True)
            print('-' * 10, flush=True)

            label_y = []
            label_pred = []
            task_class_names = class_names[idx]
            for i in range(len(preds_[idx])):
                label_y.append(task_class_names[labels_[idx][i]])
                label_pred.append(task_class_names[preds_[idx][i]])

            acc = metrics.accuracy_score(label_y, label_pred)
            precision = metrics.precision_score(label_y, label_pred, average="weighted")
            recall = metrics.recall_score(label_y, label_pred, average="weighted")
            f1_score = metrics.f1_score(label_y, label_pred, average="weighted")

            result = str("{0:.3f}".format(acc)) + "\t" + str(
                "{0:.3f}".format(precision)) + "\t" + str("{0:.3f}".format(recall)) + "\t" + str(
                "{0:.3f}".format(f1_score)) + "\n"

            print(result)
            conf_mat_str = train_utils.format_conf_mat(label_y, label_pred)
            agr_met = train_utils.compute_aggregate_scores(label_y, label_pred, task_class_names)
            task = tasks[idx]
            phase_object[task] = {
                "accuracy": acc,
                "results": result,
                "classification_report": class_report,
                "confusion_matrix": conf_mat,
                "conf_mat_str": conf_mat_str,
                "gold": labels_[idx],
                "pred": preds_[idx],
                "agr_met": agr_met,
                "execution_time": str(time_elapsed)
            }

        all_phases_result[phase] = train_utils.convert_to_json(phase_object)

    return all_phases_result


def test_model_save_results(model, dataloader, num_classes, tasks, class_names=None, device=None):
    print("Testing the model and saving the results:", flush=True)

    model.eval()
    num_tasks = len(num_classes)
    preds_ = [[] for _ in range(num_tasks)]
    labels_ = [[] for _ in range(num_tasks)]
    since = datetime.now().replace(microsecond=0)
    with torch.no_grad():
        for _, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = [labels[i].to(device) for i in range(len(labels))]

            outputs = model(inputs)

            idx_start = 0
            for idx in range(num_tasks):
                output_idx = outputs[:, idx_start:idx_start + num_classes[idx]]
                label_idx = labels[idx]
                valid_idx = torch.nonzero(1 * (label_idx != -1)).view(-1)
                if len(valid_idx) == 0:
                    continue
                idx_start += num_classes[idx]
                _, preds = torch.max(output_idx[valid_idx], 1)

                preds_[idx].extend(preds.cpu().numpy().tolist())
                labels_[idx].extend(label_idx[valid_idx].data.cpu().numpy().tolist())
    time_elapsed = datetime.now().replace(microsecond=0) - since
    print('-' * 10, flush=True)
    phase_object = {}
    for idx in range(num_tasks):
        acc = metrics.accuracy_score(labels_[idx], preds_[idx])
        class_report = metrics.classification_report(labels_[idx], preds_[idx], digits=3)
        conf_mat = metrics.confusion_matrix(labels_[idx], preds_[idx])
        print('Accuracy: {:.4f}'.format(acc), flush=True)
        print(class_report, flush=True)
        print(conf_mat, flush=True)
        print('-' * 10, flush=True)
        print(flush=True)

        label_y = []
        label_pred = []
        task_class_names = class_names[idx]
        for i in range(len(preds_[idx])):
            label_y.append(task_class_names[labels_[idx][i]])
            label_pred.append(task_class_names[preds_[idx][i]])

        acc = metrics.accuracy_score(label_y, label_pred)
        precision = metrics.precision_score(label_y, label_pred, average="weighted")
        recall = metrics.recall_score(label_y, label_pred, average="weighted")
        f1_score = metrics.f1_score(label_y, label_pred, average="weighted")

        result = str("{0:.3f}".format(acc)) + "\t" + str(
            "{0:.3f}".format(precision)) + "\t" + str("{0:.3f}".format(recall)) + "\t" + str(
            "{0:.3f}".format(f1_score)) + "\n"

        print(result)
        conf_mat_str = train_utils.format_conf_mat(labels_[idx], preds_[idx])
        agr_met = train_utils.compute_aggregate_scores(label_y, label_pred, task_class_names)
        task = tasks[idx]
        phase_object[task] = {
            "accuracy": acc,
            "results": result,
            "classification_report": class_report,
            "confusion_matrix": conf_mat,
            "conf_mat_str": conf_mat_str,
            "gold": labels_[idx],
            "pred": preds_[idx],
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

    # df = pd.read_csv(train_path, sep=sep, na_filter=False)
    # task_names = [df.columns[i] for i in range(3, len(df.columns))]
    task_names = args.task_names.split(',')
    num_classes = [num_classes_tasks[x] for x in task_names]
    total_classes = sum(num_classes)
    model, resize_image_size, input_size = model_utils.initialize_model(model_name, total_classes, keep_frozen=False,
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
        'train': dataset.MultitaskDataset(train_path, sep, data_dir, data_transforms['train'], task_names),
        'val': dataset.MultitaskDataset(dev_path, sep, data_dir, data_transforms['val'], task_names)}

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

    model, time_elapsed = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_classes, task_names,
                                      device=device, num_epochs=num_epochs, curr_epoch=curr_epoch,
                                      curr_evals=curr_evals, class_names=class_names, fig_dir=fig_dir,
                                      checkpoint_dir=checkpoint_dir, run_name=args.name,
                                      is_inception=(model_name == "inception"))

    all_phases_result = test_model(model, dataloaders_dict, num_classes, task_names,
                                   class_names=class_names, device=device)
    all_phases_result['train']['execution_time'] = str(time_elapsed)

    if args.test:
        test_transforms = data_transforms['val']
        test_dataset = dataset.MultitaskDataset(args.test, sep, data_dir, test_transforms, task_names)
        print("There are {} test images.".format(len(test_dataset)))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        results = test_model_save_results(model, test_dataloader, num_classes, task_names,
                                          class_names=class_names, device=device)

        all_phases_result['test'] = train_utils.convert_to_json(results)

    dir_name = os.path.dirname(args.out_file)
    os.makedirs(dir_name, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(train_utils.convert_to_json(all_phases_result), f)


if __name__ == "__main__":
    main()
