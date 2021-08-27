import os
import json
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn import metrics


def load_checkpoint(best_state_path, model=None, optimizer=None, scheduler=None, epoch=0, evals=(None, None),
                    device="cpu"):
    """
    loads saved state to resume training
    :return: model, optimizer, scheduler, epoch, evals (losses, accuracies)
    """
    if os.path.isfile(best_state_path):
        # Load states
        checkpoint = torch.load(best_state_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # Update settings
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        accuracies = checkpoint['accuracies']
        evals = (losses, accuracies)
        print("Loaded checkpoint '{}' (epoch {}) successfully.".format(best_state_path, epoch), flush=True)
        epoch += 1
    else:
        print("No checkpoint found.", flush=True)
    return model, optimizer, scheduler, epoch, evals


def load_model_weight(best_state_path, model=None, device="cpu"):
    if os.path.isfile(best_state_path):
        checkpoint = torch.load(best_state_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint '{}' successfully.".format(best_state_path), flush=True)
    else:
        print("No checkpoint found.", flush=True)
    return model


def save_checkpoint(state, is_best, filename, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(checkpoint_dir, "best_state.pth"))


def save_best_checkpoint(state, is_best, filename, checkpoint_dir):
    """Saves the best model only

    Parameters
    ----------
    state : dict
        State of the model to be saved
    is_best : boolean
        Whether or not current model is the best model
    filename : str
        Name of the file to be saved
    checkpoint_dir : str
        Path to models directory
    """
    if is_best:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, "best_state.pth"))


def two_scales(ax1, task_name, x_data, train_loss, val_loss, train_acc, val_acc):
    ax2 = ax1.twinx()
    ax1.plot(x_data, train_loss, '--', label='train loss')
    ax1.plot(x_data, val_loss, '--', label='val loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    if task_name:
        ax1.set_title(task_name)
    ax2.plot(x_data, train_acc, label='train acc')
    ax2.plot(x_data, val_acc, label='val acc')
    ax2.set_ylabel('accuracy')
    ax1.legend()
    ax2.legend()
    return ax1, ax2


def convert_to_json(obj):
    """
    convert a dictionary to JSON-able object, converting all numpy arrays to python ists
    """
    converted_obj = {}

    for key, value in obj.items():
        if isinstance(value, dict):
            converted_obj[key] = convert_to_json(value)
        else:
            converted_obj[key] = np.array(value).tolist()
    return converted_obj


def save_json(out_dir, file_name, losses, accuracies, task_names):
    out_file = os.path.join(out_dir, file_name + '.json')
    json_object = {
        'tasks': task_names,
        'accuracies': accuracies,
        'losses': losses
    }
    with open(out_file, 'w') as f:
        json.dump(convert_to_json(json_object), f)


def format_conf_mat(y_true, y_pred):
    conf_mat = pd.crosstab(np.array(y_true), np.array(y_pred), rownames=['gold'], colnames=['pred'], margins=True)
    pred_columns = conf_mat.columns.tolist()
    gold_rows = conf_mat.index.tolist()
    header = "Pred\nGold"
    for h in pred_columns:
        header = header + "\t" + str(h)
    conf_mat_str = header + "\n"
    index = 0
    for r_index, row in conf_mat.iterrows():
        row_str = str(gold_rows[index])
        index += 1
        for col_item in row:
            row_str = row_str + "\t" + str(col_item)
        conf_mat_str = conf_mat_str + row_str + "\n"

    return conf_mat_str


def compute_aggregate_scores(all_labels, all_predictions, all_classes):
    prf_per_class = metrics.precision_recall_fscore_support(all_labels, all_predictions, labels=all_classes,
                                                            average=None)[:-1]
    prf_micro = metrics.precision_recall_fscore_support(all_labels, all_predictions, labels=all_classes,
                                                        average='micro')[:-1]
    prf_macro = metrics.precision_recall_fscore_support(all_labels, all_predictions, labels=all_classes,
                                                        average='macro')[:-1]
    prf_weighted = metrics.precision_recall_fscore_support(all_labels, all_predictions, labels=all_classes,
                                                           average='weighted')[:-1]
    aggregated_metrics = {
        "prf_per_class": prf_per_class,
        "prf_per_class_labels": all_classes,
        "prf_micro": prf_micro,
        "prf_macro": prf_macro,
        "prf_weighted": prf_weighted,
    }

    return aggregated_metrics
