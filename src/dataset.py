import os

import pandas as pd
from PIL import Image
from torch.utils.data import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MultitaskDataset(Dataset):
    """
    Dataset class for multi-task image classification
    """

    def __init__(self, file_path, sep, root_dir, transform, task_names):
        """

        :param file_path: File containing image file path and labels
        :param sep: separator to read label csv file
        :param root_dir: root directory for image files
        :param transform: PIL transforms to apply
        :param task_names: list of tasks to use
        """
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(file_path, sep=sep, na_filter=False)
        # in our csv files, labels start from 4th column (index 3)
        self.X = df['image_path'].tolist()
        self.Y = [df[x].tolist() for x in task_names]

        self.classes = []
        self.class_to_indices = []

        for item in self.Y:
            cls, cls_to_idx = self._find_classes(item)
            self.classes.append(cls)
            self.class_to_indices.append(cls_to_idx)
        self.sample_x = self.X
        self.sample_y = []
        for img in range(len(self.Y[0])):
            y = []
            for task in range(len(self.Y)):
                y.append(self.class_to_indices[task][self.Y[task][img]])
            self.sample_y.append(y)

    def __getitem__(self, index):
        path, labels = self.sample_x[index], self.sample_y[index]
        f = open(os.path.join(self.root_dir, path), 'rb')
        img = Image.open(f)
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.sample_x)

    @staticmethod
    def _find_classes(y):
        """

        :param y: list of class names for a task
        :return: list of unique classes and a dictionary of class_name to index mapping
        If a task has unlabeled class(annotated by ""), label index start from -1 corresponding to that class
        else the index starts from 0
        """
        classes_set = set(y)
        classes = list(classes_set)
        classes.sort()
        if classes[0] == '':
            class_to_idx = {classes[i]: i - 1 for i in range(len(classes))}
        else:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class SingleTaskDataset(Dataset):
    """
    Dataset class for single-task image classification
    """

    def __init__(self, file_path, task_name, sep, root_dir, transform=None):
        """

        :param file_path: File containing image file path and labels
        :param sep: separator to read label csv file
        :param root_dir: root directory for image files
        :param transform: PIL transforms to apply
        """
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(file_path, sep=sep, dtype=str)
        self.X = df['image_path'].tolist()
        self.y = df[task_name].tolist()
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = list(zip(self.X, [self.class_to_idx[i] for i in self.y]))

    def __getitem__(self, index):
        path, label = self.samples[index]
        f = open(os.path.join(self.root_dir, path), 'rb')
        img = Image.open(f)
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _find_classes(self):
        classes_set = set(self.y)
        classes = list(classes_set)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
