from torchvision import models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def set_parameter_requires_grad(model, keep_frozen):
    """
    If keep_frozen is True make model parameters frozen i.e., no gradient update
    """
    if keep_frozen:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, keep_frozen=False, use_pretrained=True):
    """

    :param model_name: allowed [resnet18, resnet50, resnet101, alexnet, vgg11, vgg16, squeezenet, densenet, inception,
    mobilenet_v2, efficientnet-b1, efficientnet-b7]
    :param num_classes: number of classes in the classification layer
    :param keep_frozen: if True feature layer weights are kep frozen, else full network is fine-tuned
    :param use_pretrained: if True loads ImageNet pretrained weight, else trains from scratch
    :return: Tuple of [model, resize_image_size, input_size]
    """
    model = None
    # model input image size
    input_image_size = 0
    # image is resized to this size before cropping as input_image_size
    resize_image_size = 0
    if model_name == "resnet18":
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "vgg11":
        model = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "vgg16":
        model = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "densenet":
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        # Handle the auxiliary net
        num_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features, num_classes)
        # Handle the primary net
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        input_image_size = 299
        resize_image_size = 299
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model, keep_frozen)
        num_features = model.last_channel
        model.fc = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "efficientnet-b1":
        model = EfficientNet.from_pretrained('efficientnet-b1')
        set_parameter_requires_grad(model, keep_frozen)
        # noinspection PyProtectedMember
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    elif model_name == "efficientnet-b7":
        model = EfficientNet.from_pretrained('efficientnet-b7')
        set_parameter_requires_grad(model, keep_frozen)
        # noinspection PyProtectedMember
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
        input_image_size = 224
        resize_image_size = 256
    else:
        print("Invalid model name, exiting...", flush=True)
        exit()

    return model, resize_image_size, input_image_size
