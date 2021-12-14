import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import sys
import os
import functools


def read_image(path):
    from PIL import Image
    img = Image.open(path)
    img = img.convert("RGB")
    composed = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)])
    return composed(img).unsqueeze(0)


def show_image(x):
    x = ((x + 1) / 2).clamp(0, 1)
    x = torchvision.utils.make_grid(x, normalize = False)
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(x)
    plt.show()


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


# code from biggan
class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def preprocessor(img, resolution):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    img = transform(img)
    return img.unsqueeze(0)


# Load a model's weights
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


def load_weights(G,
                 D,
                 weights_root,
                 name_suffix = None,
                 G_ema = None,
                 strict = False):
    def map_func(storage, location):
        return storage.cuda()

    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, weights_root))
    else:
        print('Loading weights from %s...' % weights_root)
    if G is not None:
        G.load_state_dict(
            torch.load(
                '%s/%s.pth' %
                (weights_root, join_strings('_', ['G', name_suffix])),
                map_location = map_func),
            strict = strict)
    if D is not None:
        D.load_state_dict(
            torch.load(
                '%s/%s.pth' %
                (weights_root, join_strings('_', ['D', name_suffix])),
                map_location = map_func),
            strict = strict)
    if G_ema is not None:
        print('Loading ema generator...')
        G_ema.load_state_dict(
            torch.load(
                '%s/%s.pth' %
                (weights_root, join_strings('_', ['G_ema', name_suffix])),
                map_location = map_func),
            strict = strict)


def to_device(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs


def show(img):
    plt.figure(figsize = (12, 8))
    npimg = img.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(npimg)
    plt.show()


def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def noprint(flag):
    def wrapper(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if flag is False:
                func(*args, **kwargs)
            else:
                with HiddenPrints():
                    func(*args, **kwargs)

        return _wrapper

    return wrapper


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Prints:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group = 1000, ratio = 1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):  # 更新优化器参数
            if i >= num_group:
                param_group['lr'] = 0
            else:
                param_group['lr'] = learning_rate * ratio ** i
