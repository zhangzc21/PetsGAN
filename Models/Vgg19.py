import torch
from torchvision import models
from torch import nn
from model.MeanShift import MeanShift


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad = False, rgb_range = 1):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained = True).features

        self.slice1 = torch.nn.ModuleList()
        for x in range(36):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, X):
        h = self.sub_mean(X)
        feature = []
        for name, f in enumerate(self.slice1):
            h = f(h)
            if str(name) in ['3', '8', '17', '26', '35']:
                feature.append(h)
        return feature


if __name__ == '__main__':
    from torchsummary import summary
    vgg19 = Vgg19(requires_grad = False)
    summary(vgg19.cuda(), input_size = (3,256,256), batch_size = -1)
    pass
