from torch import nn

def conv1x1(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1, padding_mode = 'zeros'):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                     stride = stride, padding = padding, bias = True, padding_mode = padding_mode)


def conv3x3(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1, padding_mode = 'zeros'):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                     stride = stride, padding = padding, bias = True, padding_mode = padding_mode)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, res_scale = 1, padding_mode = 'zeros', norm = False):
        super(ResBlock, self).__init__()
        self.norm = norm
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = padding_mode)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = padding_mode)
    def forward(self, x):
        out = self.conv1(x)
        if self.norm is True:
            out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, norm = 'in',
                 act = 'leakyrelu', padding_mode = 'zeros', spectral_norm = False, dilation = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = padding_mode, dilation = dilation)
        if spectral_norm is True:
            self.conv = nn.utils.spectral_norm(self.conv)
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'identity':
            self.norm = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace = True, negative_slope = 0.2)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResBlocks(nn.Module):
    def __init__(self, n = 2, in_channels = 64, out_channels = 64, padding_mode = 'replicate', kernel_size = 3, padding =1, stride = 1, norm = False):
        super(ResBlocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = padding_mode))
        for i in range(n):
            self.blocks.append(ResBlock(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = padding_mode, norm = norm))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvBlocks(nn.Module):
    def  __init__(self, n = 2, in_channels = 64, out_channels = 64, padding_mode = 'replicate', kernel_size = 3, padding =1, stride = 1, norm = False):
        super(ConvBlocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = padding_mode))
        for i in range(n):
            self.blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = padding_mode))
            if norm is True:
                self.blocks.append(
                    nn.InstanceNorm2d(out_channels))
            self.blocks.append(
                nn.ReLU())

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
