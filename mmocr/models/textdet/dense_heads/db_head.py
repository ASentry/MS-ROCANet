import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
import torch.nn.functional as F

from .head_mixin import HeadMixin


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        # m.append(nn.MaxPool2d(kernel_size=(2, 2)))
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


# in-scale non-local attention
class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, conv=default_conv):
        super(NonLocalAttention, self).__init__()

        # self.conv_match1 = common.involuntionBlock(channel, channel // reduction, 3)
        # self.conv_match2 = common.involuntionBlock(channel, channel // reduction, 3)
        # self.conv_assembly = common.involuntionBlock(channel, channel, 3)
        self.conv_match1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(0, 2, 1).view(N, -1, H, W)


@HEADS.register_module()
class DBHead(HeadMixin, nn.Module):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB
    """

    def __init__(self,
                 in_channels,
                 with_bias=False,
                 decoding_type='db',
                 text_repr_type='poly',
                 downsample_ratio=1.0,
                 loss=dict(type='DBLoss'),
                 train_cfg=None,
                 test_cfg=None):
        """Initialization.

        Args:
            in_channels (int): The number of input channels of the db head.
            decoding_type (str): The type of decoder for dbnet.
            text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
            downsample_ratio (float): The downsample ratio of ground truths.
            loss (dict): The type of loss for dbnet.
        """
        super().__init__()

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.text_repr_type = text_repr_type
        self.loss_module = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio
        self.decoding_type = decoding_type

        self.binarize = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid())

        self.threshold = self._init_thr(in_channels)

        self.nlocal = NonLocalAttention(channel=256, reduction=4)

    def init_weights(self):
        self.binarize.apply(self.init_class_parameters)
        self.threshold.apply(self.init_class_parameters)

    def init_class_parameters(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def diff_binarize(self, prob_map, thr_map, k):
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(self, inputs):
        inputs_nlocal = self.nlocal(inputs)
        prob_map = self.binarize(inputs_nlocal)
        thr_map = self.threshold(inputs)
        binary_map = self.diff_binarize(prob_map, thr_map, k=50)
        outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)
        return (outputs,)

    def _init_thr(self, inner_channels, bias=False):
        in_channels = inner_channels
        seq = nn.Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        return seq