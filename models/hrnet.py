import numbers
import collections
import logging
import torch
from torch import nn
from torch.nn import functional as F
from models.mobilenet_base import _make_divisible
from models.mobilenet_base import ConvBNReLU
from models.mobilenet_base import get_active_fn
from models.mobilenet_base import InvertedResidualChannels, InvertedResidualChannelsFused
from mmseg.utils import resize
import json
from utils import distributed as udist

__all__ = ['HighResolutionNet']

checkpoint_kwparams = None
# checkpoint_kwparams = json.load(open('checkpoint.json'))


class InvertedResidual(InvertedResidualChannels):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 kernel_sizes,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 **kwargs):

        def _expand_ratio_to_hiddens(expand_ratio):
            if isinstance(expand_ratio, list):
                assert len(expand_ratio) == len(kernel_sizes)
                expand = True
            elif isinstance(expand_ratio, numbers.Number):
                expand = expand_ratio != 1
                expand_ratio = [expand_ratio for _ in kernel_sizes]
            else:
                raise ValueError(
                    'Unknown expand_ratio type: {}'.format(expand_ratio))
            hidden_dims = [int(round(inp * e)) for e in expand_ratio]
            return hidden_dims, expand

        hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
        if checkpoint_kwparams:
            assert oup == checkpoint_kwparams[0][0]
            if udist.is_master():
                logging.info('loading: {} -> {}, {} -> {}'.format(
                    hidden_dims, checkpoint_kwparams[0][4], kernel_sizes, checkpoint_kwparams[0][3]))
            hidden_dims = checkpoint_kwparams[0][4]
            kernel_sizes = checkpoint_kwparams[0][3]
            checkpoint_kwparams.pop(0)

        super(InvertedResidual,
              self).__init__(inp,
                             oup,
                             stride,
                             hidden_dims,
                             kernel_sizes,
                             expand,
                             active_fn=active_fn,
                             batch_norm_kwargs=batch_norm_kwargs)
        self.expand_ratio = expand_ratio


def get_block_wrapper(block_str):
    """Wrapper for MobileNetV2 block.
    Use `expand_ratio` instead of manually specified channels number."""

    assert block_str == 'InvertedResidualChannels'
    return InvertedResidual


class ParallelModule(nn.Module):
    def __init__(self,
                 num_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 num_blocks=[2, 2],
                 num_channels=[32, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
        super(ParallelModule, self).__init__()

        self.num_branches = num_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes

        self._check_branches(
            num_branches, num_blocks, num_channels)
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels)

    def _check_branches(self, num_branches, num_blocks, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index],
                    num_channels[branch_index],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return x


class FuseModule(nn.Module):
    '''
        Consistent with HRNET:
        1. self.use_hr_format, eg: fuse 3 branches, and then add 4th branch from 3rd branch. (default fuse 4 branches)
        2. use_hr_format, if the channels are the same and stride==1, use None rather than fuse. (default, always fuse)
            and use convbnrelu, and kernel_size=1 when upsample.
            also control the relu here (last layer no relu)
        3. self.in_channels_large_stride, use 16->16->64 instead of 16->32->64 for large stride. (default, True)
        4. The only difference in self.use_hr_format when adding a branch:
            is we use add 4th branch from 3rd branch, add 5th branch from 4rd branch
            hrnet use add 4th branch from 3rd branch, add 5th branch from 3rd branch (2 conv layers)
            actually only affect 1->2 stage
            can be hard coded: self.use_hr_format = self.use_hr_format and not(out_branches == 2 and in_branches == 1)
        5. hrnet have a fuse layer at the end, we remove it
    '''
    def __init__(self,
                 in_branches=1,
                 out_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 in_channels=[16],
                 out_channels=[16, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6'),
                 use_hr_format=False,
                 only_fuse_neighbor=True,
                 directly_downsample=True):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.only_fuse_neighbor = only_fuse_neighbor
        self.in_channels_large_stride = True  # see 3.
        if only_fuse_neighbor:
            self.use_hr_format = out_branches > in_branches
            # w/o self, are two different flags. (see 1.)
        else:
            self.use_hr_format = out_branches > in_branches and \
                                 not (out_branches == 2 and in_branches == 1)  # see 4.

        self.relu = self.active_fn()
        if use_hr_format:
            block = ConvBNReLU  # See 2.

        fuse_layers = []
        for i in range(out_branches if not self.use_hr_format else in_branches):
            fuse_layer = []
            for j in range(in_branches):
                if only_fuse_neighbor:
                    if j < i - 1 or j > i + 1:
                        fuse_layer.append(None)
                        continue
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.active_fn if not use_hr_format else None,
                            kernel_size=1  # for hr format
                        ),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    if use_hr_format and in_channels[j] == out_channels[i]:
                        fuse_layer.append(None)
                    else:
                        fuse_layer.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=1,
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.active_fn if not use_hr_format else None,
                                kernel_size=3  # for hr format
                            ))
                else:
                    downsamples = []
                    if directly_downsample:
                        downsamples.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=2 ** (i - j),
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.active_fn if not use_hr_format else None,
                                kernel_size=3  # for hr format
                            ))
                    else:
                        for k in range(i - j):
                            if self.in_channels_large_stride:
                                if k == i - j - 1:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            out_channels[i],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn if not use_hr_format else None,
                                            kernel_size=3  # for hr format
                                        ))
                                else:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            in_channels[j],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn,
                                            kernel_size=3  # for hr format
                                        ))
                            else:
                                if k == 0:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            out_channels[j + 1],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn if not (use_hr_format and i == j + 1) else None,
                                            kernel_size=3  # for hr format
                                        ))
                                elif k == i - j - 1:
                                    downsamples.append(
                                        block(
                                            out_channels[j + k],
                                            out_channels[i],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn if not use_hr_format else None,
                                            kernel_size=3  # for hr format
                                        ))
                                else:
                                    downsamples.append(
                                        block(
                                            out_channels[j + k],
                                            out_channels[j + k + 1],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn,
                                            kernel_size=3  # for hr format
                                        ))
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        if self.use_hr_format:
            for branch in range(in_branches, out_branches):
                fuse_layers.append(nn.ModuleList([block(
                    out_channels[branch - 1],
                    out_channels[branch],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=2,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn,
                    kernel_size=3  # for hr format
                )]))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        if not self.only_fuse_neighbor:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                y = self.fuse_layers[i][0](x[0]) if self.fuse_layers[i][0] else x[0]  # hr_format, None
                for j in range(1, self.in_branches):
                    if self.fuse_layers[i][j]:
                        y = y + self.fuse_layers[i][j](x[j])
                    else:  # hr_format, None
                        y = y + x[j]
                x_fuse.append(self.relu(y))
            if self.use_hr_format:
                for branch in range(self.in_branches, self.out_branches):
                    x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        else:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                flag = 1
                for j in range(i-1, i+2):
                    if 0 <= j < self.in_branches:
                        if flag:
                            y = self.fuse_layers[i][j](x[j]) if self.fuse_layers[i][j] else x[j]  # hr_format, None
                            flag = 0
                        else:
                            if self.fuse_layers[i][j]:
                                y = y + resize(
                                    self.fuse_layers[i][j](x[j]),
                                    size=y.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)
                            else:  # hr_format, None
                                y = y + x[j]
                x_fuse.append(self.relu(y))
            if self.use_hr_format:
                for branch in range(self.in_branches, self.out_branches):
                    x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        return x_fuse


class HeadModule(nn.Module):
    def __init__(self,
                 pre_stage_channels=[16, 32, 64, 128],
                 head_channels=None,  # [32, 64, 128, 256],
                 last_channel=1024,
                 avg_pool_size=7,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6'),
                 concat_head_for_cls=False):
        super(HeadModule, self).__init__()

        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.avg_pool_size = avg_pool_size
        self.concat_head_for_cls = concat_head_for_cls

        # Increasing the #channels on each resolution
        if head_channels:
            incre_modules = []
            for i, channels in enumerate(pre_stage_channels):
                incre_module = block(
                    pre_stage_channels[i],
                    head_channels[i],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
                incre_modules.append(incre_module)
            self.incre_modules = nn.ModuleList(incre_modules)
        else:
            head_channels = pre_stage_channels
            self.incre_modules = []

        if not self.concat_head_for_cls:
            # downsampling modules
            downsamp_modules = []
            for i in range(len(pre_stage_channels) - 1):
                downsamp_module = block(
                    head_channels[i],
                    head_channels[i + 1],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=2,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
                downsamp_modules.append(downsamp_module)
            self.downsamp_modules = nn.ModuleList(downsamp_modules)
        else:
            self.downsamp_modules = []

        self.final_layer = ConvBNReLU(
            head_channels[-1] if not self.concat_head_for_cls else sum(head_channels),
            last_channel,
            kernel_size=1,
            batch_norm_kwargs=batch_norm_kwargs,
            active_fn=active_fn)

    def forward(self, x_list):
        if self.concat_head_for_cls:
            if self.incre_modules:
                for i in range(len(x_list)):
                    x_list[i] = self.incre_modules[i](x_list[i])
            x_incre = [resize(input=x,
                              size=x_list[-1].shape[2:],
                              mode='bilinear',
                              align_corners=False) for x in x_list]
            x = torch.cat(x_incre, dim=1)
        else:
            if self.incre_modules:
                x = self.incre_modules[0](x_list[0])
                for i in range(len(self.downsamp_modules)):
                    x = self.incre_modules[i + 1](x_list[i + 1]) \
                        + self.downsamp_modules[i](x)
            else:
                x = x_list[0]
                for i in range(len(self.downsamp_modules)):
                    x = x_list[i + 1] \
                        + self.downsamp_modules[i](x)

        x = self.final_layer(x)

        # assert x.size()[2] == self.avg_pool_size

        if torch._C._get_tracing_state():
            x = x.flatten(start_dim=2).mean(dim=2)
        else:
            x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size(0), -1)
        return x


class HighResolutionNet(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_stride=4,
                 input_channel=[16, 16],
                 last_channel=1024,
                 head_channels=None,
                 bn_momentum=0.1,
                 bn_epsilon=1e-5,
                 dropout_ratio=0.2,
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 width_mult=1.0,
                 round_nearest=8,
                 expand_ratio=4,
                 kernel_sizes=[3, 5, 7],
                 inverted_residual_setting=None,
                 task='classification',
                 align_corners=False,
                 start_with_atomcell=False,
                 fcn_head_for_seg=False,
                 initial_for_heatmap=False,
                 **kwargs):
        super(HighResolutionNet, self).__init__()

        batch_norm_kwargs = {
            'momentum': bn_momentum,
            'eps': bn_epsilon
        }

        self.avg_pool_size = input_size // 32
        self.input_stride = input_stride
        self.input_channel = [_make_divisible(item * width_mult, round_nearest) for item in input_channel]
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = get_active_fn(active_fn)
        self.kernel_sizes = kernel_sizes
        self.expand_ratio = expand_ratio
        self.task = task
        self.align_corners = align_corners
        self.initial_for_heatmap = initial_for_heatmap

        self.block = get_block_wrapper(block)
        self.inverted_residual_setting = inverted_residual_setting

        downsamples = []
        if self.input_stride > 1:
            downsamples.append(ConvBNReLU(
                3,
                input_channel[0],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        if self.input_stride > 2:
            if start_with_atomcell:
                downsamples.append(InvertedResidual(input_channel[0],
                                                    input_channel[0],
                                                    1,
                                                    1,
                                                    [3],
                                                    self.active_fn,
                                                    self.batch_norm_kwargs))
            downsamples.append(ConvBNReLU(
                input_channel[0],
                input_channel[1],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        self.downsamples = nn.Sequential(*downsamples)

        features = []
        for index in range(len(inverted_residual_setting)):
            in_branches = 1 if index == 0 else inverted_residual_setting[index - 1][0]
            in_channels = [input_channel[1]] if index == 0 else inverted_residual_setting[index - 1][-1]
            features.append(
                FuseModule(
                    in_branches=in_branches,
                    out_branches=inverted_residual_setting[index][0],
                    in_channels=in_channels,
                    out_channels=inverted_residual_setting[index][-1],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
            features.append(
                ParallelModule(
                    num_branches=inverted_residual_setting[index][0],
                    num_blocks=inverted_residual_setting[index][1],
                    num_channels=inverted_residual_setting[index][2],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )

        if self.task == 'classification':
            features.append(HeadModule(
                pre_stage_channels=inverted_residual_setting[-1][2],
                head_channels=head_channels,
                last_channel=last_channel,
                avg_pool_size=self.avg_pool_size,
                block=self.block,
                expand_ratio=self.expand_ratio,
                kernel_sizes=self.kernel_sizes,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_ratio),
                nn.Linear(last_channel, num_classes),
            )
        elif self.task == 'segmentation':
            if fcn_head_for_seg:
                self.transform = ConvBNReLU(
                    sum(inverted_residual_setting[-1][-1]),
                    last_channel,
                    kernel_size=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn
                )
            else:
                self.transform = self.block(
                        sum(inverted_residual_setting[-1][-1]),
                        last_channel,
                        expand_ratio=self.expand_ratio,
                        kernel_sizes=self.kernel_sizes,
                        stride=1,
                        batch_norm_kwargs=self.batch_norm_kwargs,
                        active_fn=self.active_fn,
                    )
            self.classifier = nn.Conv2d(last_channel,
                                        num_classes,
                                        kernel_size=1)

        self.features = nn.Sequential(*features)

        self.init_weights()

    def init_weights(self):
        if udist.is_master():
            logging.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not self.initial_for_heatmap:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_named_block_list(self):
        """Get `{name: module}` dictionary for all inverted residual blocks."""
        blocks = list(self.features.named_children())
        all_cells = []
        for name, block in blocks:
            if isinstance(block, ParallelModule):
                parallel_module = block.branches
                for i, parallel_branch in enumerate(parallel_module):
                    for j, cell in enumerate(parallel_branch):
                        all_cells.append(
                            ('features.{}.branches.{}.{}'.format(name, i, j), cell))
            if isinstance(block, FuseModule):
                fuse_module = block.fuse_layers
                for i, fuse_branch in enumerate(fuse_module):
                    for j, fuse_path in enumerate(fuse_branch):
                        if isinstance(fuse_path, self.block):
                            all_cells.append(
                                ('features.{}.fuse_layers.{}.{}'.format(name, i, j), fuse_path))
                        if isinstance(fuse_path, nn.Sequential):
                            for k, cell in enumerate(fuse_path):
                                if isinstance(cell, self.block):
                                    all_cells.append(
                                        ('features.{}.fuse_layers.{}.{}.{}'.format(name, i, j, k), cell))
            if isinstance(block, HeadModule):
                incre_module = block.incre_modules
                downsample_module = block.downsamp_modules
                for i, cell in enumerate(incre_module):
                    if isinstance(cell, self.block):
                        all_cells.append(
                            ('features.{}.incre_modules.{}'.format(name, i), cell))
                for i, cell in enumerate(downsample_module):
                    if isinstance(cell, self.block):
                        all_cells.append(
                            ('features.{}.downsamp_modules.{}'.format(name, i), cell))
        for name, block in self.named_children():
            if isinstance(block, self.block):
                all_cells.append(
                    ('{}'.format(name), block))

        return collections.OrderedDict(all_cells)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        inputs = self.transform(inputs)
        return inputs

    def forward(self, x):
        x = self.downsamples(x)
        x = self.features([x])
        if self.task == 'segmentation':
            x = self._transform_inputs(x)
        x = self.classifier(x)
        return x


Model = HighResolutionNet
