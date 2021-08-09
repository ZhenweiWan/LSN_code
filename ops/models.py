# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from torch.nn.init import normal_, constant_
from detectron2.layers import ModulatedDeformConv
import torchvision
import torch


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 partial_bn=True, print_spec=True, pretrain='imagenet', fc_lr5=False,
                 temporal_pool=False, short_len=8, long_len=1,
                 dataset='something', cttlstm=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.cttlstm = cttlstm
        self.short_len = short_len
        self.long_len = long_len
        self.new_length = new_length
        self.dataset = dataset

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if (modality == 'RGB') or (self.dataset == 'something_v2'):
            self.single_frame_channel = 3
        else:
            self.single_frame_channel = 2

        if not (self.single_frame_channel * self.new_length == 3):
            self._reconstruct_first_layer()
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)

            if self.cttlstm:
                print('Adding CTTLSTM module...')
                from ops.CTTLSTM import _convlstmBlock
                self.base_model.layer1.add_module('CTTLSTM', _convlstmBlock(256))
                self.base_model.layer3.add_module('CTTLSTM', _convlstmBlock(1024))

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)


        elif base_model == 'res2net':
            from model.Res2Net import res2net50_26w_4s
            self.base_model = res2net50_26w_4s(pretrained=True if self.pretrain == 'imagenet' else False)

            if self.cttlstm:
                print('Adding CTTLSTM module...')
                from ops.CTTLSTM import _convlstmBlock
                self.base_model.layer1.add_module('CTTLSTM', _convlstmBlock(256))
                self.base_model.layer3.add_module('CTTLSTM', _convlstmBlock(1024))

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        ln = []
        custom_ops = []

        cttlstm_weights = []
        cttlstm_bias = []

        conv_cnt = 0
        bn_cnt = 0
        ln_cnt = 0
        CTTLSTM_FLAG = False

        for m in self.modules():
            # CTTLSTM
            if hasattr(self.base_model.layer4, 'CTTLSTM'):
                if m == self.base_model.layer4.CTTLSTM:
                    CTTLSTM_FLAG = True
            if hasattr(self.base_model.layer3, 'CTTLSTM'):
                if m == self.base_model.layer4:
                    CTTLSTM_FLAG = False
                elif m == self.base_model.layer3.CTTLSTM:
                    CTTLSTM_FLAG = True
            if hasattr(self.base_model.layer2, 'CTTLSTM'):
                if m == self.base_model.layer3:
                    CTTLSTM_FLAG = False
                elif m == self.base_model.layer2.CTTLSTM:
                    CTTLSTM_FLAG = True
            if hasattr(self.base_model.layer1, 'CTTLSTM'):
                if m == self.base_model.layer2:
                    CTTLSTM_FLAG = False
                elif m == self.base_model.layer1.CTTLSTM:
                    CTTLSTM_FLAG = True

            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m,
                                                                                              torch.nn.Conv3d) or isinstance(
                m, ModulatedDeformConv):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])

                elif CTTLSTM_FLAG:
                    cttlstm_weights.append(ps[0])
                    if len(ps) == 2:
                        cttlstm_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.LayerNorm):
                ln_cnt += 1
                if ln_cnt == 1:
                    ln.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': ln, 'lr_mult': 1, 'decay_mult': 0,
             'name': "LN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
            {'params': cttlstm_weights, 'lr_mult': 5, 'decay_mult': 1,
             'name': "cttlstm_weights"},
            {'params': cttlstm_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "cttlstm_bias"},
        ]

    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    def forward(self, input):
        if (self.modality == 'RGB') or (self.dataset == 'something_v2'):
            input = input.view((-1, self.short_len,
                                self.new_length * 3) + input.size()[-2:])
        else:
            input = input.view((-1, self.short_len,
                                self.new_length * 2) + input.size()[-2:])
        (b, short_len, ch, h, w) = input.size()
        input = input.view(b * short_len, ch, h, w)

        base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _reconstruct_first_layer(self):
        print('Reconstructing first conv...')

        modules = list(self.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x],
                                                          nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.single_frame_channel * \
                                             self.new_length,) + \
                          kernel_size[2:]

        new_kernels = params[0].data.repeat([1, self.new_length, ] + \
                                            [1] * (len(kernel_size[2:]))).contiguous()

        new_conv = nn.Conv2d(self.single_frame_channel * \
                             self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride,
                             conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
