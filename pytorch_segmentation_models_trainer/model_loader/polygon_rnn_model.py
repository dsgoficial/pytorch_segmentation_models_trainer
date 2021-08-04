# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-02
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer 
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   Part of the code is from                                              *
 *   https://github.com/AlexMa011/pytorch-polygon-rnn                      *
 ****
"""
from collections import OrderedDict
from logging import log
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.init
import torch.utils.model_zoo as model_zoo
import wget
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils import tensor_utils
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        Code extracted from https://github.com/AlexMa011/pytorch-polygon-rnn
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur],
                             dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
                                             dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        return (
            Variable(
                torch.zeros(
                    batch_size, self.hidden_dim, self.height, self.width
                ).to(device)
            ),
            Variable(
                torch.zeros(
                    batch_size, self.hidden_dim, self.height, self.width
                ).to(device)
            )
        )


class ConvLSTM(nn.Module):
    """
    Code extracted from https://github.com/AlexMa011/pytorch-polygon-rnn
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists
        # having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[
                i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                #                 print(cur_layer_input.shape)

                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device=torch.device('cpu')):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device=device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(
                    [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class PolygonRNN(nn.Module):
    """
    Code extracted from https://github.com/AlexMa011/pytorch-polygon-rnn
    """
    def __init__(self, load_vgg=True):
        super(PolygonRNN, self).__init__()

        def _make_basic(input_size, output_size, kernel_size, stride, padding):
            """

            :rtype: nn.Sequential
            """
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size, stride,
                          padding),
                nn.ReLU(),
                nn.BatchNorm2d(output_size)
            )

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.model4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.convlayer1 = _make_basic(128, 128, 3, 1, 1)
        self.convlayer2 = _make_basic(256, 128, 3, 1, 1)
        self.convlayer3 = _make_basic(512, 128, 3, 1, 1)
        self.convlayer4 = _make_basic(512, 128, 3, 1, 1)
        self.convlayer5 = _make_basic(512, 128, 3, 1, 1)
        self.poollayer = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convlstm = ConvLSTM(input_size=(28, 28),
                                 input_dim=131,
                                 hidden_dim=[32, 8],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=True)
        self.lstmlayer = nn.LSTM(28 * 28 * 8 + (28 * 28 + 3) * 2, 28 * 28 * 2,
                                 batch_first=True)
        self.linear = nn.Linear(28 * 28 * 2, 28 * 28 + 3)
        self.init_weights(load_vgg=load_vgg)

    def init_weights(self, load_vgg=True):
        '''
        Initialize weights of PolygonNet
        :param load_vgg: bool
                    load pretrained vgg model or not
        '''

        for name, param in self.convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.lstmlayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight' in name:
                # nn.init.xavier_normal_(param)
                nn.init.orthogonal_(param)
        for name, param in self.named_parameters():
            if 'bias' in name and 'convlayer' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'convlayer' in name and '0' in name:
                nn.init.xavier_normal_(param)
        if load_vgg:
            self.load_vgg()

    def load_vgg(self):
        vgg_file = Path('vgg16_bn-6c64b313.pth')
        vgg16_dict = torch.load('vgg16_bn-6c64b313.pth') \
            if vgg_file.is_file() else self.download_vgg16()
        vgg_name = [
            name for name in vgg16_dict \
                if 'feature' in name and 'running' not in name
        ]
        for idx, (name, param) in enumerate(self.named_parameters()):
            if 'model' in name:
                param.data.copy_(vgg16_dict[vgg_name[idx]])

    def download_vgg16(self):
        try:
            wget.download(
                        'https://download.pytorch.org/models/vgg16_bn'
                        '-6c64b313.pth')
            vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
        except:
            vgg16_dict = torch.load(model_zoo.load_url(
                        'https://download.pytorch.org/models/vgg16_bn'
                        '-6c64b313.pth'))
                    
        return vgg16_dict

    def compute_output(self, model, x, convlayer=None, apply_pooling=True):
        output_k = model(x)
        output_kk = self.poollayer(output_k) if apply_pooling else output_k
        output_kk = convlayer(output_kk) if convlayer is not None else output_kk
        return output_k, output_kk

    def forward(self, input_data1, first, second, third):
        bs, length_s = second.shape[0], second.shape[1]
        output1, output11 = self.compute_output(self.model1, input_data1)
        output2, output22 = self.compute_output(self.model2, output1, convlayer=self.convlayer2, apply_pooling=False)
        output3, output33 = self.compute_output(self.model3, output2, convlayer=self.convlayer3, apply_pooling=False)
        output4, output44 = self.compute_output(self.model4, output3, convlayer=self.convlayer4, apply_pooling=False)
        output44 = self.upsample(output44)
        output = torch.cat([output11, output22, output33, output44], dim=1)
        output = self.convlayer5(output)
        output = output.unsqueeze(1)
        output = output.repeat(1, length_s, 1, 1, 1)
        device = output.device
        padding_f = torch.zeros([bs, 1, 1, 28, 28]).to(device)

        input_f = first[:, :-3].view(-1, 1, 28, 28).unsqueeze(1).repeat(1,
                                                                        length_s - 1,
                                                                        1, 1,
                                                                        1)
        input_f = torch.cat([padding_f, input_f], dim=1)
        input_s = second[:, :, :-3].view(-1, length_s, 1, 28, 28)
        input_t = third[:, :, :-3].view(-1, length_s, 1, 28, 28)
        output = torch.cat([output, input_f, input_s, input_t], dim=2)

        output = self.convlstm(output)[0][-1]

        shape_o = output.shape
        output = output.contiguous().view(bs, length_s, -1)
        output = torch.cat([output, second, third], dim=2)
        output = self.lstmlayer(output)[0]
        output = output.contiguous().view(bs * length_s, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, length_s, -1)

        return output

    def test(self, input_data1, len_s):
        bs = input_data1.shape[0]
        result = torch.zeros([bs, len_s]).to(input_data1.device)
        output1, output11 = self.compute_output(self.model1, input_data1)
        output2, output22 = self.compute_output(self.model2, output1, convlayer=self.convlayer2, apply_pooling=False)
        output3, output33 = self.compute_output(self.model3, output2, convlayer=self.convlayer3, apply_pooling=False)
        output4, output44 = self.compute_output(self.model4, output3, convlayer=self.convlayer4, apply_pooling=False)
        output44 = self.upsample(output44)
        output = torch.cat([output11, output22, output33, output44], dim=1)
        feature = self.convlayer5(output)

        padding_f = torch.zeros([bs, 1, 1, 28, 28]).float().to(input_data1.device)
        input_s = torch.zeros([bs, 1, 1, 28, 28]).float().to(input_data1.device)
        input_t = torch.zeros([bs, 1, 1, 28, 28]).float().to(input_data1.device)

        output = torch.cat([feature.unsqueeze(1), padding_f, input_s, input_t],
                           dim=2)

        output, hidden1 = self.convlstm(output)
        output = output[-1]
        output = output.contiguous().view(bs, 1, -1)
        second = torch.zeros([bs, 1, 28 * 28 + 3]).to(input_data1.device)
        second[:, 0, 28 * 28 + 1] = 1
        third = torch.zeros([bs, 1, 28 * 28 + 3]).to(input_data1.device)
        third[:, 0, 28 * 28 + 2] = 1
        output = torch.cat([output, second, third], dim=2)

        output, hidden2 = self.lstmlayer(output)
        output = output.contiguous().view(bs, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, 1, -1)
        output = (output == output.max(dim=2, keepdim=True)[0]).float()
        first = output
        result[:, 0] = (output.argmax(2))[:, 0]

        for i in range(len_s - 1):
            second = third
            third = output
            input_f = first[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_s = second[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_t = third[:, :, :-3].view(-1, 1, 1, 28, 28)
            input1 = torch.cat(
                [feature.unsqueeze(1), input_f, input_s, input_t], dim=2)
            output, hidden1 = self.convlstm(input1, hidden1)
            output = output[-1]
            output = output.contiguous().view(bs, 1, -1)
            output = torch.cat([output, second, third], dim=2)
            output, hidden2 = self.lstmlayer(output, hidden2)
            output = output.contiguous().view(bs, -1)
            output = self.linear(output)
            output = output.contiguous().view(bs, 1, -1)
            output = (output == output.max(dim=2, keepdim=True)[0]).float()
            result[:, i + 1] = (output.argmax(2))[:, 0]

        return result


class PolygonRNNPLModel(Model):
    def __init__(self, cfg):
        super(PolygonRNNPLModel, self).__init__(cfg)
    
    def get_model(self):
        return PolygonRNN(
            load_vgg=self.cfg.model.load_vgg \
                if "load_vgg" in self.cfg.model else True)
    
    def get_loss_function(self):
        return nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.compute(batch)
        tensorboard_logs = {'acc': {'train': acc}}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def compute(self, batch):
        x = Variable(batch['image'])
        x1 = Variable(batch['x1'])
        x2 = Variable(batch['x2'])
        x3 = Variable(batch['x3'])
        ta = Variable(batch['ta'])
        output = self.model(x, x1, x2, x3)
        result = output.contiguous().view(-1, 28 * 28 + 3)
        target = ta.contiguous().view(-1)
        loss = self.loss_function(result, target)
        result_index = torch.argmax(result, 1)
        correct = (target == result_index).float().sum().item()
        acc = correct * 1.0 / target.shape[0]
        return loss,acc
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute(batch)
        tensorboard_logs = {'acc': {'val': acc}}
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'val_loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([
            torch.tensor(x['log']['acc']['train']) for x in outputs]).mean()
        tensorboard_logs = {
            'avg_loss': {'train' : avg_loss},
            'avg_acc': {'train' : avg_acc},
        }
        self.log_dict(tensorboard_logs, logger=True)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([
            torch.tensor(x['log']['acc']['val']) for x in outputs]).mean()
        tensorboard_logs = {
            'avg_loss': {'val' : avg_loss},
            'avg_acc': {'val' : avg_acc},
        }
        self.log_dict(tensorboard_logs, logger=True)
