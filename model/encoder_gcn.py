"""
Code originally from https://github.com/CVI-SZU/ME-GraphAU file: model/MEFL.py
Modified and adapted to work in the ELEGANT framework by Andrew Sumsion
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .vision_transformer import load_vit_b_16, load_vit_b_32, load_vit_l_16, load_vit_l_32, load_inception_v3
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *
from .AGG import AGG
from .COAL import COAL


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes * num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)
        else:
            start = self.start
            end = self.end

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(
            torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(
            torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, numEncoderLayers, secondDimensionSize, numLandmarks):
        super(Head, self).__init__()
        # The head of network
        # Modules: 1. AGG
        #          2. GNN
        #          3. COAL
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # edge fc: for edge prediction

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.numLandmarks = numLandmarks
        # head_embedding_dim = 48 # A small dimension for the head embedding. Divisible by num_classes for BP4D (12) and DISFA (8)
        head_embedding_dim = num_classes

        # AGG module:
        self.change_dimension, self.positional_encoding, self.transformer_encoder, self.numberHeads = AGG(num_classes, in_channels, secondDimensionSize, numEncoderLayers, head_embedding_dim).getAGG()

        # GNN module:
        self.gnn = GNN(head_embedding_dim, self.num_classes)

        # COAL module:
        self.sc, self.edge_fc, self.relu, self.emb_layer, self.lmk_layer1, self.lmk_layer2 = COAL(num_classes, head_embedding_dim, numLandmarks).getInfo()

        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):

        #flatten x so that it keeps the first dimension, but the rest of it is flattened
        # x_flat = x.flatten(start_dim=1)
        # Average pool x across the second dimension
        x_flat = torch.mean(x, dim=1)
        
        x_flat = self.change_dimension(x_flat)
        # [B, numberHeads * head_embedding_dim]
        

        # Reshape x_flat to be [B, numberHeads, head_embedding_dim]
        x_flat = x_flat.view(-1, self.numberHeads, x_flat.shape[-1] // self.numberHeads)
        

        # Add the head embeddings to the input
        head_indices = torch.arange(self.numberHeads, device=x.device).expand(x_flat.size(0), -1)
        head_embeddings = self.positional_encoding(head_indices)
        
        # Add the positional encoding: 
        token_positions = x_flat + head_embeddings
        
        # Normalize the token positions
        token_positions = F.normalize(token_positions, p=2, dim=-1)

        #make an encoder to pass the token_positions through
        token_positions = self.transformer_encoder(token_positions)
        nodes = token_positions[:, :self.num_classes, :]
        edges = token_positions[:, self.num_classes:, :]

        f_v, f_e = self.gnn(nodes, edges)

        b, n, c = f_v.shape
        f_v_reshaped = f_v.view(b, -1)
        emb_out = self.emb_layer(f_v_reshaped)

        # get the landmark predictions
        lmk_out = self.lmk_layer1(f_v_reshaped)
        lmk_out = self.relu(lmk_out)
        lmk_out = self.lmk_layer2(lmk_out)
        lmk_out = self.relu(lmk_out)
        #resize to be b x self.numLandmarks x 2
        lmk_out = lmk_out.view(b, self.numLandmarks, 2)

        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
        cl_edge = self.edge_fc(f_e)
        return cl, cl_edge, emb_out, lmk_out


class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', numEncoderLayers=2, numLandmarks=None):
        super(MEFARG, self).__init__()
        self.expand_dim = False
        self.useLogits = False
        if 'vit' in backbone:
            if backbone == 'vit_b_16':
                self.backbone, self.in_channels = load_vit_b_16()
            elif backbone == 'vit_b_32':
                self.backbone, self.in_channels = load_vit_b_32()
            elif backbone == 'vit_l_16':
                self.backbone, self.in_channels = load_vit_l_16()
            elif backbone == 'vit_l_32':
                self.backbone, self.in_channels = load_vit_l_32()
            else:
                raise Exception("Error: wrong backbone name: ", backbone)
            # self.out_channels = self.in_channels // 2
            self.expand_dim = True
            self.secondDimensionSize = 1

        elif 'inception' in backbone:
            if backbone == 'inception_v3':
                self.backbone, self.in_channels = load_inception_v3()
            else:
                raise Exception("Error: wrong backbone name: ", backbone)
            # self.out_channels = self.in_channels // 4
            self.expand_dim = True
            self.useLogits = True
            self.secondDimensionSize = 1

        elif 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            # self.out_channels = self.in_channels // 2
            self.backbone.head = None
            tmp = torch.randn(1, 3, 224, 224)
            tmp_out = self.backbone(tmp)
            self.secondDimensionSize = tmp_out.shape[1]
            tmp = None
            tmp_out = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)
        self.out_channels = 512
        while self.out_channels % num_classes != 0:
            self.out_channels += 1
        
        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, numEncoderLayers, self.secondDimensionSize, numLandmarks)

    def forward(self, x):
        
        # x: b d c
        x = self.backbone(x)
        if self.useLogits and not isinstance(x, torch.Tensor):
            x = x.logits
        if self.expand_dim:
            x = x.unsqueeze(1)

        x = self.global_linear(x)
        cl, cl_edge, emb_out,lmk_out = self.head(x)
        return cl, cl_edge, emb_out, lmk_out