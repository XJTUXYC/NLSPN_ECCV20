"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPN implementation
"""


from .common import *
from .modulated_deform_conv_func import ModulatedDeformConvFunction
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLSPNModel(nn.Module):
    def __init__(self, args):
        super(NLSPNModel, self).__init__()

        self.args = args
        
        assert (self.args.prop_kernel % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(self.args.prop_kernel)

        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 32, kernel=3, stride=1, bn=False)
        self.conv1_dep = conv_bn_relu(1, 32, kernel=3, stride=1, bn=False)

        if self.args.network == 'resnet18':
            net = get_resnet18(not self.args.from_scratch)
        elif self.args.network == 'resnet34':
            net = get_resnet34(not self.args.from_scratch)
        else:
            raise NotImplementedError

        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3

        del net

        # 1/8
        self.conv5 = conv_bn_relu(256, 256, kernel=3, stride=2)

        # Shared Decoder
        # 1/4
        self.dec4 = convt_bn_relu(256, 128, kernel=3, stride=2, padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2, padding=1, output_padding=1)
        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2, padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1, bn=False, relu=True)

        # Off_Aff Branch
        # 1/1
        self.off_aff_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1)
        if self.args.offset:
            self.off_aff_dec0 = conv_bn_relu(64+64, 3*self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=self.args.zero_init_aff)
        else:
            self.off_aff_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=self.args.zero_init_aff)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(64+64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        self.ch_f = 1
        
        # Assume zero offset for center pixels
        self.idx_ref = self.num_neighbors // 2

        if self.args.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            if self.args.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num_neighbors * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.args.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num_neighbors * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.args.prop_kernel, self.args.prop_kernel)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = int((self.args.prop_kernel - 1) / 2)
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        
        if self.args.use_GRU:
            self.GRU = ConvGRU(args)
            
            # TODO: delet it
            self.encode_aff = nn.Sequential(
                conv_bn_relu(self.num_neighbors+1, 16, kernel=3, stride=2, bn=False),
                conv_bn_relu(16, 2*args.GRU_hidden_dim, kernel=3, stride=2, bn=False),
                conv_bn_relu(2*args.GRU_hidden_dim, args.GRU_hidden_dim, kernel=3, stride=2, bn=False, relu=False),
                nn.Tanh()
            )
            
            self.encode_dep = nn.Sequential(
                conv_bn_relu(1, 16, kernel=3, stride=2, bn=False),
                conv_bn_relu(16, 2*args.GRU_input_dim, kernel=3, stride=2, bn=False),
                conv_bn_relu(2*args.GRU_input_dim, args.GRU_input_dim, kernel=3, stride=2, bn=False)
            )
            
            self.decode_aff = nn.Sequential(
                convt_bn_relu(args.GRU_hidden_dim, 2*args.GRU_hidden_dim, kernel=3, stride=2, padding=1, output_padding=1, bn=False),
                convt_bn_relu(2*args.GRU_hidden_dim, 16, kernel=3, stride=2, padding=1, output_padding=1, bn=False),
                convt_bn_relu(16, self.num_neighbors, kernel=3, stride=2, padding=1, output_padding=1, bn=False, relu=False, zero_init=self.args.zero_init_aff)
            )

        if self.args.use_S2D:
            self.S2D = S2D()

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def _affinity_normalization(self, aff):
        if self.args.affinity in ['AS', 'ASS']:
            pass
        elif self.args.affinity == 'TC':
            aff = torch.tanh(aff) / self.aff_scale_const
        elif self.args.affinity == 'TGASS':
            aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
        else:
            raise NotImplementedError

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.args.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.args.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff = self._aff_insert(aff)
        
        return aff
    
    def _propagate_once(self, feat, offset, aff):
        if offset is not None:
            feat = ModulatedDeformConvFunction.apply(
                feat, offset, aff, self.w, self.b, self.stride, self.padding,
                self.dilation, self.groups, self.deformable_groups, self.im2col_step
            )
        else:
            feat = F.pad(feat, (1,1,1,1), mode="replicate")
            _, _, H, W = feat.size()
            feat_list = []
            feat_list.append(feat[:, :, 0:H-2, 0:W-2])
            feat_list.append(feat[:, :, 0:H-2, 1:W-1])
            feat_list.append(feat[:, :, 0:H-2, 2:W-0])
            feat_list.append(feat[:, :, 1:H-1, 0:W-2])
            feat_list.append(feat[:, :, 1:H-1, 1:W-1])
            feat_list.append(feat[:, :, 1:H-1, 2:W-0])
            feat_list.append(feat[:, :, 2:H-0, 0:W-2])
            feat_list.append(feat[:, :, 2:H-0, 1:W-1])
            feat_list.append(feat[:, :, 2:H-0, 2:W-0])
            feat = torch.cat(feat_list, dim=1)
            feat = feat * aff
            feat = torch.sum(feat, dim=1, keepdim=True)
            
        return feat
    
    def _aff_head(self, aff_feat):
        aff = self.decode_aff(aff_feat)
        
        aff = self._clip_as(aff, self.args.patch_height, self.args.patch_width)
        aff = self._affinity_normalization(aff)
        
        return aff
    
    # TODO: center clip
    def _clip_as(self, fd, He, We, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        return fd
    
    def _off_insert(self, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, self.num_neighbors, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num_neighbors, dim=1))
        list_offset.insert(self.idx_ref, torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)
        
        return offset
    
    def _aff_insert(self, aff):
        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num_neighbors, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)
        
        return aff

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb) # b*32*H*W
        if self.args.use_S2D:
            fe1_dep = self.S2D(dep) # b*32*H*W
        else:
            fe1_dep = self.conv1_dep(dep) # b*32*H*W

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1) # b*64*H*W

        fe2 = self.conv2(fe1) # b*64*H*W
        fe3 = self.conv3(fe2) # b*128*H/2*W/2
        fe4 = self.conv4(fe3) # b*256*H/4*W/4
        
        fe5 = self.conv5(fe4) # b*256*H/8*W/8

        # Shared Decoding
        fd4 = self.dec4(fe5) # b*128*H/4*W/4
        fd3 = self.dec3(self._concat(fd4, fe4)) # b*(128+256)*H/8*W/8 -> b*64*H/4*W/4
        fd2 = self.dec2(self._concat(fd3, fe3)) # b*(64+128)*H/4*W/4 -> b*64*H/2*W/2

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2)) # b*(64+64)*H*W -> b*64*H*W
        pred_init = self.id_dec0(self._concat(id_fd1, fe1)) # b*(64+64)*H*W -> b*1*H*W

        # Off Aff Decoding
        off_aff_fd1 = self.off_aff_dec1(self._concat(fd2, fe2)) # b*128*H*W -> b*64*H*W
        off_aff = self.off_aff_dec0(self._concat(off_aff_fd1, fe1)) # b*128*H*W -> b*24*H*W
        
        if self.args.offset:
            off = off_aff[:, :2*self.num_neighbors, :, :] # b*16*H*W
            aff = off_aff[:, 2*self.num_neighbors:, :, :] # b*8*H*W
        else:
            off = None
            aff = off_aff

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2)) # b*128*H*W -> b*32*H*W
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1)) # b*96*H*W -> b*1*H*W
        else:
            confidence = None

        # Diffusion
        assert self.ch_f == pred_init.shape[1]

        if self.args.conf_prop:
            assert confidence is not None

        if off is not None:
            off = self._off_insert(off) # b*18*H*W
        aff = self._affinity_normalization(aff) # b*9*H*W

        # Propagation
        if self.args.preserve_input:
            assert pred_init.shape == dep.shape
            mask_fix = torch.sum(dep > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(dep)
            
            if confidence is not None:
                confidence = (1.0 - mask_fix) * confidence + mask_fix

        new_pred = pred_init

        list_pred = []

        for k in range(1, self.args.prop_time + 1):
            if k == 1:
                if self.args.preserve_input:
                    # Input preservation for each iteration
                    new_pred = (1.0 - mask_fix) * new_pred + mask_fix * dep
                
                if self.args.always_clip:
                    # Remove negative depth
                    new_pred = torch.clamp(new_pred, min=0)
                
            if confidence is not None:
                new_pred = self._propagate_once(new_pred*confidence, off, aff)
            else:
                new_pred = self._propagate_once(new_pred, off, aff)

            if self.args.preserve_input:
                # Input preservation for each iteration
                new_pred = (1.0 - mask_fix) * new_pred + mask_fix * dep
                
            if self.args.always_clip:
                # Remove negative depth
                new_pred = torch.clamp(new_pred, min=0)
                                                     
            list_pred.append(new_pred)
            
            if k<self.args.prop_time and self.args.use_GRU:
                dep_feat = self.encode_dep(new_pred/self.args.max_depth)
                
                if k == 1:
                    aff_feat = self.encode_aff(aff)
    
                aff_feat = self.GRU(h=aff_feat, x=dep_feat)
                
                aff = self._aff_head(aff_feat)
          
        if not self.args.always_clip:
            # Remove negative depth
            new_pred = torch.clamp(new_pred, min=0)

        output = {'pred': new_pred, 'pred_init': pred_init, 'pred_inter': list_pred,
                  'offset': off, 'aff': aff,
                  'gamma': self.aff_scale_const.data, 'confidence': confidence}

        return output
    
    
class ConvGRU(nn.Module):
    def __init__(self, args):
        super(ConvGRU, self).__init__()
        
        self.convz = nn.Conv2d(args.GRU_hidden_dim+args.GRU_input_dim, args.GRU_hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(args.GRU_hidden_dim+args.GRU_input_dim, args.GRU_hidden_dim, 3, padding=1)
        
        self.convq = nn.Conv2d(args.GRU_hidden_dim+args.GRU_input_dim, args.GRU_hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
    
    
class S2D(nn.Module):
    def __init__(self):
        super(S2D, self).__init__()

        self.min_pool_sizes = [3,5,7,9]
        self.max_pool_sizes = [11,13]

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        self.pool_convs = nn.Sequential(
            conv_bn_relu(in_channels, 8, kernel=1, stride=1, bn=False),
            conv_bn_relu(8, 16, kernel=1, stride=1, bn=False)
        )

        self.conv = conv_bn_relu(16+1, 32, kernel=3, stride=1, bn=False)

    def forward(self, dep):
        pool_pyramid = []

        # Use min and max pooling to densify and increase receptive field
        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            # Set flag (999) for any zeros and max pool on -z then revert the values
            z_pool = -pool(torch.where(dep == 0, -999 * torch.ones_like(dep), -dep))
            # Remove any 999 from the results
            z_pool = torch.where(z_pool == 999, torch.zeros_like(dep), z_pool)

            pool_pyramid.append(z_pool)

        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(dep)

            pool_pyramid.append(z_pool)

        # Stack max and minpools into pyramid
        pool_pyramid = torch.cat(pool_pyramid, dim=1)

        # Learn weights for different kernel sizes, and near and far structures
        dep_feat = self.pool_convs(pool_pyramid)

        dep_feat = torch.cat([dep_feat, dep], dim=1)
        dep_feat = self.conv(dep_feat)

        return dep_feat
