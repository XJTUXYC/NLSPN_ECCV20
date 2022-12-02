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


class NLSPNModel(nn.Module):
    def __init__(self, args):
        super(NLSPNModel, self).__init__()

        self.args = args
        
        assert (self.args.prop_kernel % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(self.args.prop_kernel)

        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1    
        
        self.backbone = BackBone(args, output_dim=256)
        
        self.shared_decoder = SharedDecoder()
        
        # Init Depth Branch
        self.id_dec = nn.Sequential(
            conv_bn_relu(64+64, 64, kernel=3, stride=1),
            conv_bn_relu(64, 1, kernel=3, stride=1, bn=False)
        )

        # Off_Aff Branch
        self.off_aff_dec = nn.Sequential(
            conv_bn_relu(64+64, 128, kernel=3, stride=1),
            conv_bn_relu(128, 3*self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=self.args.zero_init_aff)
        )

        # Confidence Branch
        if self.args.conf_prop:
            self.cf_dec = nn.Sequential(
                conv_bn_relu(64+64, 64, kernel=3, stride=1),
                conv_bn_relu(64, 1, kernel=3, stride=1, bn=False, relu=False),
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
        
        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]
    
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
    
    def _propagate_once(self, feat, offset, aff, conf=None):
        if conf is not None:
            feat = ModulatedDeformConvFunction.apply(
                feat*conf, offset, aff, self.w, self.b, self.stride, self.padding,
                self.dilation, self.groups, self.deformable_groups, self.im2col_step)
        else:
            feat = ModulatedDeformConvFunction.apply(
                feat, offset, aff, self.w, self.b, self.stride, self.padding,
                self.dilation, self.groups, self.deformable_groups, self.im2col_step)

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
        fe0, fe1, fe2, fe3 = self.backbone(rgb, dep) # 64/1,64/2,96/4,256/8

        # Shared Decoding
        fd0 = self.shared_decoder(fe0, fe1, fe2, fe3) # b*64*H*W

        # Init Depth Decoding
        init_dep = self.id_dec(fd0) # b*1*H*W
        assert init_dep.shape == dep.shape
        
        if self.args.preserve_input:
            mask_fix = torch.sum(dep > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(dep)
            init_dep = (1.0 - mask_fix) * init_dep + mask_fix * dep
            
        if self.args.always_clip:
            init_dep = torch.clamp(init_dep, min=0)

        # Off Aff Decoding
        off_aff = self.off_aff_dec(fd0) # b*24*H*W
        
        off = off_aff[:, :2*self.num_neighbors, :, :] # b*16*H*W
        off = self._off_insert(off) # b*18*H*W
        aff = off_aff[:, 2*self.num_neighbors:, :, :] # b*8*H*W
        aff = self._affinity_normalization(aff) # b*9*H*W

        # Confidence Decoding
        if self.args.conf_prop:
            confidence = self.cf_dec(fd0) # b*1*H*W
            if self.args.preserve_input:
                confidence = (1.0 - mask_fix) * confidence + mask_fix
        else:
            confidence = None

        # Propagation
        new_dep = init_dep
        list_dep = []

        for k in range(1, self.args.prop_time + 1):
            # DCN
            new_dep = self._propagate_once(new_dep, off, aff, confidence)

            if self.args.preserve_input:
                new_dep = (1.0 - mask_fix) * new_dep + mask_fix * dep
                
            if self.args.always_clip:
                new_dep = torch.clamp(new_dep, min=0)
                                                     
            list_dep.append(new_dep)
            
            # GRU
            if k<self.args.prop_time and self.args.use_GRU:
                dep_feat = self.encode_dep(new_dep/self.args.max_depth)
                
                if k == 1:
                    aff_feat = self.encode_aff(aff)
    
                aff_feat = self.GRU(h=aff_feat, x=dep_feat)
                
                aff = self._aff_head(aff_feat)
          
        if not self.args.always_clip:
            new_dep = torch.clamp(new_dep, min=0)

        output = {'pred': new_dep, 'pred_init': init_dep, 'pred_inter': list_dep,
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
            conv_bn_relu(in_channels, 16, kernel=7, stride=1, bn=False),
            conv_bn_relu(16, 32-1, kernel=3, stride=1, bn=False)
        )
        
        self.conv = conv_bn_relu(32, 32, kernel=3, stride=1, bn=False)

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
    
    
class BackBone(nn.Module):
    def __init__(self, args, output_dim=128):
        super(BackBone, self).__init__()

        self.args = args
        
        self.conv_rgb = conv_bn_relu(3, 32, kernel=7, stride=1, bn=False)
        self.conv_dep = conv_bn_relu(1, 32, kernel=7, stride=1, bn=False)
        if self.args.use_S2D:
            self.S2D = S2D()

        self.conv_rgb_dep = nn.Conv2d(64, 64, stride=1, kernel_size=1)

        self.in_planes = 64
        
        # 1 layer = 2 blocks = 4 convs
        self.layer1 = self._make_layer(64,  stride=2)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.outconv = nn.Conv2d(128, output_dim, stride=1, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, rgb, dep):
        fe0_rgb = self.conv_rgb(rgb) # b*32*H*W
        if self.args.use_S2D:
            fe0_dep = self.S2D(dep) # b*32*H*W
        else:
            fe0_dep = self.conv_dep(dep) # b*32*H*W

        fe0 = self.conv_rgb_dep(torch.cat((fe0_rgb, fe0_dep), dim=1)) # b*64*H*W
        
        fe1 = self.layer1(fe0) # B*64*W/2*H/2
        fe2 = self.layer2(fe1) # B*96*W/4*H/4
        fe3 = self.layer3(fe2) # B*128*W/8*H/8

        fe3 = self.outconv(fe3) # B*256*W/8*H/8

        return fe0, fe1, fe2, fe3
    

class SharedDecoder(nn.Module):
    def __init__(self):
        super(SharedDecoder, self).__init__()
        
        self.dec3 = convt_bn_relu(256, 128, kernel=3, stride=2, padding=1, output_padding=1)
        self.dec2 = convt_bn_relu(128+96, 96, kernel=3, stride=2, padding=1, output_padding=1)
        self.dec1 = convt_bn_relu(96+64, 64, kernel=3, stride=2, padding=1, output_padding=1)

    def forward(self, fe0, fe1, fe2, fe3):
        fd2 = self.dec3(fe3) # b*128*H/4*W/4
        fd1 = self.dec2(concat(fd2, fe2)) # b*(128+96)*H/4*W/4 -> b*96*H/2*W/2
        fd0 = self.dec1(concat(fd1, fe1)) # b*(96+64)*H/2*W/2 -> b*64*H*W
        
        fd0 = concat(fd0, fe0) # b*(64+64)*H*W
        return fd0
    
    

