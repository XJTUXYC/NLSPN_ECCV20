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

        
class NLSPN(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time
        self.affinity = self.args.affinity

        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            # self.conv_offset_aff = nn.Conv2d(
            #     self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
            #     padding=pad_g, bias=True
            # )
            # self.conv_offset_aff.weight.data.zero_()
            # self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        
        if self.args.use_GRU:
            self.GRU = ConvGRU(args)
            
            self.encode_aff0 = conv_bn_relu(8, 16, kernel=7, stride=2, padding=3, bn=False)
            self.encode_aff1 = conv_bn_relu(16, int(args.GRU_hidden_dim/2), kernel=5, stride=2, padding=2, bn=False)
            self.encode_aff2 = conv_bn_relu(int(args.GRU_hidden_dim/2), args.GRU_hidden_dim, kernel=3, stride=2, padding=1, bn=False, relu=False)
            
            self.encode_dep0 = conv_bn_relu(1, 16, kernel=7, stride=2, padding=3, bn=False)
            self.encode_dep1 = conv_bn_relu(16, int(args.GRU_input_dim/2), kernel=5, stride=2, padding=2, bn=False)
            self.encode_dep2 = conv_bn_relu(int(args.GRU_input_dim/2), args.GRU_input_dim, kernel=3, stride=2, padding=1, bn=False)
            
            # self.encode_aff_dep = conv_bn_relu(args.GRU_input_dim, args.GRU_input_dim, kernel=3, stride=1, padding=1, bn=False)
            
            self.aff_head0 = convt_bn_relu(args.GRU_hidden_dim, int(args.GRU_hidden_dim/2), kernel=3, stride=2, padding=1, output_padding=1, bn=False)
            self.aff_head1 = convt_bn_relu(int(args.GRU_hidden_dim/2), 16, kernel=3, stride=2, padding=1, output_padding=1, bn=False)
            self.aff_head2 = convt_bn_relu(16, 8, kernel=3, stride=2, padding=1, output_padding=1, bn=False, relu=False, zero_init=True)

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            # offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(guidance, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)
        else:
            raise NotImplementedError
        
        aff = self._affinity_normalization(aff)

        return offset, aff
    
    def _affinity_normalization(self, aff):
        if self.affinity in ['AS', 'ASS']:
            pass
        elif self.affinity == 'TC':
            aff = torch.tanh(aff) / self.aff_scale_const
        elif self.affinity == 'TGASS':
            aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
        else:
            raise NotImplementedError

        # # Apply confidence
        # # TODO : Need more efficient way
        # if self.args.conf_prop:
        #     list_conf = []
        #     offset_each = torch.chunk(offset, self.num + 1, dim=1)

        #     modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

        #     for idx_off in range(0, self.num + 1):
        #         ww = idx_off % self.k_f
        #         hh = idx_off // self.k_f

        #         if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
        #             continue

        #         offset_tmp = offset_each[idx_off].detach()

        #         # NOTE : Use --legacy option ONLY for the pre-trained models
        #         # for ECCV20 results.
        #         if self.args.legacy:
        #             offset_tmp[:, 0, :, :] = \
        #                 offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
        #             offset_tmp[:, 1, :, :] = \
        #                 offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

        #         conf_tmp = ModulatedDeformConvFunction.apply(
        #             confidence, offset_tmp, modulation_dummy, self.w_conf,
        #             self.b, self.stride, 0, self.dilation, self.groups,
        #             self.deformable_groups, self.im2col_step)
        #         list_conf.append(conf_tmp)

        #     conf_aff = torch.cat(list_conf, dim=1)
        #     aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)
        
        return aff
    
    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat
    
    def _get_aff_feature(self, aff):
        aff_feat = self.encode_aff0(aff)
        aff_feat = self.encode_aff1(aff_feat)
        aff_feat = torch.tanh(self.encode_aff2(aff_feat))
        
        return aff_feat
    
    def _get_dep_feature(self, dep):     
        dep_feat = self.encode_dep0(dep)
        dep_feat = self.encode_dep1(dep_feat)
        dep_feat = self.encode_dep2(dep_feat)

        # aff_dep_feat = torch.cat([aff_feat, dep_feat], dim=1)
        # aff_dep_feat = self.encode_aff_dep(aff_dep_feat)
        
        return dep_feat
    
    def _aff_head(self, aff_feat):
        aff = self.aff_head0(aff_feat)
        aff = self.aff_head1(aff)
        aff = self.aff_head2(aff)
        
        return aff
    
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

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        assert self.ch_f == feat_init.shape[1]

        if self.args.conf_prop:
            assert confidence is not None

        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)

        # Propagation
        if self.args.preserve_input:
            assert feat_init.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)
            
            if confidence is not None:
                confidence = (1.0 - mask_fix) * confidence + mask_fix

        feat_result = feat_init

        list_feat = []

        for k in range(1, self.prop_time + 1):
            if k==1 and self.args.preserve_input:
                # Input preservation for each iteration
                feat_result = (1.0 - mask_fix) * feat_result + mask_fix * feat_fix
                
            if confidence is not None:
                feat_result = self._propagate_once(feat_result*confidence, offset, aff)
            else:
                feat_result = self._propagate_once(feat_result, offset, aff)

            # Input preservation for each iteration
            if self.args.preserve_input:
                feat_result = (1.0 - mask_fix) * feat_result + mask_fix * feat_fix
                                                     
            list_feat.append(feat_result)
            
            if k<self.prop_time and self.args.use_GRU:
                list_aff = list(torch.chunk(aff, self.num+1, dim=1))
                list_aff.pop(self.idx_ref)
                aff = torch.cat(list_aff, dim=1)
                
                dep_feat = self._get_dep_feature(feat_result)
                if k==1:
                    aff_feat = self._get_aff_feature(aff)
    
                aff_feat = self.GRU(h=aff_feat, x=dep_feat)
                
                aff = self._aff_head(aff_feat)
                aff = self._clip_as(aff, self.args.patch_height, self.args.patch_width)
                aff = self._affinity_normalization(aff)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data


class NLSPNModel(nn.Module):
    def __init__(self, args):
        super(NLSPNModel, self).__init__()

        self.args = args

        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=5, stride=1, padding=2,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=5, stride=1, padding=2,
                                      bn=False)

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
        self.conv5 = conv_bn_relu(256, 256, kernel=3, stride=2, padding=1)

        # Shared Decoder
        # 1/4
        self.dec4 = convt_bn_relu(256, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)

        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, 3*self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False, zero_init=True)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                self.args.prop_kernel)

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

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb) # b*48*H*W
        fe1_dep = self.conv1_dep(dep) # b*16*H*W

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

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2)) # b*128*H*W -> b*64*H*W
        guide = self.gd_dec0(self._concat(gd_fd1, fe1)) # b*128*H*W -> b*24*H*W

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2)) # b*128*H*W -> b*32*H*W
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1)) # b*96*H*W -> b*1*H*W
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, dep, rgb)
            
        # Remove negative depth
        y = torch.clamp(y, min=0)

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': confidence}

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
