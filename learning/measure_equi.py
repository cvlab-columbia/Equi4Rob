from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor
import random
from learning.transform_geo import rot_img
import torch.nn.functional as F

def norm2(v):
    v = v / (torch.sum(v**2, dim=1, keepdim=True)**0.5 + 1e-10)
    return v

def feature_post_norm(f):
    feature = norm2(torch.flatten(f, start_dim=1))
    return feature


def Measure_Equi_Set_Defense_good_inloop(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True, layer_ind=1):
    '''
    , yes, this one can also reproduce, so the good performance is due to the design choice for transformation.
    Reimplement our working reversal with this ICCV adding delta approach, and reproduce the results, thus the 
    code is no bug, the only key to improve acc and reproduce our prior one is to use the same transformations.

    :param x: 
    :param y: 
    :param net: 
    :param Loss: 
    :param epsilon: 
    :param steps: 
    :param dataset: 
    :param step_size: 
    :param info: 
    :param using_noise: 
    :return: 
    '''
    # image need to be among range (0,1) to use scripted transfroms. TODO: czm

    angle_r = 15

    use_CE = False
    if use_CE:
        def innormalize(x):
            return x

    keep_ratio=0.1
    tmp = net_list[0](innormalize(x))[layer_ind]
    global_mask = torch.rand([1, 1, tmp.size(2), tmp.size(3)]).cuda()<keep_ratio



    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True
    x_new = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_new = x_new.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    x_adv = x_new


    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    if use_CE:
        x_mask = torch.ones_like(f0)
        x_mask = x_mask > 0.5
        x_mask = torch.flatten(x_mask, start_dim=2)
        x_mask = x_mask.permute(0, 2, 1)
        x_mask = x_mask.reshape(x_mask.size(0) * x_mask.size(1), x_mask.size(2))

        feature = torch.flatten(f0, start_dim=2)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(feature.size(0) * feature.size(1), feature.size(2))

        f_list = [feature]
        mask_list = [x_mask]
    else:
        f_list = [feature_post_norm(f0 * global_mask)]
        mask_list = [torch.flatten(torch.ones_like(f0), start_dim=1)]
    
    for cnt, each in enumerate(net_list[1:]):
        if cnt==0:
            ratio_s = random.uniform(0.8, 1.2)
            w_s = 2
        if cnt==1:
            w_s = 0.3
            ratio_s = random.uniform(0.8, 1.2)
        if cnt==2:
            w_s = 2
            ratio_s = random.uniform(0.8, 1.2)
        if cnt==3:
            w_s = 0.5
        if cnt<4:
            ratio_s = random.uniform(0.8, 1.2)
            x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
        elif cnt == 6:
            # color and resize
            x_4 = scripted_transforms(x_adv)
            ratio_s = random.uniform(0.8, 1.2)
            x_4 = torch.nn.functional.upsample(x_4, size=(int(x_adv.size(2)), int(x_adv.size(3)*ratio_s)), mode='bilinear')
        elif cnt == 7:
            # flip and resize
            ratio_s = random.uniform(0.8, 1.2)
            x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)), int(x_adv.size(3)*ratio_s)), mode='bilinear')
            x_4 = torch.flip(x_4, dims=[3])
        
        elif cnt == 4 or cnt == 5:
            theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
            x_4 = rot_img(x_adv, theta_5, x_adv.dtype)
        f1 = each(innormalize(x_4))[layer_ind]


        if cnt == 4 or cnt == 5:
            f1 = rot_img(f1, -theta_5, f1.dtype)
            x_mask = torch.ones_like(f0)
            x_mask = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)
        else:
            f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear')
            x_mask = torch.ones_like(f0)
            if cnt == 7:
                f1 = torch.flip(f1, dims=[3])

        if use_CE:
            x_mask = x_mask > 0.5
            x_mask = torch.flatten(x_mask, start_dim=2)
            x_mask = x_mask.permute(0, 2, 1)
            x_mask = x_mask.reshape(x_mask.size(0) * x_mask.size(1), x_mask.size(2))

            feature = torch.flatten(f1, start_dim=2)
            feature = feature.permute(0, 2, 1)
            feature = feature.reshape(feature.size(0) * feature.size(1), feature.size(2))
        else:
            x_mask = torch.flatten(x_mask, start_dim=1)>0.5
            feature = feature_post_norm(f1 * global_mask)

        f_list.append(feature)
        mask_list.append(x_mask)

    # TODO: add negative.
    def paired_loss(in_list):
        loss=0
        tmp_cnt = 0
        for i, a in enumerate(in_list):
            for j, b in enumerate(in_list):
                if i<j:
                    tmp_cnt += 1
                    loss = loss - torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])/a.size(0)  # TODO: use contrastive loss in channel
        return -loss / tmp_cnt

    def paired_loss_categorical(in_list):
        loss=0
        for i, a in enumerate(in_list):
            for j, b in enumerate(in_list):
                # if i<j:
                #     print(a.size(), b.size(), F.log_softmax(a, dim=-1).size(), mask_list[i].size(), mask_list[j].size(), 'feimg')
                    loss = loss + torch.sum(-b.detach()*F.log_softmax(a, dim=-1)*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
        return loss
    if use_CE:
        cost = paired_loss_categorical(f_list)
    else:
        cost = paired_loss(f_list)
        # print('cost', cost.item())

    
    return cost.item()
