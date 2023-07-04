
from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor
import random
from learning.transform_geo import rot_img
from learning.utils_learn import fast_hist, per_class_iu
import numpy as np



def norm2(v):
    v = v / (torch.sum(v**2, dim=1, keepdim=True)**0.5 + 1e-10)
    return v


def Equi_Score(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=1

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.
    # x_1_mask = torch.ones_like(x_adv)
    # x_1_mask = rot_img(rot_img(x_1_mask, theta_1, x_1.dtype), -theta_1, x_1.dtype)
    # x_1 = rot_img(x_1, theta_1, x_1.dtype)

    # print(torch.sum((x_1_mask>=0.95), dim=[0,1,2,3]), 'mask is 0')


    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_2_mask = torch.ones_like(x_adv)
    # x_2_mask = rot_img(rot_img(x_2_mask, theta_2, x_2.dtype), -theta_2, x_2.dtype)
    # x_2 = rot_img(x_2, theta_2, x_2.dtype)

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_3_mask = torch.ones_like(x_adv)
    # x_3_mask = rot_img(rot_img(x_3_mask, theta_3, x_3.dtype), -theta_3, x_3.dtype)
    # x_3 = rot_img(x_3, theta_3, x_3.dtype)

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_5 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_6 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    f5 = rot_img(f5, -theta_5, f5.dtype)
    x_mask = torch.ones_like(f0)
    x_mask5 = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)

    f6 = rot_img(f6, -theta_6, f0.dtype)
    x_mask = torch.ones_like(f0)
    x_mask6 = rot_img(rot_img(x_mask, theta_6, f0.dtype), -theta_6, f0.dtype)

    mask_list=[torch.flatten(torch.ones_like(f0), start_dim=1) for i in range(5)] + [torch.flatten(x_mask5, start_dim=1)>0.5, torch.flatten(x_mask6, start_dim=1)>0.5]

    f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
    f2 = torch.nn.functional.upsample(f2, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f3 = torch.nn.functional.upsample(f3, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f5 = torch.nn.functional.upsample(f5, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f6 = torch.nn.functional.upsample(f6, size=(f0.size(2), f0.size(3)), mode='bilinear')

    # f1 = rot_img(f1, -theta_1, f1.dtype)
    # f2 = rot_img(f2, -theta_2, f2.dtype)
    # f3 = rot_img(f3, -theta_3, f3.dtype)

    f_list_raw = [f0, f1, f2, f3, f4, f5, f6]
    # f_list_norm_channel = [norm_channel(e) in e in f_list]

    f_list = [norm2(torch.flatten(e, start_dim=1)) for e in f_list_raw]

    # TODO: add negative.
    def paired_loss(in_list):
        loss=0
        for i, a in enumerate(in_list):
            for j, b in enumerate(in_list):
                if i<j:
                    loss = loss + torch.sum(a*b*mask_list[i]*mask_list[j], dim=[1])  # TODO: use contrastive loss in channel
        return loss

    cost = paired_loss(f_list)

    return cost.detach().cpu().numpy() # should be [batch] vector


def Equi_Score_w_overlap(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True, num_classes=19):
    '''
    This produces the best ROC curve.
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=0

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.
    # x_1_mask = torch.ones_like(x_adv)
    # x_1_mask = rot_img(rot_img(x_1_mask, theta_1, x_1.dtype), -theta_1, x_1.dtype)
    # x_1 = rot_img(x_1, theta_1, x_1.dtype)

    # print(torch.sum((x_1_mask>=0.95), dim=[0,1,2,3]), 'mask is 0')


    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_2_mask = torch.ones_like(x_adv)
    # x_2_mask = rot_img(rot_img(x_2_mask, theta_2, x_2.dtype), -theta_2, x_2.dtype)
    # x_2 = rot_img(x_2, theta_2, x_2.dtype)

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_3_mask = torch.ones_like(x_adv)
    # x_3_mask = rot_img(rot_img(x_3_mask, theta_3, x_3.dtype), -theta_3, x_3.dtype)
    # x_3 = rot_img(x_3, theta_3, x_3.dtype)

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_5 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_6 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    f5 = rot_img(f5, -theta_5, f5.dtype)
    x_mask = torch.ones_like(f0)
    x_mask5 = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)

    f6 = rot_img(f6, -theta_6, f0.dtype)
    x_mask = torch.ones_like(f0)
    x_mask6 = rot_img(rot_img(x_mask, theta_6, f0.dtype), -theta_6, f0.dtype)

    mask_list=[torch.flatten(torch.ones_like(f0), start_dim=1) for i in range(5)] + [torch.flatten(x_mask5, start_dim=1)>0.5, torch.flatten(x_mask6, start_dim=1)>0.5]

    f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
    f2 = torch.nn.functional.upsample(f2, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f3 = torch.nn.functional.upsample(f3, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f5 = torch.nn.functional.upsample(f5, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f6 = torch.nn.functional.upsample(f6, size=(f0.size(2), f0.size(3)), mode='bilinear')

    # f1 = rot_img(f1, -theta_1, f1.dtype)
    # f2 = rot_img(f2, -theta_2, f2.dtype)
    # f3 = rot_img(f3, -theta_3, f3.dtype)

    f_list_raw = [f1, f2, f3, f4, f5, f6]
    # f_list_norm_channel = [norm_channel(e) in e in f_list]

    gt = torch.argmax(f0, 1)
    gt = gt.cpu().numpy()
    # print('gt', gt)
    # import pdb; pdb.set_trace()

    hist = [np.zeros((num_classes, num_classes)) for _ in range(gt.shape[0])]

    mAP_list = []
    for bs in range(gt.shape[0]):
        for f in f_list_raw:
            _, pred = torch.max(f, 1)  # TODO: check why not argmax

            pred = pred.cpu().numpy() if torch.cuda.is_available() else pred.numpy()
            tmp_p = pred[bs]
            tmp_gt = gt[bs]
            hist[bs] += fast_hist(tmp_p.flatten(), tmp_gt.flatten(), num_classes)
        
        mAP = round(np.nanmean(per_class_iu(hist[bs])) * 100, 2)
        mAP_list.append(mAP)

    return np.asarray(mAP_list)


def Equi_Score_onehot_var(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True, num_classes=19):
    '''
    Use onehot variance hao suggested.
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=0

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    f5 = rot_img(f5, -theta_5, f5.dtype)
    x_mask = torch.ones_like(f0)
    x_mask5 = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)

    f6 = rot_img(f6, -theta_6, f0.dtype)
    x_mask = torch.ones_like(f0)
    x_mask6 = rot_img(rot_img(x_mask, theta_6, f0.dtype), -theta_6, f0.dtype)

    mask_list=[torch.flatten(torch.ones_like(f0), start_dim=1) for i in range(5)] + [torch.flatten(x_mask5, start_dim=1)>0.5, torch.flatten(x_mask6, start_dim=1)>0.5]

    f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
    f2 = torch.nn.functional.upsample(f2, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f3 = torch.nn.functional.upsample(f3, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f5 = torch.nn.functional.upsample(f5, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f6 = torch.nn.functional.upsample(f6, size=(f0.size(2), f0.size(3)), mode='bilinear')

    # f1 = rot_img(f1, -theta_1, f1.dtype)
    # f2 = rot_img(f2, -theta_2, f2.dtype)
    # f3 = rot_img(f3, -theta_3, f3.dtype)

    f_list_raw = [f1, f2, f3, f4, f5, f6]
    # f_list_norm_channel = [norm_channel(e) in e in f_list]

    gt = torch.argmax(f0, 1).long()
    # one_hot = torch.nn.functional.one_hot(gt, )
    # gt = gt.cpu().numpy()
    # print('gt', gt)
    # import pdb; pdb.set_trace()

    # hist = [np.zeros((num_classes, num_classes)) for _ in range(gt.shape[0])]

    var_list = []
    for bs in range(gt.shape[0]):
        gt_example = gt[bs]
        gt_flatten = torch.flatten(gt_example)
        one_hot_gt = torch.nn.functional.one_hot(gt_flatten, num_classes=19)
        var = 0
        for f in f_list_raw:
            # _, pred = torch.max(f, 1)
            eq_gt = torch.argmax(f, 1).long()
            eq_gt_example = eq_gt[bs]
            eq_gt_flatten = torch.flatten(eq_gt_example)
            eq_one_hot_gt = torch.nn.functional.one_hot(eq_gt_flatten, num_classes=19)

            var = var + torch.sum((eq_one_hot_gt - one_hot_gt)**2)
        
        var_list.append(var.item())

    return np.asarray(var_list)


def Equi_Score_w_var(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True, num_classes=19):
    '''
    This produces the best ROC curve.
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=0

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.
    # x_1_mask = torch.ones_like(x_adv)
    # x_1_mask = rot_img(rot_img(x_1_mask, theta_1, x_1.dtype), -theta_1, x_1.dtype)
    # x_1 = rot_img(x_1, theta_1, x_1.dtype)

    # print(torch.sum((x_1_mask>=0.95), dim=[0,1,2,3]), 'mask is 0')


    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_2_mask = torch.ones_like(x_adv)
    # x_2_mask = rot_img(rot_img(x_2_mask, theta_2, x_2.dtype), -theta_2, x_2.dtype)
    # x_2 = rot_img(x_2, theta_2, x_2.dtype)

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_3_mask = torch.ones_like(x_adv)
    # x_3_mask = rot_img(rot_img(x_3_mask, theta_3, x_3.dtype), -theta_3, x_3.dtype)
    # x_3 = rot_img(x_3, theta_3, x_3.dtype)

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_5 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_6 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    f5 = rot_img(f5, -theta_5, f5.dtype)
    x_mask = torch.ones_like(f0)
    x_mask5 = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)

    f6 = rot_img(f6, -theta_6, f0.dtype)
    x_mask = torch.ones_like(f0)
    x_mask6 = rot_img(rot_img(x_mask, theta_6, f0.dtype), -theta_6, f0.dtype)

    mask_list=[torch.flatten(torch.ones_like(f0), start_dim=1) for i in range(5)] + [torch.flatten(x_mask5, start_dim=1)>0.5, torch.flatten(x_mask6, start_dim=1)>0.5]

    f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
    f2 = torch.nn.functional.upsample(f2, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f3 = torch.nn.functional.upsample(f3, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f5 = torch.nn.functional.upsample(f5, size=(f0.size(2), f0.size(3)), mode='bilinear')
    f6 = torch.nn.functional.upsample(f6, size=(f0.size(2), f0.size(3)), mode='bilinear')

    f_list_raw = [f1, f2, f3, f4, f5, f6]

    variance = [0 for _ in range(f0.size(0))]

    for bs in range(f0.size(0)):
        for f in f_list_raw:
            tmp_f = f[bs]
            # print('tmp f ', tmp_f.size())
            # tmp_f = tmp_f.view((tmp_f.size(0), tmp_f.size(1) * tmp_f.size(2)))
            tmp_f0 = f0[bs]
            # tmp_f0 = tmp_f0.view((tmp_f.size(0), tmp_f.size(1) * tmp_f.size(2)))

            var = torch.mean((tmp_f - tmp_f0)**2)
            variance[bs] = variance[bs] + var.item()

    return np.asarray(variance)


def Contrastive_Var_Score(x, y, net_list, ssl_net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=0

    angle_r = 15
    new_x = x
    X_transformed1 = scripted_transforms(new_x)
    X_transformed2 = scripted_transforms(new_x)
    X_transformed3 = scripted_transforms(new_x)
    X_transformed4 = scripted_transforms(new_x)

    x_list = [X_transformed1, X_transformed2, X_transformed3, X_transformed4]
    out_list = []

    for each in x_list:
        f0 = net_list[0](innormalize(each))[layer_ind]
        out = torch.nn.functional.adaptive_avg_pool2d(f0, output_size=(1,1))
        out = torch.flatten(out, start_dim=1)

        output = ssl_net(out)
        out_list.append(output)

    # f_list_raw = [f0, f1, f2, f3, f4, f5, f6]
    # f_list = [torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1)) for f1 in f_list_raw]
    # f_list = [norm2(torch.flatten(f1, start_dim=1)) for f1 in f_list]

    variance = [0 for _ in range(f0.size(0))]

    for bs in range(out_list[0].size(0)):
        for f in out_list[1:]:
            tmp_f = f[bs]
            # print('tmp f ', tmp_f.size())
            # tmp_f = tmp_f.view((tmp_f.size(0), tmp_f.size(1) * tmp_f.size(2)))
            tmp_f0 = out_list[0][bs]
            # tmp_f0 = tmp_f0.view((tmp_f.size(0), tmp_f.size(1) * tmp_f.size(2)))

            var = torch.mean((tmp_f - tmp_f0)**2)
            variance[bs] = variance[bs] + var.item()


    return np.asarray(variance)


def Inv_Score(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=1

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.
    # x_1_mask = torch.ones_like(x_adv)
    # x_1_mask = rot_img(rot_img(x_1_mask, theta_1, x_1.dtype), -theta_1, x_1.dtype)
    # x_1 = rot_img(x_1, theta_1, x_1.dtype)

    # print(torch.sum((x_1_mask>=0.95), dim=[0,1,2,3]), 'mask is 0')


    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_2_mask = torch.ones_like(x_adv)
    # x_2_mask = rot_img(rot_img(x_2_mask, theta_2, x_2.dtype), -theta_2, x_2.dtype)
    # x_2 = rot_img(x_2, theta_2, x_2.dtype)

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_3_mask = torch.ones_like(x_adv)
    # x_3_mask = rot_img(rot_img(x_3_mask, theta_3, x_3.dtype), -theta_3, x_3.dtype)
    # x_3 = rot_img(x_3, theta_3, x_3.dtype)

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_5 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_6 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    # f5 = rot_img(f5, -theta_5, f5.dtype)
    # x_mask = torch.ones_like(f0)
    # x_mask5 = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)

    # f6 = rot_img(f6, -theta_6, f0.dtype)
    # x_mask = torch.ones_like(f0)
    # x_mask6 = rot_img(rot_img(x_mask, theta_6, f0.dtype), -theta_6, f0.dtype)


    # f1 = rot_img(f1, -theta_1, f1.dtype)
    # f2 = rot_img(f2, -theta_2, f2.dtype)
    # f3 = rot_img(f3, -theta_3, f3.dtype)

    f_list_raw = [f0, f1, f2, f3, f4, f5, f6]
    f_list = [torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1)) for f1 in f_list_raw]
    f_list = [norm2(torch.flatten(f1, start_dim=1)) for f1 in f_list]


    # TODO: add negative.
    def paired_loss(in_list):
        loss=0
        for i, a in enumerate(in_list):
            for j, b in enumerate(in_list):
                if i<j:
                    loss = loss + torch.sum(a*b, dim=[1])  # TODO: use contrastive loss in channel
        return loss

    cost = paired_loss(f_list)

    return cost.detach().cpu().numpy()


def Inv_Score_var(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=0 # use last layer.

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.
    # x_1_mask = torch.ones_like(x_adv)
    # x_1_mask = rot_img(rot_img(x_1_mask, theta_1, x_1.dtype), -theta_1, x_1.dtype)
    # x_1 = rot_img(x_1, theta_1, x_1.dtype)

    # print(torch.sum((x_1_mask>=0.95), dim=[0,1,2,3]), 'mask is 0')


    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_2_mask = torch.ones_like(x_adv)
    # x_2_mask = rot_img(rot_img(x_2_mask, theta_2, x_2.dtype), -theta_2, x_2.dtype)
    # x_2 = rot_img(x_2, theta_2, x_2.dtype)

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_3_mask = torch.ones_like(x_adv)
    # x_3_mask = rot_img(rot_img(x_3_mask, theta_3, x_3.dtype), -theta_3, x_3.dtype)
    # x_3 = rot_img(x_3, theta_3, x_3.dtype)

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_5 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_6 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    f_list_raw = [f0, f1, f2, f3, f4, f5, f6]
    f_list = [torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1)) for f1 in f_list_raw]
    f_list = [norm2(torch.flatten(f1, start_dim=1)) for f1 in f_list]

    variance = [0 for _ in range(f0.size(0))]

    for bs in range(f0.size(0)):
        for f in f_list[1:]:
            tmp_f = f[bs]
            # print('tmp f ', tmp_f.size())
            # tmp_f = tmp_f.view((tmp_f.size(0), tmp_f.size(1) * tmp_f.size(2)))
            tmp_f0 = f_list[0][bs]
            # tmp_f0 = tmp_f0.view((tmp_f.size(0), tmp_f.size(1) * tmp_f.size(2)))

            var = torch.mean((tmp_f - tmp_f0)**2)
            variance[bs] = variance[bs] + var.item()

    return np.asarray(variance)


def Inv_Score_OneHot_var(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
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
    
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()

    layer_ind=0 # use last layer.

    angle_r = 15

    x_adv = x

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 1
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.
    # x_1_mask = torch.ones_like(x_adv)
    # x_1_mask = rot_img(rot_img(x_1_mask, theta_1, x_1.dtype), -theta_1, x_1.dtype)
    # x_1 = rot_img(x_1, theta_1, x_1.dtype)

    # print(torch.sum((x_1_mask>=0.95), dim=[0,1,2,3]), 'mask is 0')


    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_2 = random.uniform(-angle_r, angle_r)/180*3.1415926
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    w_s = 0.3
    theta_1 = 1
    x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_2_mask = torch.ones_like(x_adv)
    # x_2_mask = rot_img(rot_img(x_2_mask, theta_2, x_2.dtype), -theta_2, x_2.dtype)
    # x_2 = rot_img(x_2, theta_2, x_2.dtype)

    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
    w_s = 2
    theta_1 = 0.25
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')
    # x_3_mask = torch.ones_like(x_adv)
    # x_3_mask = rot_img(rot_img(x_3_mask, theta_3, x_3.dtype), -theta_3, x_3.dtype)
    # x_3 = rot_img(x_3, theta_3, x_3.dtype)

    # s = random.uniform(0.5, 2)
    w_s = random.uniform(0.5, 2)
    ratio_s = random.uniform(0.8, 1.2)
    w_s = 0.5
    theta_1 = 4
    # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_5 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_5 = rot_img(x_adv, theta_5, x_1.dtype)

    # w_s = random.uniform(0.2, 0.5)
    # ratio_s = random.uniform(0.8, 1.2)
    # # x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
    # x_6 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

    theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
    x_6 = rot_img(x_adv, theta_6, x_1.dtype)
    
    # print('x4', x_4.size())
    f0 = net_list[0](innormalize(x_adv))[layer_ind]
    f1 = net_list[1](innormalize(x_1))[layer_ind]
    f2 = net_list[2](innormalize(x_2))[layer_ind]
    f3 = net_list[3](innormalize(x_3))[layer_ind]

    f4 = net_list[4](innormalize(x_4))[layer_ind]
    f5 = net_list[5](innormalize(x_5))[layer_ind]
    f6 = net_list[6](innormalize(x_6))[layer_ind]

    f_list_raw = [f0, f1, f2, f3, f4, f5, f6]
    f_list = [torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1)) for f1 in f_list_raw]

    variance = [0 for _ in range(f0.size(0))]

    gt = torch.argmax(f_list[0], 1).long()
    var_list=[]
    for bs in range(f0.size(0)):
        
        gt_example = gt[bs]
        gt_flatten = torch.flatten(gt_example)
        one_hot_gt = torch.nn.functional.one_hot(gt_flatten, num_classes=19)
        var = 0

        for f in f_list[1:]:
            eq_gt = torch.argmax(f, 1).long()
            eq_gt_example = eq_gt[bs]
            eq_gt_flatten = torch.flatten(eq_gt_example)
            eq_one_hot_gt = torch.nn.functional.one_hot(eq_gt_flatten, num_classes=19)

            var = var + torch.sum((eq_one_hot_gt - one_hot_gt)**2)
        
        var_list.append(var.item())

    return np.asarray(variance)

