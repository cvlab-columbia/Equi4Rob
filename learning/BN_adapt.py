from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor
import random
from learning.transform_geo import rot_img

def norm2(v):
    v = v / (torch.sum(v**2, dim=1, keepdim=True)**0.5 + 1e-10)
    return v

def BN_Equi_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False):
    '''
    Generates recalibration using equivariance

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
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    tensor_std=tensor_std.cuda()
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()


    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std


    x_adv = Variable(x_adv, requires_grad=True)
    ds = [Variable(torch.ones((1,)).cuda(), requires_grad=True) for _ in range(4)]
    os = [Variable(torch.zeros((1,)).cuda(), requires_grad=True) for _ in range(4)]

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports

        angle_r = 15

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
        w_s = 0.3
        theta_1 = 1
        x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

        w_s = random.uniform(0.5, 2)
        ratio_s = random.uniform(0.8, 1.2)
        theta_3 = random.uniform(-angle_r, angle_r)/180*3.1415926
        w_s = 2
        theta_1 = 0.25
        x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

        # s = random.uniform(0.5, 2)
        w_s = random.uniform(0.5, 2)
        ratio_s = random.uniform(0.8, 1.2)
        w_s = 0.5
        theta_1 = 4
        x_4 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')

        theta_5 = random.uniform(-angle_r, angle_r)/180*3.1415926
        x_5 = rot_img(x_adv, theta_5, x_1.dtype)

        theta_6 = random.uniform(-angle_r, angle_r)/180*3.1415926
        x_6 = rot_img(x_adv, theta_6, x_1.dtype)

        f0 = net_list[0](x_adv, ds, os)[1]
        f1 = net_list[1](x_1, ds, os)[1]
        f2 = net_list[2](x_2, ds, os)[1]
        f3 = net_list[3](x_3, ds, os)[1]

        f4 = net_list[4](x_4, ds, os)[1]
        f5 = net_list[5](x_5, ds, os)[1]
        f6 = net_list[6](x_6, ds, os)[1]

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

        f_list_raw = [f0, f1, f2, f3, f4, f5, f6]
        # f_list_norm_channel = [norm_channel(e) in e in f_list]

        f_list = [norm2(torch.flatten(e, start_dim=1)) for e in f_list_raw]

        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)
        # print('cost', i, cost)

        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        # TODO: add noinse to grad like Lagine dynamic
        # gradient = x_adv.grad
        # gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) #+ torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
        # if SGD_wN:
        #     gradient = gradient + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps

        # gradient.sign_()
        # x_adv = x_adv - step_size_tensor * gradient# x_adv.grad  # use minus to gradient descent

        # x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        g_ds = [each.grad for each in ds]
        g_os = [each.grad for each in os]

        lr = 0.0001
        ds = [v-g*lr for g, v in zip(g_ds, ds)]
        os = [v-g*lr for g, v in zip(g_os, os)]

        ds = [Variable(e.data, requires_grad=True) for e in ds]
        os = [Variable(e.data, requires_grad=True) for e in os]
    
    return ds, os


def BN_Equi_Defense_2(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False):
    '''
    Generates recalibration using equivariance

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
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    tensor_std=tensor_std.cuda()
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()


    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()


    x_adv = Variable(x_adv, requires_grad=True)
    ds = [Variable(torch.ones((1,)).cuda(), requires_grad=True) for _ in range(4)]
    os = [Variable(torch.zeros((1,)).cuda(), requires_grad=True) for _ in range(4)]

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports

        angle_r = 15

        w_s = random.uniform(0.3, 2)
        ratio_s = random.uniform(0.8, 1.2)
        x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.

        f0 = net_list[0](x_adv, ds, os)[1]
        f1 = net_list[1](x_1, ds, os)[1]    
        f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here


        f_list_raw = [f0, f1]
        # f_list_norm_channel = [norm_channel(e) in e in f_list]

        f_list = [norm2(torch.flatten(e, start_dim=1)) for e in f_list_raw]

        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b, dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)
        # print('cost', i, cost)

        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        
        # TODO: add noinse to grad like Lagine dynamic
        # gradient = x_adv.grad
        # gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) #+ torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
        # if SGD_wN:
        #     gradient = gradient + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps

        # gradient.sign_()
        # x_adv = x_adv - step_size_tensor * gradient# x_adv.grad  # use minus to gradient descent

        # x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        g_ds = [each.grad for each in ds]
        g_os = [each.grad for each in os]

        lr = 0.0001
        ds = [v-g*lr for g, v in zip(g_ds, ds)]
        os = [v-g*lr for g, v in zip(g_os, os)]

        ds = [Variable(e.data, requires_grad=True) for e in ds]
        os = [Variable(e.data, requires_grad=True) for e in os]

        # print('cost', i, cost.item())
    
    return ds, os


def BN_Equi_Defense_4(x, y, net_list, ds, os, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False):
    '''
    Generates recalibration using equivariance

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
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    tensor_std=tensor_std.cuda()
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()


    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()



    x_adv = Variable(x_adv, requires_grad=True)
    ds = [Variable(eds.detach().cuda(), requires_grad=True) for eds in ds]
    os = [Variable(eos.detach().cuda(), requires_grad=True) for eos in os]

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports

        angle_r = 15

        layer_ind = 0

        w_s = random.uniform(0.3, 2)
        ratio_s = random.uniform(0.8, 1.2)
        x_1 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.

        w_s = random.uniform(0.3, 2)
        ratio_s = random.uniform(0.8, 1.2)
        x_2 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.

        w_s = random.uniform(0.3, 2)
        ratio_s = random.uniform(0.8, 1.2)
        x_3 = torch.nn.functional.upsample(x_adv, size=(int(x_adv.size(2)*w_s), int(x_adv.size(3)*w_s*ratio_s)), mode='bilinear')  # TODO: bug, both are w_s now.

        f0 = net_list[0](x_adv, ds, os)[layer_ind]
        f1 = net_list[1](x_1, ds, os)[layer_ind]    
        f2 = net_list[2](x_2, ds, os)[layer_ind]   
        f3 = net_list[3](x_3, ds, os)[layer_ind]   
        f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
        f2 = torch.nn.functional.upsample(f2, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
        f3 = torch.nn.functional.upsample(f3, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here


        f_list_raw = [f0, f1, f2, f3]
        # f_list_norm_channel = [norm_channel(e) in e in f_list]

        f_list = [norm2(torch.flatten(e, start_dim=1)) for e in f_list_raw]

        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b, dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)
        # print('cost', i, cost)

        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        # TODO: add noinse to grad like Lagine dynamic
        # gradient = x_adv.grad
        # gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) #+ torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
        # if SGD_wN:
        #     gradient = gradient + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps

        # gradient.sign_()
        # x_adv = x_adv - step_size_tensor * gradient# x_adv.grad  # use minus to gradient descent

        # x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        # print('grad norm', torch.norm(ds[0].grad))
        g_ds = [each.grad + torch.normal(mean=torch.zeros_like(each.grad), std=torch.ones_like(each.grad)*0.1) for each in ds]
        g_os = [each.grad for each in os]

        lr = 0.0001 #* (steps-i)/steps

        ds = [v-g*lr for g, v in zip(g_ds, ds)]
        os = [v-g*lr for g, v in zip(g_os, os)]

        ds = [Variable(e.data, requires_grad=True) for e in ds]
        os = [Variable(e.data, requires_grad=True) for e in os]
    
    return ds, os