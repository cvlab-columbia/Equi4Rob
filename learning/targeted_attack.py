from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor
import random
from learning.transform_geo import rot_img

def get_torch_std(info):
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    return tensor_std

def houdini(X, Y, Y_pred, task_loss):
    max_preds, _    = Y_pred.max(axis=1)
    true_preds      = torch.gather(Y_pred,  1, Y.unsqueeze(1)).squeeze(1)

    normal_dist     = torch.distributions.Normal(0.0, 1.0)
    probs           = 1.0 - normal_dist.cdf(true_preds - max_preds)
    loss            = torch.mean(probs * task_loss)

    return loss





def targeted_PGD_attack(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True):
    '''
    Generates attacked image for a single task.

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

    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)
        # elif dataset == 'ade20k':
        #     h_adv = net(x_adv,segSize = (256,256))
        cost = Loss(h_adv[0], y) #TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - step_size_tensor * x_adv.grad # change to minus to gradient descent.
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv


def targeted_PGD_attack_adaptive_inv(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True):
    '''
    Generates attacked that also considers our invariance defense

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

    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        tmp_out = net_list[0](x_adv)

        f0 = tmp_out[1]

        s = random.uniform(0.5, 2)
        x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')
        f4 = net_list[4](x_4)[1]

        # f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
        f4 = torch.nn.functional.adaptive_avg_pool2d(f4, output_size=(1,1))
        f0 = torch.nn.functional.adaptive_avg_pool2d(f0, output_size=(1,1))

        f0 = torch.flatten(f0, start_dim=1)
        # f1 = torch.flatten(f1, start_dim=1)
        # f2 = torch.flatten(f2, start_dim=1)
        # f3 = torch.flatten(f3, start_dim=1)

        f4 = torch.flatten(f4, start_dim=1)

        f0 = norm2(f0)
        # f1 = norm2(f1)
        # f2 = norm2(f2)
        # f3 = norm2(f3)
        f4 = norm2(f4)

        # elif dataset == 'ade20k':
        #     h_adv = net(x_adv,segSize = (256,256))
        cost = Loss(tmp_out[0], y) - torch.sum(f0*f4, dim=[0, 1]) #TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv


def targeted_PGD_attack_adaptive_equi(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, ada_lambda=1):
    '''
    Generates attacked that also considers our equivariance defense

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

    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        tmp_output = net_list[0](x_adv)
        f0 = tmp_output[1]
        
        angle_r = 15

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
        f0 = net_list[0](x_adv)[1]
        f1 = net_list[1](x_1)[1]
        f2 = net_list[2](x_2)[1]
        f3 = net_list[3](x_3)[1]

        f4 = net_list[4](x_4)[1]
        f5 = net_list[5](x_5)[1]
        f6 = net_list[6](x_6)[1]

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

        # f0 = torch.flatten(f0, start_dim=1)
        # f1 = torch.flatten(f1, start_dim=1)
        # f2 = torch.flatten(f2, start_dim=1)
        # f3 = torch.flatten(f3, start_dim=1)

        # f4 = torch.flatten(f4, start_dim=1)

        # f0 = norm2(f0)
        # f1 = norm2(f1)
        # f2 = norm2(f2)
        # f3 = norm2(f3)
        # f4 = norm2(f4)

        # cost = -torch.sum(f0*f1, dim=[0, 1])-torch.sum(f0*f2, dim=[0, 1])-torch.sum(f0*f3, dim=[0, 1])-torch.sum(f0*f4, dim=[0, 1])
        # cost = -torch.sum(f0*f4, dim=[0, 1])

        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss
        
        # def paired_loss(in_list):
        #     loss=0
        #     for i, a in enumerate(in_list):
        #         if i>0:
        #             loss = loss -torch.sum(in_list[0]*a, dim=[0, 1])  # TODO: use contrastive loss in channel
        #     return loss

        equi_cost = paired_loss(f_list)


        cost = Loss(tmp_output[0], y) + equi_cost * ada_lambda#TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv

def norm2(v):
    v = v / (torch.sum(v**2, dim=1, keepdim=True)**0.5 + 1e-10)
    return v

def Rotation_Equi_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False):
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

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

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

        f0 = net_list[0](x_adv)[1]
        f1 = net_list[1](x_1)[1]
        f2 = net_list[2](x_2)[1]
        f3 = net_list[3](x_3)[1]

        f4 = net_list[4](x_4)[1]
        f5 = net_list[5](x_5)[1]
        f6 = net_list[6](x_6)[1]

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
        gradient = x_adv.grad
        gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) #+ torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
        if SGD_wN:
            gradient = gradient + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps

        gradient.sign_()
        x_adv = x_adv - step_size_tensor * gradient# x_adv.grad  # use minus to gradient descent

        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
    
    return x_adv



def Rotation_Equi_Defense8(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False):
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

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

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

        x_7 = torch.flip(x_adv, dims=[3])

        f0 = net_list[0](x_adv)[1]
        f1 = net_list[1](x_1)[1]
        f2 = net_list[2](x_2)[1]
        f3 = net_list[3](x_3)[1]

        f4 = net_list[4](x_4)[1]
        f5 = net_list[5](x_5)[1]
        f6 = net_list[6](x_6)[1]

        f7 = net_list[6](x_7)[1]

        f5 = rot_img(f5, -theta_5, f5.dtype)
        x_mask = torch.ones_like(f0)
        x_mask5 = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)

        f6 = rot_img(f6, -theta_6, f0.dtype)
        x_mask = torch.ones_like(f0)
        x_mask6 = rot_img(rot_img(x_mask, theta_6, f0.dtype), -theta_6, f0.dtype)

        f7 = torch.flip(f7, dims=[3])

        mask_list=[torch.flatten(torch.ones_like(f0), start_dim=1) for i in range(5)] + [torch.flatten(x_mask5, start_dim=1)>0.5, torch.flatten(x_mask6, start_dim=1)>0.5] + [torch.flatten(torch.ones_like(f0), start_dim=1)]

        f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear') # just fix bug here
        f2 = torch.nn.functional.upsample(f2, size=(f0.size(2), f0.size(3)), mode='bilinear')
        f3 = torch.nn.functional.upsample(f3, size=(f0.size(2), f0.size(3)), mode='bilinear')
        f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
        f5 = torch.nn.functional.upsample(f5, size=(f0.size(2), f0.size(3)), mode='bilinear')
        f6 = torch.nn.functional.upsample(f6, size=(f0.size(2), f0.size(3)), mode='bilinear')

        f_list_raw = [f0, f1, f2, f3, f4, f5, f6, f7]
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
        gradient = x_adv.grad
        gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) #+ torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
        if SGD_wN:
            gradient = gradient + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps

        gradient.sign_()
        x_adv = x_adv - step_size_tensor * gradient# x_adv.grad  # use minus to gradient descent

        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
    
    return x_adv

def SGLD_Equi_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, lambda_reg=0.5, feature_used='second_to_last'):
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
    layer_ind=1
    if feature_used=='second_to_last':
        layer_ind=1
    elif feature_used=='last':
        layer_ind=2

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

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports

        # x_1 = torch.rot90(x_adv, 1, [2,3]) # TODO: do arbitrary angle
        # x_2 = torch.rot90(x_adv, 2, [2,3])
        # x_3 = torch.rot90(x_adv, 3, [2,3])

        angle_r = 15

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
        f0 = net_list[0](x_adv)[layer_ind]
        f1 = net_list[1](x_1)[layer_ind]
        f2 = net_list[2](x_2)[layer_ind]
        f3 = net_list[3](x_3)[layer_ind]

        f4 = net_list[4](x_4)[layer_ind]
        f5 = net_list[5](x_5)[layer_ind]
        f6 = net_list[6](x_6)[layer_ind]

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

        # f0 = torch.flatten(f0, start_dim=1)
        # f1 = torch.flatten(f1, start_dim=1)
        # f2 = torch.flatten(f2, start_dim=1)
        # f3 = torch.flatten(f3, start_dim=1)

        # f4 = torch.flatten(f4, start_dim=1)

        # f0 = norm2(f0)
        # f1 = norm2(f1)
        # f2 = norm2(f2)
        # f3 = norm2(f3)
        # f4 = norm2(f4)

        # cost = -torch.sum(f0*f1, dim=[0, 1])-torch.sum(f0*f2, dim=[0, 1])-torch.sum(f0*f3, dim=[0, 1])-torch.sum(f0*f4, dim=[0, 1])
        # cost = -torch.sum(f0*f4, dim=[0, 1])

        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss
        
        # def paired_loss(in_list):
        #     loss=0
        #     for i, a in enumerate(in_list):
        #         if i>0:
        #             loss = loss -torch.sum(in_list[0]*a, dim=[0, 1])  # TODO: use contrastive loss in channel
        #     return loss

        cost = paired_loss(f_list)
        # print('cost', i, cost)

        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        # TODO: add noinse to grad like Lagine dynamic
        gradient = x_adv.grad
        # print('grad norm', torch.mean(gradient**2, dim=[0,1,2,3])**0.5, step_size)
        gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)*2) * (steps-1-i)/steps
        # gradient = gradient + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)*0.3) * (steps-1-i)/steps
        # x_adv.grad.sign_()
        gradient.sign_()
        lr_step = 0.5 + (steps-1-i)/steps * 1.5
        x_adv = x_adv - step_size_tensor * gradient * lr_step# x_adv.grad  # use minus to gradient descent
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        # TODO: add noinse to grad like Lagine dynamic
        # gradient = x_adv.grad + 2 * lambda_reg * (x_adv - x) # Adding Gaussian Prior Regularization Term.

        # step_size = 0.01 * (1 + i) ** (-1)
        # # / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10)
        # gradient = gradient  + \
        #     torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient))

        # # gradient.sign_()
        # print('grad norm', torch.mean(gradient**2, dim=[0,1,2,3])**0.5, step_size)
        # x_adv = x_adv - step_size  * gradient # * step_size_tensor # x_adv.grad  # use minus to gradient descent
        # x_adv = clamp_tensor(x_adv, upper_bound, lower_bound) # remove due to add L2 prior

        # x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        # #TODO: volatile option for backward, check later
         
    return x_adv



# def Rotation_Invariance_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True):
#     '''
#     Generates recalibration with invariance

#     :param x: 
#     :param y: 
#     :param net: 
#     :param Loss: 
#     :param epsilon: 
#     :param steps: 
#     :param dataset: 
#     :param step_size: 
#     :param info: 
#     :param using_noise: 
#     :return: 
#     '''

#     # print('crop ', torch.max(x), torch.min(x))
#     # print('size', x.size())

#     std_array = np.asarray(info["std"])
#     tensor_std = torch.from_numpy(std_array)
#     tensor_std = tensor_std.unsqueeze(0)
#     tensor_std = tensor_std.unsqueeze(2)
#     tensor_std = tensor_std.unsqueeze(2).float()
#     tensor_std = tensor_std.cuda()

#     GPU_flag = False
#     if torch.cuda.is_available():
#         GPU_flag=True

#     x_adv = x.clone()

#     pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
#     pert_upper = x_adv + pert_epsilon
#     pert_lower = x_adv - pert_epsilon


#     upper_bound = torch.ones_like(x_adv)
#     lower_bound = torch.zeros_like(x_adv)

#     upper_bound = forward_transform(upper_bound, info)
#     lower_bound = forward_transform(lower_bound, info)

#     upper_bound = torch.min(upper_bound, pert_upper)
#     lower_bound = torch.max(lower_bound, pert_lower)

#     #TODO: print and check the bound


#     ones_x = torch.ones_like(x).float()
#     if GPU_flag:
#         Loss = Loss.cuda()
#         x_adv = x_adv.cuda()
#         upper_bound = upper_bound.cuda()
#         lower_bound = lower_bound.cuda()
#         tensor_std = tensor_std.cuda()
#         ones_x = ones_x.cuda()
#         y = y.cuda()

#     step_size_tensor = ones_x * step_size / tensor_std

#     if using_noise:
#         noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
#         if GPU_flag:
#             noise = noise.cuda()
#         noise = noise / tensor_std
#         x_adv = x_adv + noise
#         x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

#     x_adv = Variable(x_adv, requires_grad=True)

#     for i in range(steps):
#         x_1 = torch.rot90(x_adv, 1, [2,3]) # TODO: do arbitrary angle
#         x_2 = torch.rot90(x_adv, 2, [2,3])
#         x_3 = torch.rot90(x_adv, 3, [2,3])

#         s = random.uniform(0.5, 2)
#         x_4 = torch.nn.functional.upsample(x_adv, scale_factor=s, mode='bilinear')

#         # print('x4', x_4.size())
#         f0 = net_list[0](x_adv)[1]
#         # f1 = net_list[1](x_1)[1]
#         # f2 = net_list[2](x_2)[1]
#         # f3 = net_list[3](x_3)[1]

#         f4 = net_list[4](x_4)[1]

#         # f1 = torch.rot90(f1, 3, [2,3])
#         # f2 = torch.rot90(f2, 2, [2,3])
#         # f3 = torch.rot90(f3, 1, [2,3])

#         # f4 = torch.nn.functional.upsample(f4, size=(f0.size(2), f0.size(3)), mode='bilinear')
#         f4 = torch.nn.functional.adaptive_avg_pool2d(f4, output_size=(1,1))
#         f0 = torch.nn.functional.adaptive_avg_pool2d(f0, output_size=(1,1))

#         f0 = torch.flatten(f0, start_dim=1)
#         # f1 = torch.flatten(f1, start_dim=1)
#         # f2 = torch.flatten(f2, start_dim=1)
#         # f3 = torch.flatten(f3, start_dim=1)

#         f4 = torch.flatten(f4, start_dim=1)

#         f0 = norm2(f0)
#         # f1 = norm2(f1)
#         # f2 = norm2(f2)
#         # f3 = norm2(f3)
#         f4 = norm2(f4)

#         # cost = -torch.sum(f0*f1, dim=[0, 1])-torch.sum(f0*f2, dim=[0, 1])-torch.sum(f0*f3, dim=[0, 1])-torch.sum(f0*f4, dim=[0, 1])
#         cost = -torch.sum(f0*f4, dim=[0, 1])

#         for net in net_list:
#             net.zero_grad()

#         if x_adv.grad is not None:
#             x_adv.grad.data.fill_(0)
#         cost.backward()

#         x_adv.grad.sign_()
#         x_adv = x_adv - step_size_tensor * x_adv.grad  # use minus to gradient descent
#         #print(x_adv.data[:,4,4])
#         x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
#         # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
#         # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
#         x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
#         #TODO: volatile option for backward, check later

#     return x_adv



def Rotation_Invariance_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True):
    '''
    Generates recalibration with invariance

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

    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    tensor_std = tensor_std.cuda()

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        angle_r=15
        layer_ind=1
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
        f0 = net_list[0](x_adv)[layer_ind]
        f1 = net_list[1](x_1)[layer_ind]
        f2 = net_list[2](x_2)[layer_ind]
        f3 = net_list[3](x_3)[layer_ind]

        f4 = net_list[4](x_4)[layer_ind]
        f5 = net_list[5](x_5)[layer_ind]
        f6 = net_list[6](x_6)[layer_ind]

        f_list = [f0, f1, f2, f3, f4, f5, f6]
        f_list = [torch.nn.functional.adaptive_avg_pool2d(e, output_size=(1,1)) for e in f_list]
        f_list = [norm2(torch.flatten(e, start_dim=1)) for e in f_list]


        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b, dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss
        
        # def paired_loss(in_list):
        #     loss=0
        #     for i, a in enumerate(in_list):
        #         if i>0:
        #             loss = loss -torch.sum(in_list[0]*a, dim=[0, 1])  # TODO: use contrastive loss in channel
        #     return loss

        cost = paired_loss(f_list)

        for net in net_list:
            net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - step_size_tensor * x_adv.grad  # use minus to gradient descent
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv
