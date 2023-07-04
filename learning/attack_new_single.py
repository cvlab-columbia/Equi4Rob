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

upper_bound, lower_bound = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def PGD_attack_new(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, innormalize=lambda x: x, norm = "l_inf", depth=False):
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
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    # print('delta shape', delta.size())

    for i in range(steps):
        delta = Variable(delta.data, requires_grad=True)
        h_adv = net(innormalize(x_adv+delta))
        # import pdb; pdb.set_trace()

        # if depth:
        #     cost = Loss(torch.mean(h_adv[0], dim=1), y)
        # else:
        cost = Loss(h_adv[0], y) 
        net.zero_grad()
        cost.backward()

        if norm == "l_inf":
            # d = torch.clamp(d + step_size * torch.sign(g), min=-epsilon, max=epsilon)
            # print('d', d.max())
            delta.grad.sign_()
            delta = delta + step_size * delta.grad
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
            # print('eps, att', epsilon*255)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data


def MIM_attack_new(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, innormalize=lambda x: x, norm = "l_inf",  momentum=0.5):
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
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    # print('delta shape', delta.size())

    for i in range(steps):
        delta = Variable(delta.data, requires_grad=True)
        h_adv = net(innormalize(x_adv+delta))

        cost = Loss(h_adv[0], y) #TODO: works, but is this the correct place to convert to long??
        net.zero_grad()
        cost.backward()

        if norm == "l_inf":
            # d = torch.clamp(d + step_size * torch.sign(g), min=-epsilon, max=epsilon)
            # print('d', d.max())
            if i==0:
                g = delta.grad.detach() / torch.sum(torch.abs(delta.grad.detach()))
            else:
                g = momentum * g + delta.grad.detach() / torch.sum(torch.abs(delta.grad.detach()))
            delta = delta + step_size * torch.sign(g)
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
            # print('eps, att', epsilon*255)
        elif norm == "l_2":
            grad = delta.grad.detach()
            d = delta
            g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = grad/(g_norm + 1e-10)
            if i==0:
                g = scaled_g
            else:
                g = momentum * g + scaled_g
            g_norm2 = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g_2 = g / (g_norm2 + 1e-10)
            d = (d + scaled_g_2*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data



def PGD_attack_new_adaptive(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, innormalize=lambda x: x, norm = "l_inf",
                            attack_type='', scripted_transforms=lambda x: x, transform_delta=True, ada_lambda=1, layer_ind=1):
    '''
    Generates attacked image that is adaptive to the defense approach.

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
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    # print('delta shape', delta.size())

    for i in range(steps):
        angle_r = 15
        delta = Variable(delta.data, requires_grad=True)
        h_adv = net_list[0](innormalize(x_adv+delta))

        cost = Loss(h_adv[0], y)

 
        f0 = net_list[0](innormalize(x_adv))[layer_ind]
        f_list = [feature_post_norm(f0)] 
        mask_list = []
        
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

            if cnt == 0:
                mask_list.append(torch.flatten(torch.ones_like(f0), start_dim=1))

            if cnt == 4 or cnt == 5:
                f1 = rot_img(f1, -theta_5, f1.dtype)
                x_mask = torch.ones_like(f0)
                x_mask = rot_img(rot_img(x_mask, theta_5, f0.dtype), -theta_5, f0.dtype)
            else:
                f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear')
                x_mask = torch.ones_like(f0)
                if cnt == 7:
                    f1 = torch.flip(f1, dims=[3])

            x_mask = torch.flatten(x_mask, start_dim=1)>0.5
            feature = feature_post_norm(f1)
            f_list.append(feature)
            mask_list.append(x_mask)

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost_ada = paired_loss(f_list)

        cost = cost - ada_lambda* cost_ada

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)

        net_list[0].zero_grad()
        cost.backward()

        if norm == "l_inf":
            # d = torch.clamp(d + step_size * torch.sign(g), min=-epsilon, max=epsilon)
            # print('d', d.max())
            delta.grad.sign_()
            delta = delta + step_size * delta.grad
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
            # print('eps, att', epsilon*255)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data
    

def feature_post_norm(f):
    feature = norm2(torch.flatten(f, start_dim=1))
    return feature

def Equi_Set_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
    Generates recalibration using equivariance
    Applying all specified transformation to each view

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
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports
        delta = Variable(delta.data, requires_grad=True)

        f0 = net_list[0](innormalize(x_adv+delta))[1]
        f_list = [feature_post_norm(f0)] 
        mask_list = []
        
        for cnt, each in enumerate(net_list[1:]):
            doflip = random.uniform(0,1)
            r = random.uniform(-1,3)
            angle_r = 15
            theta_r = random.uniform(-angle_r, angle_r)/180*3.1415926

            # DO T transformation
            # DO the flip
            # if transform_delta:
            x_s1 = x_adv + delta
            # else:
            #     x_s1 = x_adv
            if  'flip' in attack_type:
                if doflip>0.5:
                    x_s1 = torch.flip(x_s1, dims=[3]) # TODO: need check dim

            if 'jitter' in attack_type:
                x_s1 = scripted_transforms(x_s1)

            if 'resize' in attack_type:
                w_s = random.uniform(0,3)
                if w_s<1:
                    w_s=0.3
                elif w_s<2:
                    w_s=0.5
                else:
                    w_s=2
                ratio_s = random.uniform(0.8, 1.2)
                x_s1 = torch.nn.functional.upsample(x_s1, size=(int(x_s1.size(2)*w_s), int(x_s1.size(3)*w_s*ratio_s)), mode='bilinear')

            
            # DO the 90 rot, this does not have pixel lost
            if 'rot' in attack_type:
                if r<=0:
                    pass
                elif r<=1:
                    x_s1 = torch.rot90(x_s1, 1, [2,3])
                elif r>=2:
                    x_s1 = torch.rot90(x_s1, 3, [2,3])
                else:
                    x_s1 = torch.rot90(x_s1, 2, [2,3])

            if 'rsmall' in attack_type:
                x_s1 = rot_img(x_s1, theta_r, x_s1.dtype)

            # forward image through network
            # if transform_delta:
            f1 = net_list[1](innormalize(x_s1))[1] 
            # DO T^-1 transformation
            if 'rsmall' in attack_type:
                f1 = rot_img(f1, -theta_r, f1.dtype)
                x_mask = torch.ones_like(f0)
                x_mask = rot_img(rot_img(x_mask, theta_r, f0.dtype), -theta_r, f0.dtype)
            else:
                x_mask = torch.ones_like(f0)
            if cnt == 0:
                mask_list.append(torch.flatten(torch.ones_like(f0), start_dim=1))

            if 'rot' in attack_type:
                if r<=0:
                    pass
                elif r<=1:
                    f1 = torch.rot90(f1, 3, [2,3])
                elif r>=2:
                    f1 = torch.rot90(f1, 1, [2,3])
                else:
                    f1 = torch.rot90(f1, 2, [2,3])
            
            if 'resize' in attack_type:
                f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear')

            if  'flip' in attack_type:
                if doflip>0.5:
                    f1 = torch.flip(f1, dims=[3])

            x_mask = torch.flatten(x_mask, start_dim=1)>0.5
            feature = feature_post_norm(f1)
            f_list.append(feature)
            mask_list.append(x_mask)

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)
        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)
    print('done new')
    return delta.data


def Inv_Set_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
    Generates recalibration using equivariance
    Applying all specified transformation to each view

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
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports
        delta = Variable(delta.data, requires_grad=True)

        f0 = net_list[0](innormalize(x_adv+delta))[1]

        f0 = torch.nn.functional.adaptive_avg_pool2d(f0, output_size=(1,1))
        f0 = norm2(torch.flatten(f0, start_dim=1))

        f_list = [f0] 
        
        for cnt, each in enumerate(net_list[1:]):
            doflip = random.uniform(0,1)
            r = random.uniform(-1,3)
            angle_r = 15
            theta_r = random.uniform(-angle_r, angle_r)/180*3.1415926

            # DO T transformation
            # DO the flip
            # if transform_delta:
            x_s1 = x_adv + delta
            # else:
            #     x_s1 = x_adv
            if  'flip' in attack_type:
                if doflip>0.5:
                    x_s1 = torch.flip(x_s1, dims=[3]) # TODO: need check dim

            if 'jitter' in attack_type:
                x_s1 = scripted_transforms(x_s1)

            if 'resize' in attack_type:
                w_s = random.uniform(0,3)
                if w_s<1:
                    w_s=0.3
                elif w_s<2:
                    w_s=0.5
                else:
                    w_s=2
                ratio_s = random.uniform(0.8, 1.2)
                x_s1 = torch.nn.functional.upsample(x_s1, size=(int(x_s1.size(2)*w_s), int(x_s1.size(3)*w_s*ratio_s)), mode='bilinear')

            
            # DO the 90 rot, this does not have pixel lost
            if 'rot' in attack_type:
                if r<=0:
                    pass
                elif r<=1:
                    x_s1 = torch.rot90(x_s1, 1, [2,3])
                elif r>=2:
                    x_s1 = torch.rot90(x_s1, 3, [2,3])
                else:
                    x_s1 = torch.rot90(x_s1, 2, [2,3])

            if 'rsmall' in attack_type:
                x_s1 = rot_img(x_s1, theta_r, x_s1.dtype)

            # forward image through network
            # if transform_delta:
            f1 = net_list[1](innormalize(x_s1))[1] # 
            f1 = torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1))
            feature = norm2(torch.flatten(f1, start_dim=1))

            f_list.append(feature)

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b, dim=[0, 1])  
            return loss

        cost = paired_loss(f_list)

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)
        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)
    print('done new')
    return delta.data


def Equi_Single_Set_Defense(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='', attack_type_list=[],
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''
    The attack_type_list contains a list of defense, each image only is applied one specified defense, the 
    Generates recalibration using equivariance
    Applying only one transformation to each view.

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
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    for i in range(steps):
        # For rotation: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports
        delta = Variable(delta.data, requires_grad=True)

        f0 = net_list[0](innormalize(x_adv+delta))[1]
        f_list = [feature_post_norm(f0)] 
        mask_list = []
        
        for cnt, each in enumerate(net_list[1:]):
            specified_defense = attack_type_list[cnt]

            doflip = random.uniform(0,1)
            r = random.uniform(-1,3)
            angle_r = 15
            theta_r = random.uniform(-angle_r, angle_r)/180*3.1415926

            # DO T transformation
            # DO the flip
            # if transform_delta:
            x_s1 = x_adv + delta
            # else:
            #     x_s1 = x_adv
            if  'flip' in specified_defense:
                if doflip>0.5:
                    x_s1 = torch.flip(x_s1, dims=[3]) # TODO: need check dim

            if 'jitter' in specified_defense:
                x_s1 = scripted_transforms(x_s1)

            if 'resize' in specified_defense:
                w_s = random.uniform(0,3)
                if w_s<1:
                    w_s=0.3
                elif w_s<2:
                    w_s=0.5
                else:
                    w_s=2
                ratio_s = random.uniform(0.8, 1.2)
                x_s1 = torch.nn.functional.upsample(x_s1, size=(int(x_s1.size(2)*w_s), int(x_s1.size(3)*w_s*ratio_s)), mode='bilinear')

            
            # DO the 90 rot, this does not have pixel lost
            if 'rot' in specified_defense:
                if r<=0:
                    pass
                elif r<=1:
                    x_s1 = torch.rot90(x_s1, 1, [2,3])
                elif r>=2:
                    x_s1 = torch.rot90(x_s1, 3, [2,3])
                else:
                    x_s1 = torch.rot90(x_s1, 2, [2,3])

            if 'rsmall' in specified_defense:
                x_s1 = rot_img(x_s1, theta_r, x_s1.dtype)

            # forward image through network
            # if transform_delta:
            f1 = net_list[1](innormalize(x_s1))[1] 
            # DO T^-1 transformation
            if 'rsmall' in specified_defense:
                f1 = rot_img(f1, -theta_r, f1.dtype)
                x_mask = torch.ones_like(f0)
                x_mask = rot_img(rot_img(x_mask, theta_r, f0.dtype), -theta_r, f0.dtype)
            else:
                x_mask = torch.ones_like(f0)
            if cnt == 0:
                mask_list.append(torch.flatten(torch.ones_like(f0), start_dim=1))

            if 'rot' in attack_type:
                if r<=0:
                    pass
                elif r<=1:
                    f1 = torch.rot90(f1, 3, [2,3])
                elif r>=2:
                    f1 = torch.rot90(f1, 1, [2,3])
                else:
                    f1 = torch.rot90(f1, 2, [2,3])
            
            if 'resize' in attack_type:
                f1 = torch.nn.functional.upsample(f1, size=(f0.size(2), f0.size(3)), mode='bilinear')

            if  'flip' in attack_type:
                if doflip>0.5:
                    f1 = torch.flip(f1, dims=[3])

            x_mask = torch.flatten(x_mask, start_dim=1)>0.5
            feature = feature_post_norm(f1)
            f_list.append(feature)
            mask_list.append(x_mask)

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)
        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)
    print('done new')
    return delta.data


def Equi_Set_Defense_good(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True):
    '''

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
    x_new = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_new = x_new.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_new).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_new, upper_bound-x_new)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True
    layer_ind=1

    for i in range(steps):
        angle_r = 15
        delta = Variable(delta.data, requires_grad=True)
        x_adv = delta + x_new

        w_s = random.uniform(0.5, 2)
        ratio_s = random.uniform(0.8, 1.2)
        theta_1 = random.uniform(-angle_r, angle_r)/180*3.1415926
        w_s = 2
        theta_1 = 1

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


        w_s = random.uniform(0.5, 2)
        ratio_s = random.uniform(0.8, 1.2)
        w_s = 0.5
        theta_1 = 4

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

        f_list_raw = [f0, f1, f2, f3, f4, f5, f6]

        f_list = [norm2(torch.flatten(e, start_dim=1)) for e in f_list_raw]

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)
        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)
    
    return delta.data





def Equi_Set_Defense_good_inloop(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True, layer_ind=1, keep_ratio=2):
    '''

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

    use_CE = False
    if use_CE:
        def innormalize(x):
            return x

    
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

    delta = torch.zeros_like(x_new).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_new, upper_bound-x_new)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    for i in range(steps):
        angle_r = 15
        delta = Variable(delta.data, requires_grad=True)
        x_adv = delta + x_new


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
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss - torch.sum(a*b*mask_list[i]*mask_list[j], dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

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

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)
        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            # gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) # + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (1)/(i+1)
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)
    
    return delta.data




def Inv_Set_Defense_good_inloop(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, SGD_wN=False, attack_type='',
    innormalize=lambda x: x, norm='l_inf', scripted_transforms=lambda x: x, transform_delta=True, layer_ind=1):
    '''
    

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
    x_new = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_new = x_new.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_new).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_new, upper_bound-x_new)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    for i in range(steps):
        angle_r = 15
        delta = Variable(delta.data, requires_grad=True)
        x_adv = delta + x_new


        f0 = net_list[0](innormalize(x_adv))[1]

        f0 = torch.nn.functional.adaptive_avg_pool2d(f0, output_size=(1,1))
        f0 = norm2(torch.flatten(f0, start_dim=1))

        f_list = [f0] 
        
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

            f1 = torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1))
            feature = norm2(torch.flatten(f1, start_dim=1))

            f_list.append(feature)

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b, dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)

        for net in net_list:
            net.zero_grad()

        if delta.grad is not None:
            delta.grad.data.fill_(0)
        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) # + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)
    
    return delta.data



def PGD_attack_new_adaptive_inv(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True,
        innormalize=lambda x: x, norm = "l_inf", depth=False, ada_lambda=1, layer_ind=1, scripted_transforms=lambda x:x):
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

    delta = torch.zeros_like(x_new).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_new, upper_bound-x_new)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    # print('delta shape', delta.size())

    for i in range(steps):
        delta = Variable(delta.data, requires_grad=True)

        x_adv = delta + x_new
        h_adv = net_list[0](innormalize(x_adv))

        cost_classifier = Loss(h_adv[0], y) 

        angle_r = 15

        f0 = net_list[0](innormalize(x_adv))[1]

        f0 = torch.nn.functional.adaptive_avg_pool2d(f0, output_size=(1,1))
        f0 = norm2(torch.flatten(f0, start_dim=1))

        f_list = [f0] 
        
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

            f1 = torch.nn.functional.adaptive_avg_pool2d(f1, output_size=(1,1))
            feature = norm2(torch.flatten(f1, start_dim=1))

            f_list.append(feature)

        # TODO: add negative.
        def paired_loss(in_list):
            loss=0
            for i, a in enumerate(in_list):
                for j, b in enumerate(in_list):
                    if i<j:
                        loss = loss -torch.sum(a*b, dim=[0, 1])  # TODO: use contrastive loss in channel
            return loss

        cost = paired_loss(f_list)

        cost = cost_classifier - ada_lambda * cost

        for net in net_list:
            net.zero_grad()

        cost.backward()

        if norm == "l_inf":
            gradient = delta.grad
            gradient = gradient / (torch.sum(gradient**2, dim=[0,1,2,3])**0.5 + 1e-10) # + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * (steps-1-i)/steps
            gradient.sign_()

            delta = delta - step_size * gradient
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data