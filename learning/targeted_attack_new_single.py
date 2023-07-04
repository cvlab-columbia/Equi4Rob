'''Targeted attack with new way of using original image + delta attack, delta is optimized, image is transformed after image+delta.'''

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

upper_bound, lower_bound = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def targeted_PGD_attack_new(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, innormalize=lambda x: x, norm = "l_inf"):
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
            delta.grad.sign_()
            delta = delta - step_size * delta.grad   # gradient descent to minimize
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
            # print('eps, att', epsilon*255)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)  # gradient descent to minimize
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data



def targted_MIM_attack_new(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, innormalize=lambda x: x, norm = "l_inf",  momentum=0.5):
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
            delta = delta - step_size * torch.sign(g)
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
            d = (d - scaled_g_2*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data


def targeted_PGD_attack_new_adaptive(x, y, net_list, Loss, epsilon, steps, dataset, step_size, info, using_noise=True, innormalize=lambda x: x, norm = "l_inf",
                            attack_type='', scripted_transforms=lambda x: x, transform_delta=True):
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
        delta = Variable(delta.data, requires_grad=True)
        h_adv = net_list[0](innormalize(x_adv+delta))

        cost = Loss(h_adv[0], y)

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
            f1 = net_list[1](innormalize(x_s1))[1] # TODO: two version, either add delta to the image after transformation, or add to the image and also apply transformation to delta.
            # else:
            #     f1 = net_list[1](innormalize(x_s1+delta))[1] # TODO: two version, either add delta to the image after transformation, or add to the image and also apply transformation to delta.

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

        cost_ada = paired_loss(f_list)
        cost = cost + 10* cost_ada

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
            delta = delta - step_size * delta.grad
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
            # print('eps, att', epsilon*255)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d - scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data
    

def feature_post_norm(f):
    feature = norm2(torch.flatten(f, start_dim=1))
    return feature