from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor
import random
from learning.transform_geo import rot_img
import torch.nn.functional as F
from learning.attack_new_single import Equi_Set_Defense_good_inloop, Inv_Set_Defense_good_inloop


def norm2(v):
    v = v / (torch.sum(v**2, dim=1, keepdim=True)**0.5 + 1e-10)
    return v

upper_bound, lower_bound = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


from multigpu_new_rot import RotReversal
rot_reverse_attack = RotReversal('data/ckpts/advpretrained_ssl_rot_19.pth')

def purify(net_list, X, args, normalize, label, criterion, scripted_transforms):
    if args.equi:
        reverse_delta = Equi_Set_Defense_good_inloop(X, label, net_list, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                        args.reverse_step_size, None, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize,
                                        norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)
    elif args.rotate_reverse:

        reverse_delta = rot_reverse_attack(X, net_list[0], normalize)
    else:
        reverse_delta = Inv_Set_Defense_good_inloop(X, label, net_list, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                    args.reverse_step_size, None, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize,
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)

    return X + reverse_delta


def BPDA(x, y, net_list, Loss, epsilon, steps, step_size, innormalize, norm = "l_inf",
         scripted_transforms=lambda x: x, args=None, transform_delta=True, ada_lambda=1, layer_ind=1):
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

    net = net_list[0]
    for i in range(steps):
        delta = Variable(delta.data, requires_grad=True)

        X_pfy = purify(net_list, x_adv + delta, args, innormalize, y, Loss, scripted_transforms).detach()
        X_pfy.requires_grad_()

        h_adv = net(innormalize(X_pfy))
        # import pdb; pdb.set_trace()

        # if depth:
        #     cost = Loss(torch.mean(h_adv[0], dim=1), y)
        # else:
        cost = Loss(h_adv[0], y) 
        net.zero_grad()
        cost.backward()


        # d = torch.clamp(d + step_size * torch.sign(g), min=-epsilon, max=epsilon)
        # print('d', d.max())
        # delta.grad.sign_()
        # delta = delta + step_size * delta.grad

        delta.data = (delta + step_size * X_pfy.grad.detach().sign())
        delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        X_pfy.grad.zero_()
        # print('eps, att', epsilon*255)
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data


def BPDA_targeted(x, y, net_list, Loss, epsilon, steps, step_size, innormalize, norm="l_inf",
         scripted_transforms=lambda x: x, args=None, transform_delta=True, ada_lambda=1, layer_ind=1):
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag = True
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
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound - x_adv, upper_bound - x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    # print('delta shape', delta.size())

    net = net_list[0]
    for i in range(steps):
        delta = Variable(delta.data, requires_grad=True)

        X_pfy = purify(net_list, x_adv + delta, args, innormalize, y, Loss, scripted_transforms).detach()
        X_pfy.requires_grad_()

        h_adv = net(innormalize(X_pfy))

        cost = Loss(h_adv[0], y)
        net.zero_grad()
        cost.backward()


        delta.data = (delta - step_size * X_pfy.grad.detach().sign())
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        X_pfy.grad.zero_()
        # print('eps, att', epsilon*255)

        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data


