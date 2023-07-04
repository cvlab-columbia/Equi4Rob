import torch
from torch import nn
import models.drn as drn
from models.DRNSeg import DRNSeg, DRNSegDepth



import argparse
import models.drn as drn
from models.DRNSeg import DRNSeg
# from models.FCN32s import FCN32s
import data_transforms as transforms
import json
import math
import os
from os.path import exists, join, split
import threading

import time, datetime

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import logging

from dataloaders.dataloader import get_info, get_loader
from learning.utils_learn import *

from learning.attack import PGD_attack, PGD_attack_adaptive_equi, PGD_attack_adaptive_inv, Rotation_Equi_Defense, Rotation_Invariance_Defense, SGLD_Equi_Defense
from learning.attack_new_single import PGD_attack_new, MIM_attack_new, Equi_Set_Defense,Equi_Set_Defense_good_inloop, PGD_attack_new_adaptive, PGD_attack_new_adaptive_inv, Inv_Set_Defense_good_inloop, Inv_Set_Defense  #, Inv_Set_Defense
from learning.BPDA import BPDA
from learning.houdini_attack import Houdini_attack_new
from dataloaders.utils import decode_segmap

import data_transforms as transforms

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def eval_adv(eval_data_loader, model_lists, num_classes, args=None, info=None, eval_score=None, calculate_specified_only=False,test_flag=False):

    print("___Entering Adversarial Validation validate_adv()___")

    if args.rotate_reverse:
        from multigpu_new_rot import RotReversal
        # rot_reverse_attack = RotReversal()
        rot_reverse_attack = RotReversal('data/ckpts/advpretrained_ssl_rot_19.pth')
        
        

    score = AverageMeter()
    CELoss = AverageMeter()

    # model.eval()
    hist = np.zeros((num_classes, num_classes))

    criterion = nn.NLLLoss(ignore_index=255)

    from torchvision.transforms import transforms
    s=1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transforms_customized = torch.nn.Sequential(
        transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        transforms.RandomGrayscale(p=0.2),
    )
    scripted_transforms = torch.jit.script(transforms_customized).cuda()

    mu = torch.tensor(args.datainfo['mean']).view(3,1,1).cuda()
    std = torch.tensor(args.datainfo['std']).view(3,1,1).cuda()  
    def normalize(X):
        return (X - mu)/std


    all_clean_equi=0
    all_adv_equi = 0
    all_reverse_equi = 0
    cnt = 0
    from learning.measure_equi import Measure_Equi_Set_Defense_good_inloop

    for iter, (image, label, name) in enumerate(eval_data_loader):

        #TODO: Categorise and define variables properly
        if args.adaptive_attack:
            if args.equi:
                adv_delta = PGD_attack_new_adaptive(image, label, model_lists, criterion, args.epsilon, args.steps, args.dataset,
                                    args.step_size, info, using_noise=True, attack_type=args.attack_type, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, ada_lambda=args.adapt_lambda)
            else:
                adv_delta = PGD_attack_new_adaptive_inv(image, label, model_lists, criterion, args.epsilon, args.steps, args.dataset,
                                    args.step_size, info, using_noise=True, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, ada_lambda=args.adapt_lambda)
        elif args.BPDA:
            # if args.equi:
            adv_delta = BPDA(image, label, model_lists, criterion, args.epsilon, args.steps,
                                args.step_size, innormalize=normalize,
                                norm=args.attack_norm, scripted_transforms=scripted_transforms, args=args)

        else:
            adv_delta = PGD_attack_new(image, label, model_lists[0], criterion, args.epsilon, args.steps, args.dataset,
                                args.step_size, info, using_noise=True, innormalize=normalize, norm=args.attack_norm)

            # # if you use MIM, use the following
            # adv_delta = MIM_attack_new(image, label, model_lists[0], criterion, args.epsilon, args.steps, args.dataset,
            #                     args.step_size, info, using_noise=True, innormalize=normalize, norm=args.attack_norm)
            # print('mim')
            # If you use Houdini, use the following
            # adv_delta = Houdini_attack_new(image, label, model_lists[0], criterion, args.epsilon, args.steps, args.dataset,
            #                     args.step_size, info, using_noise=True, innormalize=normalize, norm=args.attack_norm)
            # print('houdini')




        # TODO: Move variables to CUDA - see adv_train
        if torch.cuda.is_available(): #only input is necessary to be put on cuda
            input = image.cuda()
            label = label.cuda()
            # clean_input = clean_input.cuda()
            
        if args.vanilla:
            # Just adv attack
            if args.steps==0:
                final = model_lists[0](normalize(input))[0]
            else:
                final = model_lists[0](normalize(input + adv_delta))[0]
            _, pred = torch.max(final, 1)
        else:
            if args.random_reverse:
                reverse_delta = torch.zeros_like(input).cuda()
                reverse_delta.uniform_(-args.epsilon * args.reverse_time_mutiply, args.epsilon * args.reverse_time_mutiply)
            elif args.rotate_reverse:
                # print('in size', input.size())
                reverse_delta = rot_reverse_attack(input + adv_delta, model_lists[0], normalize)  
            elif args.equi:
                reverse_delta = Equi_Set_Defense_good_inloop(input + adv_delta, label, model_lists, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                    args.reverse_step_size, info, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)

            else:

                reverse_delta = Inv_Set_Defense_good_inloop(input + adv_delta, label, model_lists, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                    args.reverse_step_size, info, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)
            
            final = model_lists[0](normalize(input + adv_delta + reverse_delta))[0]
            _, pred = torch.max(final, 1)
                      
        all_clean_equi += Measure_Equi_Set_Defense_good_inloop(input, label, model_lists, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                    args.reverse_step_size, info, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)
        all_adv_equi += Measure_Equi_Set_Defense_good_inloop(input + adv_delta, label, model_lists, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                    args.reverse_step_size, info, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)
        all_reverse_equi += Measure_Equi_Set_Defense_good_inloop(input + adv_delta + reverse_delta, label, model_lists, criterion, args.epsilon * args.reverse_time_mutiply, args.reverse_step_num, args.dataset,
                                    args.reverse_step_size, info, using_noise=True, SGD_wN=args.addnoise, attack_type=args.attack_type, innormalize=normalize, 
                                    norm=args.attack_norm, scripted_transforms=scripted_transforms, transform_delta=args.transform_delta)
        cnt += 1
        # print(all_clean_equi/cnt)
        # print(all_adv_equi/cnt)
        # print(all_reverse_equi/cnt)

        if eval_score is not None:
            score.update(eval_score(final, label), input.size(0))
            CELoss.update(cross_entropy2d(final,label,size_average=False).item())

            label = label.cpu().numpy()
            pred = pred.cpu().numpy() if torch.cuda.is_available() else pred.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

        end = time.time()

        freq_print = 5

        if iter % freq_print == 0:
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))

            logger.info(' * Score {top1.avg:.3f}'.format(top1=score))
            print('reverse attack type', args.attack_type,  '\n')

        if args.debug:
            break

    logger.info(' *****\n***OverAll***\n Score {top1.avg:.3f}'.format(top1=score))

    ious = per_class_iu(hist) * 100
    logger.info(' '.join('{:.03f}'.format(i) for i in ious))

    if test_flag:
        # Note: test_flag is for running experiments


        dict_advacc = {}
        # print("TYPES ",type(round(np.nanmean(ious), 2)),type(CELoss.avg),type(score.avg))
        dict_advacc['segmentsemantic'] = {
            "iou": round(np.nanmean(ious), 2),
            "loss":CELoss.avg.item(),
            "seg_acc": score.avg
        }
        return dict_advacc

    print(' *****\n***OverAll***\n Score {top1.avg:.3f}'.format(top1=score))
    print('mIoU', np.nanmean(ious))
    print('reverse attack type', args.attack_type, 'ada lambda', args.adapt_lambda, '\n\n\n\n')
    return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    model_lists = []
    for i in range(args.num_view):
        # model = DRNSegDepth("drn_d_22", 19, pretrained_model=None,
        #                         pretrained=False, tasks=['segmentsemantic'])
        model = DRNSeg("drn_d_22", 19, pretrained_model=None,
                                  pretrained=False)
        if '.tar' in args.pretrained:
            model = torch.nn.DataParallel(model)
            model_load = torch.load(args.pretrained)
            print('model epoch', model_load['epoch'], 'precision', model_load['best_prec1'])
            model.load_state_dict(model_load['state_dict'])
        else:
            model.load_state_dict(torch.load(args.pretrained))
            model = torch.nn.DataParallel(model)
        model.cuda()
        model.eval()
        model_lists.append(model)

    # x=torch.randn(16, 3, 512, 512)

    # y = model(x)[0]
    test_loader, datainfo = get_loader(args, phase, out_name=True, nonormalize=True) # will do normalize later
    args.datainfo = datainfo
    info = get_info(args.dataset)
    mAP = eval_adv(test_loader, model_lists, args.classes, args=args, info=info, eval_score=accuracy,
                    calculate_specified_only=args.select_class)

    
    

def run_test(dataset, model, model_path, step_size, step_num, select_class, train_category, adv_test, test_batch_size, args, epsilon_attack=4, 
    attack_type='vanilla', attack_norm='l_inf', transform_delta=True, addnoise=True, num_view=7,
    reverse_step_size=255, reverse_step_num=50, reverse_time_mutiply=1.5, adapt_lambda=1, BPDA=False):
    config_file_path = "config/{}_{}_config.json".format(model, dataset)

    with open(config_file_path) as config_file:
        config = json.load(config_file)

        import socket
        if 'cv' in socket.gethostname():
            data_dir = '/proj/vondrick/mcz/MTLR/cityscape/cityscape_dataset_subsampled' # TODO:
            backup_output_dir = '/local/vondrick/mcz/backup'


        list_dir = config['list-dir']
        classes = config['classes']
        crop_size = config['crop-size']
        step = config['step']
        arch = config['arch']
        batch_size = config['batch-size']
        epochs = config['epochs']
        lr = config['lr']
        lr_mode = config['lr-mode']
        momentum = config['momentum']
        weight_decay = config['weight-decay']

        workers = config['workers']
        phase = config['phase']
        random_scale = config['random-scale']
        random_rotate = config['random-rotate']
        downsize_scale = config['downsize_scale']
        base_size = config['base_size']

        args.reg_lambda = config["reg_lambda"]
        args.drop_ratio = config["drop_ratio"]
        args.MC_times = config["MC_times"]

        args.test_batch_size = test_batch_size

        args.pixel_scale = config['pixel_scale']
        args.steps = step_num
        args.epsilon = epsilon_attack * 1.0 / args.pixel_scale
        args.step_size = step_size * 1.0 / args.pixel_scale
        args.print_freq = config['print_freq']

        args.reverse_step_size = reverse_step_size / args.pixel_scale
        args.reverse_step_num = reverse_step_num
        args.reverse_time_mutiply = reverse_time_mutiply
        args.adapt_lambda=adapt_lambda
        args.BPDA = BPDA

        print('attack scale {}  budget epsilon {} steps {} step size {}'.
              format(args.pixel_scale, args.epsilon, args.steps, args.step_size))

        args.arch = model
        args.pretrained = model_path

        args.select_class = select_class

        if select_class:
            args.train_category = train_category
            args.others_id = config['others_id']
            args.weight_mul = 1  #TODO:

            args.calculate_specified_only = True
            assert args.others_id not in args.train_category

        # Setting args from config file
        args.adv_test = adv_test
        args.dataset = dataset

        args.config = config
        args.data_dir = data_dir
        args.list_dir = list_dir
        args.classes = classes
        args.crop_size = crop_size
        args.step = step
        args.arch = arch
        args.batch_size = batch_size
        args.epochs = epochs
        args.lr = lr
        args.lr_mode = lr_mode
        args.momentum = momentum
        args.weight_decay = weight_decay
        args.workers = workers
        args.phase = phase
        args.random_scale = random_scale
        args.random_rotate = random_rotate
        args.downsize_scale = downsize_scale
        args.backup_output_dir = backup_output_dir  # To save the backup files corresponding to a training experiment.
        # print('output args.backup_output_dir', args.backup_output_dir)
        args.base_size = base_size
        assert classes > 0

        args.bn_sync = False

        args.vanilla = False
        args.equi= False
        args.adaptive_attack = False
        args.rotate_reverse = False
        args.random_reverse = False
        if 'vanilla' in attack_type:
            args.vanilla = True
        if 'equi' in attack_type:
            args.equi = True
        elif 'inv' in attack_type:
            args.equi = False
        elif 'rot' in attack_type:
            args.rotate_reverse = True
        elif 'random' in attack_type:
            args.random_reverse = True
        if 'ada' in attack_type:
            args.adaptive_attack=True
        args.feature_used = 'second_to_last' # important, this is best set up.

        args.attack_type = attack_type
        args.num_view = num_view

        # print(' '.join(sys.argv))
        # print(args)

        args.attack_norm = attack_norm
        args.transform_delta = transform_delta
        args.addnoise = addnoise

        if args.bn_sync:
            drn.BatchNorm = batchnormsync.BatchNormSync

    for key, val in vars(args).items():
        print(f'{key} = {val}')

    test_seg(args)
    print('reverse_time_mutiply', args.reverse_time_mutiply)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.debug=False
    # 7 gpus, using bs=20





    cityscape_adv_pretrained_model_path = "advtrain_drn_d_22_cityscapes.pth.tar"
    cityscape_adv_pretrained_model_path = "/proj/vondrick/mcz/2022Spring/EquiRob/train/city_adv_eps4_200e_s2_n3/last_200.pth.tar"
    ## adv, if not use BPDA, then set BPDA = False
    # set batch size =8 when you have 88GB GPU memory avaiable (8 * 2080Ti)
    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
            step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=3, args=args, epsilon_attack=4, attack_type='equi',
            attack_norm='l_inf', addnoise=True, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=2.5, BPDA=True) # 29.71, adv trained BPDA
        
    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
            step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, epsilon_attack=4, attack_type='inv',
            attack_norm='l_inf', addnoise=True, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=2.5, BPDA=True)

    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
            step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, epsilon_attack=4, attack_type='rot',
            attack_norm='l_inf', addnoise=True, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=2.5, BPDA=True)
    
    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
            step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=16, args=args, epsilon_attack=4, attack_type='random',
            attack_norm='l_inf', addnoise=True, num_view=9, reverse_step_size=255, reverse_step_num=0, reverse_time_mutiply=2)

    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
            step_size=1, step_num=0, select_class=False, train_category=[], adv_test=False, test_batch_size=16, args=args, epsilon_attack=4, attack_type='equi',
            attack_norm='l_inf', addnoise=True, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=2) # clean rev 48.74

   

    #BPDA reverse bound ablation
    # run_test(dataset='cityscape', model='drn_d_22',
    #          model_path=cityscape_adv_pretrained_model_path,
    #          step_size=1, step_num=0, select_class=False, train_category=[], adv_test=False, test_batch_size=4,
    #          args=args, epsilon_attack=4, attack_type='equi',
    #          attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20,
    #          reverse_time_mutiply=0.25, BPDA=False)  #
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, epsilon_attack=4, attack_type='equi',
    #         attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=0.25, BPDA=False) #


    # run_test(dataset='cityscape', model='drn_d_22',
    #          model_path=cityscape_adv_pretrained_model_path,
    #          step_size=1, step_num=0, select_class=False, train_category=[], adv_test=False, test_batch_size=4,
    #          args=args, epsilon_attack=4, attack_type='equi',
    #          attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20,
    #          reverse_time_mutiply=1, BPDA=False)  #
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, epsilon_attack=4, attack_type='equi',
    #         attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=1, BPDA=False) #

    # run_test(dataset='cityscape', model='drn_d_22',
    #          model_path=cityscape_adv_pretrained_model_path,
    #          step_size=1, step_num=0, select_class=False, train_category=[], adv_test=False, test_batch_size=4,
    #          args=args, epsilon_attack=4, attack_type='equi',
    #          attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20,
    #          reverse_time_mutiply=1.5, BPDA=False)  #

    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, epsilon_attack=4, attack_type='equi',
    #         attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=1.5, BPDA=False) #

    # run_test(dataset='cityscape', model='drn_d_22',
    #          model_path=cityscape_adv_pretrained_model_path,
    #          step_size=1, step_num=0, select_class=False, train_category=[], adv_test=False, test_batch_size=4,
    #          args=args, epsilon_attack=4, attack_type='equi',
    #          attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20,
    #          reverse_time_mutiply=2, BPDA=False)  #

    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_adv_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, epsilon_attack=4, attack_type='equi',
    #         attack_norm='l_inf', addnoise=False, num_view=9, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=2, BPDA=False) #



    # # # also reproduce 32.9 mIoU
    cityscape_vanilla_pretrained_model_path = "clean_drn_d_22_cityscapes.pth.tar"
    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
             step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=8, args=args, attack_type='equi',
             attack_norm='l_inf', addnoise=True, num_view=7, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=1.5)


    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
            step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=8, args=args, attack_type='inv',
            attack_norm='l_inf', addnoise=True, num_view=7, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=1.5)


    run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
            step_size=1, step_num=0, select_class=False, train_category=[], adv_test=False, test_batch_size=8, args=args, attack_type='rot',
            attack_norm='l_inf', addnoise=True, num_view=7, reverse_step_size=255, reverse_step_num=20, reverse_time_mutiply=1.5)



    # Each transformation
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='equi_jitter',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='equi_resize',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='equi_flip',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='equi_rsmall',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='equi_rot',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='equi_resize_flip_rsmall',
            # attack_norm='l_inf', addnoise=True, num_view=7)

    # # inv loss:
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='inv_jitter',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='inv_resize',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='inv_flip',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='inv_rsmall',
    #         attack_norm='l_inf', addnoise=True, num_view=7)
    # run_test(dataset='cityscape', model='drn_d_22', model_path=cityscape_vanilla_pretrained_model_path,
    #         step_size=1, step_num=50, select_class=False, train_category=[], adv_test=False, test_batch_size=4, args=args, attack_type='inv_rot',
    #         attack_norm='l_inf', addnoise=True, num_view=7)






