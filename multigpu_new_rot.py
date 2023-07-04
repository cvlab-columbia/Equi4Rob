import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.nn.functional as f
from tqdm import tqdm

from dataloaders.dataloader import get_loader
from models.DRNSeg import DRNSeg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

upper_limit, lower_limit = 1, 0

logger = logging.getLogger(__name__)


class RotAug:
    """Rotate in all angles and augment."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]
        from torchvision.transforms import transforms
        self.transforms = torch.nn.Sequential(
            # transforms.RandomResizedCrop(size=480),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
        )

    def __call__(self, x):
        out, labels = [], []
        from random import randrange

        # select one angle a time in non-square images.
        ind = randrange(4)
        # for angle in self.angles:
        angle = self.angles[ind]
        x_angle = tf.rotate(x, angle)
        out += [x_angle] + [self.transforms(x_angle)] * 3
        labels.append(angle)
        labels = labels * 4

        out = [torch.unsqueeze(item, 0) for item in out]
        print(out[0].size(), out[1].size())
        out = torch.vstack(out)
        # labels = [item for angle in labels for item in angle]
        return out, labels


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return tf.rotate(x, angle), angle


class Wrn34RotOutBranch(nn.Module):  # after the avg pooling layer
    def __init__(self, default_in_dim=19):
        super().__init__()
        self.bn3 = nn.BatchNorm1d(256)
        self.linear = nn.Linear(default_in_dim, 256)
        self.linear2 = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x


class RotReversal:
    def __init__(self, ckpt_path='checkpoints/ssl_rot_19.pth'):
        self._ckpt_path = ckpt_path
        self._ssl_model = self._init_ssl_model()
        self._rot_transform = RotAug()
        self._rot_criterion = torch.nn.CrossEntropyLoss().cuda()

    def _init_ssl_model(self):
        state_semantics = StateSemantics(self._ckpt_path)
        rot_head = Wrn34RotOutBranch()
        rot_head = nn.DataParallel(rot_head).cuda()
        rot_head = state_semantics.load_ssl_model_state(rot_head)
        rot_head.eval()
        return rot_head

    def __call__(self, x, net, innormalize=lambda x: x, norm="l_inf"):
        x_transformed, angles = self._rot_transform(x[0])
        return compute_universal_reverse_attack(net, self._ssl_model, self._rot_criterion, x_transformed, angles, norm, innormalize)
    
    def get_loss(self, x, net, innormalize=lambda x: x):
        x_transformed, angles = self._rot_transform(x[0])

        new_x = innormalize(x_transformed)
        loss = -SslTrainer.compute_ssl_rot_loss(new_x, angles, self._rot_criterion, net, self._ssl_model, no_grad=False)[0]
        return loss.item()


def generate_unique_str():
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    return timestamp + unique_str


def setup_logging(log_level, unique_str):
    
    logging.basicConfig(
        format='[%(asctime)s] - %(levelname)s - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.FileHandler('log/{}_output.log'.format(unique_str)),
            logging.StreamHandler()
        ]
    )
    logger.info('Logging is setup')


def compute_universal_reverse_attack(model, model_ssl, criterion, x, angles, norm, innormalize):
    epsilon = (6 / 255.)
    alpha = (255 / 255.)
    attack_iters = 20

    delta = torch.unsqueeze(torch.zeros_like(x[0]).cuda(), 0)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError

    delta.requires_grad = True
    for _ in range(attack_iters):
        delta_all = delta.repeat(x.size(0), 1, 1, 1)
        new_x = x / 255. + delta_all

        new_x = innormalize(255 * new_x)
        loss = -SslTrainer.compute_ssl_rot_loss(new_x, angles, criterion, model, model_ssl, no_grad=False)[0]
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        d = delta
        g = grad
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha)
            d_norm = torch.norm(d)
            d = d / (d_norm + 1e-10)
        delta.data = d.detach()
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--fname', default='train_ssl', type=str)
    parser.add_argument('--save_root_path', default='data/ckpts/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--md_path', default='data/ckpts/cifar10_rst_adv.pt.ckpt', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log', default='INFO', type=str)
    return parser.parse_args()


class StateSemantics:
    def __init__(self, ssl_state_path):
        self._ssl_state = None
        self._load_ssl_state(ssl_state_path)
        self._ssl_state_path = ssl_state_path

    def _load_ssl_state(self, ssl_state_path):
        if os.path.exists(ssl_state_path):
            self._ssl_state = torch.load(ssl_state_path)
        else:
            logger.warning('SSL state path is invalid')

    def load_ssl_model_state(self, model):
        if self._ssl_state:
            state_dict = self._ssl_state['rot_head_state_dict']
            model.load_state_dict(state_dict)
            logger.info(f'SSL rotation head weights loaded from {self._ssl_state_path}')
        return model

    @staticmethod
    def save_ssl_state(rot_head, epoch):
        state_dict = {
            'rot_head_state_dict': rot_head.state_dict()
        }
        torch.save(state_dict, f'data/ckpts/advpretrained_ssl_rot_{epoch}.pth')


class SslTrainer:
    def __init__(self):
        self.rot_transform = RotationTransform()

    def train_one_epoch(self, model, rot_head, train_batches, opt, criterion, normalize):
        train_loss, train_n, matches = 0.0, 0, 0.0
        for img, _, _ in tqdm(train_batches):
            img = img.cuda()
            batch_size = img.size(0)
            img = normalize(img)
            batch_train_loss, batch_matches = self.step(model, rot_head, img, opt, criterion)
            train_loss = batch_train_loss * batch_size
            matches += batch_matches
            train_n += batch_size
        return train_loss, train_n, matches

    def step(self, model, rot_head, x, opt, criterion):
        x_rotated, angles = self.rotate_batch_input(x)
        rot_loss, pred, target = self.compute_ssl_rot_loss(x_rotated, angles, criterion, model, rot_head)
        matches = self.test_rot(pred, target)

        opt.zero_grad()
        rot_loss.backward()
        opt.step()

        return rot_loss.item(), matches

    def rotate_batch_input(self, batch_input):
        x_rotated, angles = list(zip(*[self.rot_transform(sample_x) for sample_x in batch_input]))
        x_rotated = [x.unsqueeze(0) for x in x_rotated]
        x_rotated = torch.cat(x_rotated)
        return x_rotated, angles

    @staticmethod
    def compute_ssl_rot_loss(x, angles, criterion, model, rot_head, no_grad=True):
        if no_grad:
            with torch.no_grad():
                _, out, _ = model(x)
        else:
            _, out, _ = model(x)
        pred = rot_head(out)
        target = torch.tensor([angle / 90 for angle in angles], dtype=torch.int64).cuda()
        return criterion(pred, target), pred, target

    @staticmethod
    def test_rot(pred, target):
        with torch.no_grad():
            matches = (pred.max(1)[1] == target).sum().item()
        return matches

    @staticmethod
    def save_state_dict(rot_head, opt, epoch):
        state_dict = {
            'epoch': epoch,
            'rot_head_state_dict': rot_head.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }
        torch.save(state_dict, f'checkpoints/ssl_rot_{epoch}.pth')


def main():
    args = get_args()

    unique_str = generate_unique_str()
    setup_logging(args.log, unique_str)

    args.fname = os.path.join(args.save_root_path, args.fname, unique_str)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
    
    with open("config/drn_d_22_cityscape_config.json") as config_file:
        import json
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

        args.print_freq = config['print_freq']
        args.pixel_scale = config['pixel_scale']


        # print('attack scale {}  budget epsilon {} steps {} step size {}'.
        #       format(args.pixel_scale, args.epsilon, args.steps, args.step_size))

        args.arch = 'drn_d_22'


        # Setting args from config file
        # args.dataset = dataset

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # args.pretrained = '/proj/vondrick/mcz/2022Spring/EquiRob/Pretrained/Cityscape/drn_d_22_cityscapes.pth'
    args.pretrained = '/proj/vondrick/mcz/2022Spring/EquiRob/train/city_adv_eps4_200e_s2_n3/last_200.pth.tar'
    args.dataset = 'cityscape'
    args.data_dir = '/proj/vondrick/mcz/MTLR/cityscape/cityscape_dataset_subsampled'
    args.random_rotate = 0
    args.random_scale = 0
    args.crop_size = 256
    
    args.test_batch_size = 8
    args.workers = 8
    phase = 'val' if args.eval else 'train'
    dataset, info = get_loader(args, 'train', out_name=True, nonormalize=True)
    mu = torch.tensor(info['mean']).view(3, 1, 1).cuda()
    std = torch.tensor(info['std']).view(3, 1, 1).cuda()

    def normalize(x):
        return (x - mu) / std

    state_semantics = StateSemantics(args.ckpt)

    # model = DeepLabV3Plus()
    model = DRNSeg("drn_d_22", 19, pretrained_model=None,
                                  pretrained=False)

    # model.load_state_dict(torch.load(args.pretrained))
    # model = torch.nn.DataParallel(model)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained)['state_dict'])

    model.cuda()
    model.eval()

    rot_head = Wrn34RotOutBranch(default_in_dim=19)
    rot_head = nn.DataParallel(rot_head).cuda()
    logger.info('SSL rotation head initialized')
    rot_head = state_semantics.load_ssl_model_state(rot_head)

    rot_criterion = torch.nn.CrossEntropyLoss().cuda()

    trainer = SslTrainer()
    learning_rate = 1e-4
    params = [param for _, param in rot_head.named_parameters()]
    opt = torch.optim.Adam(params, lr=learning_rate)
    rot_head.train()
    for epoch in range(20):
        logger.info(f'Epoch number: {epoch}')
        train_loss, train_n, train_matches = trainer.train_one_epoch(model, rot_head, dataset, opt,
                                                                     rot_criterion, normalize)
        logger.info('Train loss:  %.4f, Train accuracy: %.4f' % (train_loss / train_n, train_matches / train_n))
        state_semantics.save_ssl_state(rot_head, epoch)


if __name__ == "__main__":
    main()