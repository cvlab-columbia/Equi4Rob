
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1).cuda()
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x


# #Test:
# dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# #im should be a 4D tensor of shape B x C x H x W with type dtype, range [0,255]:
# im = torch.ones((1, 3, 25, 38)).cuda()
# im[:, :, 0:1, :] = 0
# im[:, :, 24, :] = 0

# plt.imshow(im.cpu().squeeze(0).permute(1,2,0)) #To plot it im should be 1 x C x H x W
# plt.figure()
# #Rotation by np.pi/2 with autograd support:
# im = rot_img(im.cuda(), 30/180*np.pi, dtype) # Rotate image by 90 degrees.
# rotated_im = rot_img(im.cuda(), -30/180*np.pi, dtype) # Rotate image by 90 degrees.

# print(torch.sum(rotated_im<0.1, dim=[0,1,2,3]))

# plt.imshow(rotated_im.cpu().squeeze(0).permute(1,2,0))
# plt.savefig('tmp2.png')

