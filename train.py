

import numpy as np
import cv2


import torch
import torch.nn as nn
from torch.nn import init
import random
from utils import init_net
from gendatas import gendata1
from Unet.unet2 import Actor
from config import config
from mini_batch_loader import MiniBatchLoader

TRAINING_DATA_PATH1 = "./trainA.txt"
TESTING_DATA_PATH1 = "./trainA.txt"
IMAGE_DIR_PATH = ".//"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# np.random.seed(1234)
# random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
mini_batch_loader1 = MiniBatchLoader(
        TRAINING_DATA_PATH1,
        TESTING_DATA_PATH1,
        IMAGE_DIR_PATH,
        128)
train_data_size1 = MiniBatchLoader.count_paths(TRAINING_DATA_PATH1)
indices1 = np.random.permutation(train_data_size1)
lr = 0.0003
actor = init_net(Actor().to(device), 'kaiming', gpu_ids=[])
actor.load_state_dict(torch.load("./model_test1/modela400_.pth"))
optimizer = torch.optim.Adam(actor.parameters(), lr)
episodes = 0
i_index1 = 0

while episodes < config.num_episodes:
    r1 = indices1[i_index1: i_index1 + config.batch_size]
    raw_x = mini_batch_loader1.load_training_data(r1)  # bgr

    raw_n = np.random.normal(0, 25, raw_x.shape).astype(raw_x.dtype) / 255
    s = np.clip(raw_x + raw_n, a_max=1., a_min=0.)
    y1, y2 = gendata1(s)
    res1 = actor(y1.cuda())
    res1 = torch.clip(res1, max=1., min=0.)

    res2 = actor(torch.FloatTensor(s).cuda()).detach()
    fy1, fy2 = gendata1(res2.detach().cpu().numpy())
    # rt = torch.clip(rt, max=1., min=0.)
    reg = 2 * ((res1 - y2.cuda() - (fy1.cuda() - fy2.cuda()))**2).mean()
    loss = ((res1 - torch.Tensor(y2).cuda())**2).mean() + reg*2
    print("epi: ", episodes, "Loss: ", loss.data)
    if episodes % 2 == 0:
        imaget1 = np.asanyarray(res1.detach().cpu().numpy()[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
        imaget = np.asanyarray(res1.data.cpu().numpy()[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
        imaget = np.squeeze(imaget)
        cv2.imshow("rerrs", imaget)
        cv2.waitKey(1)
    if episodes % 200 == 0:
        torch.save(actor.state_dict(), "./model_test1/modela{}_.pth".format(episodes))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i_index1 + config.batch_size >= train_data_size1:
        i_index1 = 0
        indices1 = np.random.permutation(train_data_size1)
    else:
        i_index1 += config.batch_size
    if i_index1 + 2 * config.batch_size >= train_data_size1:
        i_index1 = train_data_size1 - config.batch_size
    episodes += 1