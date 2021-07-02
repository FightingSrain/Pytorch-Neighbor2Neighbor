
import torch
import cv2
import numpy as np


p1 = np.asarray([[[[1, 0],
                 [0, 0]]]])
p2 = np.asarray([[[[0, 1],
                 [0, 0]]]])
p3 = np.asarray([[[[0, 0],
                 [0, 1]]]])
p4 = np.asarray([[[[0, 0],
                 [1, 0]]]])
pool = [p4, p1, p2, p3, p4, p1]

# 低效实现
def gendata1(img):
    b, c, w, h = img.shape
    tmp = img
    res1 = np.zeros((b, c, w // 2, h // 2))
    res2 = np.zeros((b, c, w // 2, h // 2))
    for bs in range(b):
        for i in range(w // 2):
            for j in range(h // 2):
                a1 = np.random.randint(1, 5)
                patch2 = tmp[bs, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2]
                rs1 = pool[a1] * patch2 # (1, 3, 2, 2) 得到随机patch
                a2 = np.random.randint(0, 2)
                if a2 == 0:
                    rs2 = pool[a1-1] * patch2
                else:
                    rs2 = pool[a1+1] * patch2
                for o in range(3):
                    res1[bs, o, i, j] = rs1[:, o, :, :].max()
                    res2[bs, o, i, j] = rs2[:, o, :, :].max()

    return torch.Tensor(res1), torch.Tensor(res2)

# 高效实现
