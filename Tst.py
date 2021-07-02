

import torch
import numpy as np
import cv2
from Unet.unet2 import Actor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor = Actor().to(device)

actor.load_state_dict(torch.load("./model_test1/modela800_.pth"))

img = cv2.imread("./Kodak24/kodim03.png")
raw_x = np.expand_dims(img[0:256, 0:256,:].transpose(2, 0, 1), 0)/255

raw_n = np.random.normal(0, 25, raw_x.shape).astype(raw_x.dtype) / 255
s = np.clip(raw_x + raw_n, a_max=1., a_min=0.)
actor.eval()
res = actor(torch.Tensor(s).cuda()).detach().cpu().numpy()
res = np.clip(res, a_min=0., a_max=1.)
imaget = np.asanyarray(res[0, :, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
imaget = np.squeeze(imaget)
cv2.imshow("rerrs", imaget)
cv2.waitKey(0)
