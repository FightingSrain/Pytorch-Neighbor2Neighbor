import numpy as np
import skimage.measure
import scipy
import cv2
import torch
import math
from torch.nn import init
def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net
def srgb2photopro(img):
    srgb = img
    k = 0.055
    thre_srgb = 0.04045
    a = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    b = np.array(
        [[1.3459433, -0.2556075, -0.0511118], [-0.5445989, 1.5081673, 0.0205351], [0.0000000, 0.0000000, 1.2118128]])
    M = b.dot(a)
    M = M / M.sum(axis=1).reshape((-1, 1))
    thre_photopro = 1 / 512.0

    srgb[srgb <= thre_srgb] /= 12.92
    srgb[srgb > thre_srgb] = ((srgb[srgb > thre_srgb] + k) / (1 + k)) ** 2.4

    image = srgb
    sb = image[:, :, 0:1]
    sg = image[:, :, 1:2]
    sr = image[:, :, 2:3]
    photopror = sr * M[0][0] + sg * M[0][1] + sb * M[0][2]
    photoprog = sr * M[1][0] + sg * M[1][1] + sb * M[1][2]
    photoprob = sr * M[2][0] + sg * M[2][1] + sb * M[2][2]

    photopro = np.concatenate((photoprob, photoprog, photopror), axis=2)
    photopro = np.clip(photopro, 0, 1)
    photopro[photopro >= thre_photopro] = photopro[photopro >= thre_photopro] ** (1 / 1.8)
    photopro[photopro < thre_photopro] *= 16
    return photopro

def photo2srgb(img):
    photopro = img
    thre_photopro = 1 / 512.0 * 16

    a = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    b = np.array(
        [[1.3459433, -0.2556075, -0.0511118], [-0.5445989, 1.5081673, 0.0205351], [0.0000000, 0.0000000, 1.2118128]])
    M = b.dot(a)
    M = M / M.sum(axis=1).reshape((-1, 1))
    M = np.linalg.inv(M)
    k = 0.055
    thre_srgb = 0.04045 / 12.92

    photopro[photopro < thre_photopro] *= 1.0 / 16
    photopro[photopro >= thre_photopro] = photopro[photopro >= thre_photopro] ** (1.8)

    photoprob = photopro[:, :, 0:1]
    photoprog = photopro[:, :, 1:2]
    photopror = photopro[:, :, 2:3]
    sr = photopror * M[0][0] + photoprog * M[0][1] + photoprob * M[0][2]
    sg = photopror * M[1][0] + photoprog * M[1][1] + photoprob * M[1][2]
    sb = photopror * M[2][0] + photoprog * M[2][1] + photoprob * M[2][2]

    srgb = np.concatenate((sb, sg, sr), axis=2)

    srgb = np.clip(srgb, 0, 1)
    srgb[srgb > thre_srgb] = (1 + k) * srgb[srgb > thre_srgb] ** (1 / 2.4) - k
    srgb[srgb <= thre_srgb] *= 12.92

    return srgb

def bgr2hsv(img):
    editted = img

    max_bgr = editted.max(axis=2)
    min_bgr = editted.min(axis=2)

    b_g = editted[:, :, 0] - editted[:, :, 1]
    g_r = editted[:, :, 1] - editted[:, :, 2]
    r_b = editted[:, :, 2] - editted[:, :, 0]

    b_min_flg = (1 - relu(np.sign(b_g))) * relu(np.sign(r_b))
    g_min_flg = (1 - relu(np.sign(g_r))) * relu(np.sign(b_g))
    r_min_flg = (1 - relu(np.sign(r_b))) * relu(np.sign(g_r))

    epsilon = 10 ** (-5)
    h1 = 60 * g_r / (max_bgr - min_bgr + epsilon) + 60
    h2 = 60 * b_g / (max_bgr - min_bgr + epsilon) + 180
    h3 = 60 * r_b / (max_bgr - min_bgr + epsilon) + 300
    h = h1 * b_min_flg + h2 * r_min_flg + h3 * g_min_flg

    v = max_bgr
    s = (max_bgr - min_bgr) / (max_bgr + epsilon)
    return np.concatenate([np.expand_dims(h, 2), np.expand_dims(s, 2), np.expand_dims(v, 2)], 2)

def hsv2bgr(img):
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    h = h * relu(np.sign(h - 0)) * (1 - relu(np.sign(h - 360))) + (h - 360) * relu(np.sign(h - 360)) * (
                1 - relu(np.sign(h - 720))) \
        + (h + 360) * relu(np.sign(h + 360)) * (1 - relu(np.sign(h - 0)))
    h60_flg = relu(np.sign(h - 0)) * (1 - relu(np.sign(h - 60)))
    h120_flg = relu(np.sign(h - 60)) * (1 - relu(np.sign(h - 120)))
    h180_flg = relu(np.sign(h - 120)) * (1 - relu(np.sign(h - 180)))
    h240_flg = relu(np.sign(h - 180)) * (1 - relu(np.sign(h - 240)))
    h300_flg = relu(np.sign(h - 240)) * (1 - relu(np.sign(h - 300)))
    h360_flg = relu(np.sign(h - 300)) * (1 - relu(np.sign(h - 360)))

    C = v * s
    b = v - C + C * (h240_flg + h300_flg) + C * ((h / 60 - 2) * h180_flg + (6 - h / 60) * h360_flg)
    g = v - C + C * (h120_flg + h180_flg) + C * ((h / 60) * h60_flg + (4 - h / 60) * h240_flg)
    r = v - C + C * (h60_flg + h360_flg) + C * ((h / 60 - 4) * h300_flg + (2 - h / 60) * h120_flg)
    return np.concatenate([np.expand_dims(b, axis=2), np.expand_dims(g, axis=2), np.expand_dims(r, axis=2)], axis=2)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_inverse(y):
    epsilon = 10**(-3)
    y_ = y.copy()
    y_ = relu(y_-epsilon)+epsilon
    y_ = 1-epsilon-relu((1-epsilon)-y_)
    return -np.log(1/(y_)-1)
def relu(x):
    x_ = x.copy()
    x_[x_<0] = 0
    return x_
def _normal_logproba(x, mean, logstd, std=None):
    if std is None:
        std = torch.exp(logstd)
    std_sq = std.pow(2)
    logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)

    return logproba
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def gauss_filter(img):
    return cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.5)

def vis_bgrimg(img, time=1):
    image = np.asanyarray(img[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    cv2.imshow("rerrs", image)
    cv2.waitKey(time)
def vis_labimg(img, time=1):
    srs = img[:, 0:3, :, :].copy()
    # srs[:, 1, :, :] = (srs[:, 1, :, :] - 0.5) * 2
    # srs[:, 2, :, :] = (srs[:, 2, :, :] - 0.5) * 2

    srs[:, 0, :, :] *= 100
    srs[:, 1, :, :] *= 127
    srs[:, 2, :, :] *= 127
    image = np.asanyarray(lab2rgb(srs)[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    cv2.imshow("rerr", image)
    cv2.waitKey(time)

def bgr2lab(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0, b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    return np.transpose(dst, (0, 3, 1, 2))
def lab2rgb(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0,2,3,1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0,b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_Lab2RGB)
    return np.transpose(dst, (0,3,1,2))
def fft_shift(x):
    fft = scipy.fftpack.fft2(x)
    fft = scipy.fftpack.fftshift(fft)
    return fft


def shift_ifft(fft):
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    return x


def Downsample(x, mask):
    fft = scipy.fftpack.fft2(x)
    fft_good = scipy.fftpack.fftshift(fft)
    fft_bad = fft_good * mask
    fft = scipy.fftpack.ifftshift(fft_bad)
    x = scipy.fftpack.ifft2(fft)
#    x = np.abs(x)
    x = np.real(x)
    return x, fft_good, fft_bad


def SSIM(x_good, x_bad):
    assert len(x_good.shape) == 2
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def PSNR(x_good, x_bad):
    assert len(x_good.shape) == 2
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


def NMSE(x_good, x_bad):
    assert len(x_good.shape) == 2
    nmse_a_0_1 = np.sum((x_good - x_bad) ** 2)
    nmse_b_0_1 = np.sum(x_good ** 2)
    # this is DAGAN implementation, which is wrong
    nmse_a_0_1, nmse_b_0_1 = np.sqrt(nmse_a_0_1), np.sqrt(nmse_b_0_1)
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
    return nmse_0_1


def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)


def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)


def computeNMSE(o_, p_, i_):
    return NMSE(o_, p_), NMSE(o_, i_)


def DC(x_good, x_rec, mask):
    fft_good = fft_shift(x_good)
    fft_rec = fft_shift(x_rec)
    fft = fft_good * mask + fft_rec * (1 - mask)
    x = shift_ifft(fft)
    x = np.real(x)
    #x = np.abs(x)
    return x


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):
    '''
    source: https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/utils.py
    '''
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

if __name__ == "__main__":
    pass
