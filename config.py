import time

class config:
    sampling_ratio = 30
    #-------------learning_related--------------------#
    batch_size = 16
    img_size = 70
    lmbda = 0.98
    clip_now = 0.25
    num_episodes = 100000
    test_episodes = 500
    save_episodes = 20000
    resume_model = '' #'model/8_28_21_16000.pth'
    display = 100
    #-------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 0.5
    beta = 0.1
    # c_loss_coeff = 0.5 # 0.005
    # switch = 4
    warm_up_episodes = 1000
    episode_len = 5
    gamma = 0.99
    reward_method = 'square'
    # noise_scale = 0.2 #0.5
    #-------------continuous parameters--------------------#
    actions = {
        # 'z': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        # 'i': 9,
    }

    num_actions = len(actions) + 1

    # parameters_scale = {
    #     'gaussian1': 0.5,
    #     'bilateral1': 0.1,
    #     'gaussian2': 1.5,
    #     'bilateral2':  1.0,
    # }

    #-------------lr_policy--------------------#
    base_lr = 0.0001 # 1e-3
    base_lr_d = 0.0003 # 3e-4


    #-------------folder--------------------#
    dataset = 'MICCAI'
    root = 'MICCAI/data/'
