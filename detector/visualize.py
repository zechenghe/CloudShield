import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import warnings
# Ignore warnings due to pytorch save models
# https://github.com/pytorch/pytorch/issues/27972
warnings.filterwarnings("ignore", "Couldn't retrieve source code")

import time
import math
import os
import numpy as np
import pickle

import utils
import SeqGenerator
import detector
import loaddata

import matplotlib.pyplot as plt

args = utils.create_parser()
training_normal_data = loaddata.load_data_all(
    data_dir = args.data_dir,
    file_name = args.normal_data_name_train,
)

utils.plot_seq(
    seqs={
        "train_normal": training_normal_data[:, 0],
    }
)

plt.show(block=True)
