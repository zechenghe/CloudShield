import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import time
import math
import os
import numpy as np
import utils
import copy

from scipy import stats


class Detector(nn.Module):

    def __init__(self, input_size = 52, hidden_size = 64, num_layers = 1, th = 0.01):
        super(Detector, self).__init__()

        self.Nfeatures = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.mean = None
        self.std = None
        self.eps = 1e-8

        # Single-layer LSTM, can change to other models
        self.net = torch.nn.LSTM(
            input_size = self.Nfeatures,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers
        )
        self.hidden2pred = nn.Linear(hidden_size, input_size)

        # Parameters for collecting reference RED
        self.reference_sequences = []
        self.RED_collection_len = None
        self.RED_points = None
        self.RED = []

        self.stat_test = stats.ks_2samp
        self.th = th

    def set_mean(self, data_mean):
        assert data_mean.shape[1] == self.Nfeatures, "Feature size should match self.mean size"
        self.mean = data_mean[:]

    def set_std(self, data_std):
        assert data_std.shape[1] == self.Nfeatures, "Feature size should match self.std size"
        self.std = data_std[:]

    def set_RED_config(self, RED_collection_len = 50, RED_points = 40):
        self.RED_collection_len = RED_collection_len
        self.RED_points = RED_points

    def normalize(self, data):

        assert data.shape[1] == self.mean.shape[1], "Feature size should match self.mean size"
        assert data.shape[1] == self.std.shape[1], "Feature size should match self.std size"

        return (data - self.mean) / (self.std + self.eps)

    def denormalize(self, data):

        return data * (self.std + self.eps) + self.mean

    def add_reference_sequence(self, data):
        self.reference_sequences.append(data)

    def _get_reconstruction_error(self, seq, gpu=False):
        """
            Compute reconstruction errors (RE).
        """

        # LSTM input of shape (seq_len, batch, input_size)
        seq = seq.unsqueeze(1)
        init_state = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size)
            )

        if gpu:
            seq = seq.cuda()
            init_state = (init_state[0].cuda(), init_state[1].cuda())

        pred, state = self.forward(seq[:-1, :, :], init_state)
        truth = seq[1:, :, :]

        if gpu:
            pred_array = pred.detach().cpu().numpy()
            truth_array = truth.detach().cpu().numpy()
        else:
            pred_array = pred.detach().numpy()
            truth_array = truth.detach().numpy()

        RE = np.squeeze(
            np.sum(
                # Only consider the positive errors, i.e. truth > pred
                #np.maximum(truth_array - pred_array, 0)**2,
                (truth_array - pred_array)**2,
                axis=-1
                )
            )

        return RE, pred

    def update_ref_RED(self, gpu):
        # Update RED if the model is fine-tuned
        self.RED = []
        for seq in self.reference_sequences:
            self.collect_ref_RED(seq, gpu)


    def collect_ref_RED(self, seq, gpu):

        assert self.RED_collection_len != None, "Set RED_collection_len first"
        assert self.RED_points != None, "Set RED_points first"
        assert len(seq) >= self.RED_collection_len * self.RED_points + 1, "Ref sequence is too short"

        RE, _ = self._get_reconstruction_error(seq, gpu)

        t = 0
        ref_RED = []
        while (t + self.RED_collection_len * self.RED_points < len(RE)) and len(ref_RED) < 5:

            accumulate_idx = np.array(range(t, t + self.RED_collection_len * self.RED_points, self.RED_collection_len))
            accumulate_RED = np.zeros(self.RED_points)

            for l in range(self.RED_collection_len):
                accumulate_RED += RE[accumulate_idx + l]

            ref_RED.append(accumulate_RED)
            t += self.RED_collection_len * self.RED_points


        #self.RED = [np.random.choice(
        #    np.reshape(np.array(ref_RED), (-1)),
        #    size=[self.RED_points]
        #    )]
        self.RED += ref_RED[:]


    def predict(self, seq, gpu, debug=False):

        assert self.RED_collection_len != None, "Set RED_collection_len first"
        assert self.RED_points != None, "Set RED_points first"
        assert len(seq) >= self.RED_collection_len * self.RED_points + 1, "Testing sequence is too short"

        T_pred_start = time.clock()
        RE, _ = self._get_reconstruction_error(seq, gpu)
        T_pred_end = time.clock()
        print("Prediction takes ", (T_pred_end-T_pred_start), "seconds")

        p_values = []
        t = 0

        T_KS_start = time.clock()

        while t + self.RED_collection_len * self.RED_points < len(RE):
            accumulate_idx = np.array(range(t, t + self.RED_collection_len * self.RED_points, self.RED_collection_len))
            accumulate_RED = np.zeros(self.RED_points)

            for l in range(self.RED_collection_len):
                accumulate_RED += RE[accumulate_idx + l]

            max_p = 0.0
            for idx, modality in enumerate(self.RED):
                p = stats.ks_2samp(accumulate_RED, modality)[1]
                if p > max_p:
                    max_p = p
                    max_p_idx = idx

            if t == 0 and debug:
                utils.plot_cdf(
                    {
                        "Reference": self.RED[max_p_idx],
                        "Testing": accumulate_RED
                    },
                    title="p_value {p}".format(
                        p=max_p
                    )
                )

            p_values.append(max_p)
            t += 1

        T_KS_end = time.clock()
        print("Statistical test takes ", (T_KS_end-T_KS_start), "seconds")

        p_values = np.array(p_values)

        labels = p_values.copy()

        # 0 is normal, 1 is abnormal
        labels[p_values >= self.th] = 0
        labels[p_values < self.th] = 1

        return labels, p_values

    def forward(self, seq, state):
        hiddens, state = self.net(seq, state)
        pred = self.hidden2pred(hiddens)
        return pred, state


class WeightClipper(object):
    """
        A weight clipper to constrain the update of weights within a threshold.
        It is equivalent to |w'-w|_p < th.
    """

    def __init__(self, net, constraint=0.1):
        self.net = copy.deepcopy(net)
        self.constraint = constraint

    def __call__(self, updated_model):
        for net_name, para in self.net.named_parameters():
            for update_net_name, update_para in updated_model.named_parameters():
                # TODO: this is not efficient
                if net_name == update_net_name:
                    #print(name, para)
                    diff = update_para.data - para.data
                    diff = diff.clamp(-self.constraint, self.constraint)
                    update_para.data = diff + para.data
