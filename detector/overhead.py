import argparse
import numpy as np
import torch
import os
import utils
import collections
import subprocess

from sklearn.neighbors import KernelDensity
import concurrent

# Load data
id_to_feature = utils.id_to_feature
feature_list = utils.FeatureSelect.feature_list
data = collections.defaultdict(collections.defaultdict)

for bg_program in ['none', 'mysql', 'webserver', 'streamserver', 'mltrain', 'mapreduce']:
    data_dir = '../perf/data/{bg_program}_same_core/10000us/'.format(bg_program=bg_program)
    for f in os.listdir(data_dir):
        if f.endswith('.npy'):
            file_name = f.split('.')[0]
            data[bg_program][file_name] = np.load(os.path.join(data_dir, f))

pred_errors = collections.defaultdict(collections.defaultdict)
model_name = 'merged'

for bg_program in ['none', 'mysql', 'webserver', 'streamserver', 'mltrain', 'mapreduce']:
    data_dir = f'preprocessed/pred_errors/{model_name}/{bg_program}/'
    for f in os.listdir(data_dir):
        if f.endswith('.npy'):
            file_name = f.split('.')[0]
            pred_errors[bg_program][file_name] = np.load(os.path.join(data_dir, f))


# Prepare KDE mdoel
sampling = True
training_data = []
testing_data = collections.defaultdict(list)

for bg_program in pred_errors.keys():
    d = pred_errors[bg_program]['ref_and_val_normal']
    if sampling:
        sampling_idx = np.random.randint(low=0, high=len(d), size=1000)
        d = d[sampling_idx, :]

    training_data.append(d)

training_data = np.concatenate(training_data, axis=0)

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(training_data)
th = 5.0


models = {}
models['merged'] = torch.load("checkpoints/AnomalyDetectorMerged.ckpt")

print("Used features:")
for i in feature_list:
    print(id_to_feature[i])

model_name = 'merged'
model = models[model_name]


# Collect normal data
hpc_cmd = "sudo ../perf/event_open_user 3 3333 100 ../perf/data/overhead/10000us/overhead.csv"

n = 0
while True:
    monitor_process = subprocess.Popen(hpc_cmd.split())
    monitor_status = monitor_process.wait()

    data_in = model.normalize(np.float32(data['none']['test_normal'][:100, feature_list]))
    data_in_tensor = torch.tensor(data_in)

    _, pred = model._get_reconstruction_error(
        data_in_tensor,
        gpu=True,
    )

    pred = pred[:, 0, :].detach().cpu().numpy()
    pred_error = data_in[1:, :]-pred
    kde_result = kde.score_samples(pred_error)

    n += 1
    print(f"Run loop {n}")
