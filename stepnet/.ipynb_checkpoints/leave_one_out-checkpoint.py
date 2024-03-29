from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import task
from task import generate_trials
from network import Model, get_perf
import tools
import train

# rule_trains = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
#               # 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
#               'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
#               'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

# parse input arguments as:
rnn_type = str(sys.argv[1])
activation = str(sys.argv[2])
init = str(sys.argv[3])
n_rnn = int(sys.argv[4])
l2w = float(sys.argv[5])
l2h = float(sys.argv[6])
l1w = float(sys.argv[7])
l1h = float(sys.argv[8])
seed = int(sys.argv[9])
lr = float(sys.argv[10])
sigma_rec = float(sys.argv[11])/20
sigma_x = float(sys.argv[12])/20
pop_rule = int(sys.argv[13])

rule_trains_label = 'leave_one_out'

rule_trains_all = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

rule_trains_pre = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
rule_trains_pre.pop(pop_rule)

post_train = rule_trains_all[pop_rule]

s = '_'
rule_trains_str = s.join(rule_trains_pre)


    folder = str(seed)

    start_from = str(len(rule_trains_pre))+'_tasks'
    # models_dir = '/Users/lauradriscoll/Documents/data/rnn/multitask/stepnet/lr/' #compy
    models_dir = '/home/users/lndrisco/code/multitask-nets/stepnet/data/' #sherlock
    transfer_model_dir = os.path.join(models_dir,rnn_type,activation,init,str(len(rule_trains_pre))+'_tasks',str(n_rnn)+'_n_rnn','lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_'+rule_trains_str,folder)

    filedir = os.path.join('data',rnn_type,activation,init,rule_trains_label,str(len(rule_trains_pre))+'_tasks',str(n_rnn)+'_n_rnn',
        'lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_'+rule_trains_str,'post_train_'+post_train,folder)

    train.train(filedir, seed=seed,  max_steps=1e8, ruleset = 'all',rule_trains = [post_train],
    hp = { 'activation' : activation,
            'w_rec_init': init,
            'n_rnn': n_rnn,
            'l1_h': np.min([-l1h, 10**l1h]),
            'l2_h': np.min([-l2h, 10**l2h]),
            'l1_weight': np.min([-l1w, 10**l1w]),
            'l2_weight': np.min([-l2w, 10**l2w]),
            'l2_weight_init': 0,
            'n_eachring' : 2,
            'n_output' : 1+2,
            'n_input' : 1+2*2+20,
            'sigma_rec': sigma_rec,
            'sigma_x': sigma_x,
            'rnn_type': rnn_type,
            'use_separate_input': False,
	    'learning_rate': 10**(lr/2)},
    display_step=10000, rich_output=False)
