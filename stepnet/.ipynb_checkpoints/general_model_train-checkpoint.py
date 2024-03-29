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
from task import rules_dict

# parse input arguments as:
rnn_type = 'LeakyRNN' # 'LeakyGRU'
activation = 'softplus' # 'tanh' 'retanh'
init = 'diag' #'randgauss'
n_rnn = 128 #size of the network N
l2w = -6 #l2 weight regularization
l2h = -6 #l2 activity regularization
l1w = 0 #l1 weight regularization
l1h = 0 #l1 activity regularization
seed = 0 #random seed
lr = -6 #learning rate 
sigma_rec = 1/20 #recurrent unit noise
sigma_x = 2/20 #input noise
ruleset = 'all' #specifies set of tasks to train
w_rec_coeff  = 1 # coefficient on weight matrix init

rule_trains = rules_dict[ruleset]

s = '_'
rule_trains_str = s.join(rule_trains)
folder = str(seed)
net_name = 'lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_sig_rec'+str(sigma_rec)+'_sig_x'+str(sigma_x)+'_w_rec_coeff'+str(w_rec_coeff)+'_'+rule_trains_str
filedir = os.path.join('data',ruleset,rnn_type,activation,init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name,folder)
train.train(filedir, seed=seed,  max_steps=1e8, ruleset = ruleset, rule_trains = rule_trains,
    hp = { 'activation' : activation,
            'w_rec_init': init,
            'n_rnn': n_rnn,
            'l1_h': 10**l1h,
            'l2_h': 10**l2h,
            'l1_weight': 10**l1w,
            'l2_weight': 10**l2w
            'l2_weight_init': 0,
            'sigma_rec': sigma_rec,
            'sigma_x': sigma_x,
            'rnn_type': rnn_type,
            'use_separate_input': False,
	        'learning_rate': 10**(lr/2),
            'w_rec_coeff': w_rec_coeff},
            display_step=10000, rich_output=False)
