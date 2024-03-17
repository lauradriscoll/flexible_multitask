from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import task
from task import generate_trials
from network import Model, get_perf
import tools
import train


rule_trains_set = {}

rule_trains_set['w_all_but_delayanti'] = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti','delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm','dmsgo','dmsnogo','dmcgo', 'dmcnogo']
rule_trains_set['w_all_but_dmsnogo'] = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'delayanti','reactanti','delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm','dmsgo','dmcgo', 'dmcnogo']
rule_trains_set['w_key_motifs'] = ['delaygo', 'fdanti']
rule_trains_set['wo_key_motifs'] = ['reactgo', 'dmcgo']
rule_trains_set['wo_all_motifs'] = []

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
rule_trains_key = str(sys.argv[11])

folder = str(seed)
rule_trains = rule_trains_set[rule_trains_key]
s = '_'
rule_trains_str = s.join(rule_trains)

filedir = os.path.join('data',rnn_type,activation,init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',
    'lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_'+rule_trains_str,folder)

train.train(filedir, seed=seed,  max_steps=1e8, ruleset = 'all',rule_trains = rule_trains,
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
            'sigma_rec': 0.05,
            'sigma_x': 0.1,
            'rnn_type': rnn_type,
            'use_separate_input': False,
	    'learning_rate': 10**(lr/2)},display_step=10000,rich_output=False)
