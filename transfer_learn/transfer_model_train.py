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
from tools_lnd import name_best_ckpt, get_model_params

# rule_trains = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
#               'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
#               'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

# rule_trains_set = {}
# # rule_trains_set['w_all_but_dmsnogo'] = ['fdgo', 'reactgo', 'delaygo', 'fdanti','delayanti','reactanti','delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm','dmsgo', 'dmcgo', 'dmcnogo']
# rule_trains_set['w_all_but_delayanti'] = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti','delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm','dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
# # rule_trains_set['w_key_motifs'] = ['delaygo', 'fdanti']
# # rule_trains_set['wo_key_motifs'] = ['reactgo', 'dmcgo']
# # rule_trains_set['wo_all_motifs'] = []

rules_dict = \
    {'all' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'], #15
    'mante' : ['contextdm1', 'contextdm2', 'multidm'], #3
    'delay' : ['fdgo', 'delaygo', 'fdanti', 'delayanti', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'], #9
    'memory' : ['delaygo', 'delayanti', 'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'], #7
    'react' : ['reactgo', 'reactanti', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'], #6
    'anti' : ['fdanti', 'reactanti', 'delayanti', 'dmsnogo', 'dmcnogo'], #5
    'match' : ['dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'], #4
    'category' : ['dmcgo', 'dmcnogo'], #2
    'delaypro_anti' : ['fdgo','fdanti'], #2
    'pro_big' : ['fdgo', 'reactgo', 'delaygo',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmcgo'], #10
    'mem_anti_motifs' : ['delaygo','fdanti'],
    'pro_small' : ['fdgo','delaygo'],
    'irrel_anti' : ['reactgo','dmcgo']} #2

rule_trains_set = {}
for key in ['mem_anti_motifs',]:
	rule_trains_set[key] = rules_dict[key]

for rule_trains_label in rule_trains_set:
	rule_trains = rule_trains_set[rule_trains_label]

	s = '_'
	rule_trains_str = s.join(rule_trains)
	post_train = 'delayanti'

	# parse input arguments as:
	rnn_type = 'LeakyRNN'
	activation = 'softplus'
	init = 'diag'
	ruleset = 'all'
	n_rnn = int(256)
	l2w = float(-6)
	l2h = float(-6)
	l1w = float(0)
	l1h = float(0)
	lr = float(-7)
	seed = int(0)

	folder = str(seed)

	start_from = str(len(rule_trains))+'_tasks'
	# models_dir = '/Users/lauradriscoll/Documents/data/rnn/multitask/stepnet/final/' #compy
	# /Users/lauradriscoll/Documents/data/rnn/multitask/stepnet/final/pro_small/LeakyRNN/softplus/diag/2_tasks/256_n_rnn/lr7.0l2_w6.0_h6.0_fdgo_delaygo/0
	models_dir = '/home/users/lndrisco/code/multitask-nets/stepnet/data/' #sherlock
	transfer_model_dir = os.path.join(models_dir,rule_trains_label,rnn_type,activation,init,
		str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn','lr'+str(-lr)+'l2_w'+str(-l2w)+
		'_h'+str(-l2h)+'_'+rule_trains_str,folder)

	print(transfer_model_dir)

	filedir = os.path.join('data',rnn_type,activation,init,rule_trains_label,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',
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
            'out_type': ruleset,
				    'use_separate_input': False,
				    'w_rec_init': transfer_model_dir,
				    'learning_rate': 10**(lr/2)},
				    display_step=1000, rich_output=False)

