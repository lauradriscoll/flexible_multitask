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

all_rules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              # 'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']


###### EDIT THESE PARAMETERS ########
rule_trains = all_rules#['fdgo', 'fdanti','delaygo','delayanti']
l2w = -6
l2h = -7
n_rnn = 512 

#############

s = '_'
rule_trains_str = s.join(rule_trains)

for x in range(5):
	folder = str(x)
	filedir = os.path.join('data','crystals','softplus',str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',
		'l2_w'+str(-l2w)+'_h'+str(-l2h)+rule_trains_str,folder)
	train.train(filedir, seed=x,  max_steps=1e8, ruleset = 'all',rule_trains = rule_trains,
		hp = { 'activation' : 'softplus',
				'n_rnn': n_rnn,
				'l1_h': 0,
				'l2_h': 10**l2h,
	            'l1_weight': 0,
	            'l2_weight': 10**l2w,
	            'l2_weight_init': 0,
	            'n_eachring' : 2,
	            'n_output' : 1+2,
	            'n_input' : 1+2*2+20,
			    'delay_fac' : 1,
            	'sigma_rec': 0.05,
	            'sigma_x': 0.1,
			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
		display_step=1000, rich_output=False)

