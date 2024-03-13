from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import getpass
import numpy as np
ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'
else:
    p = '/home/users/lndrisco/'

net = 'transfer_learn'#'stepnet'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net) 
sys.path.insert(0, PATH_YANGNET)

from tools_lnd import plot_training, make_dendro, plot_stability, get_model_params
from analysis import variance
# from varimax_rotation import rotate_D

n_rnn = 256
l2w = -6
l2h = -6
l1w = 0
l1h = 0
lr = -7
rule_trains = all_rules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']


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
for key in ['all',]:#['pro_big','mem_anti_motifs','pro_small','irrel_anti']:
	rule_trains_set[key] = rules_dict[key]

#rule_trains_set['w_key_motifs'] = ['fdgo', 'delaygo', 'fdanti']
#rule_trains_set['wo_key_motifs'] = ['fdgo', 'reactgo', 'dmcgo']
#rule_trains_set['w_all_motifs'] = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti','delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm','dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
# rule_trains_set['all'] = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti','delayanti','delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm','dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
#rule_trains_set['wo_all_motifs'] = []

for rnn_type in ['LeakyRNN']: #'GRU',
    for activation in ['softplus']: #,'retanh','tanh'
        for init in ['diag',]:# ,'diag'
            for seed in ['0','1','2','3','4']: #['untrained',]:

                for rule_trains_label in ['all']:#rule_trains_set.keys():
                    rule_trains = rule_trains_set[rule_trains_label]

                    s = '_'
                    rule_trains_str = s.join(rule_trains)
        		    post_train = 'delayanti'

                    # models_dir = os.path.join(p,'code/multitask-nets/stepnet/data/') #sherlock
                    models_dir = os.path.join(p,'data','rnn','multitask',net,lr) #compy
        		    m = os.path.join(p,models_dir,'data',rnn_type,activation,init,rule_trains_label,str(len(rule_trains_pre))+'_tasks',str(n_rnn)+'_n_rnn',
                            	'lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_'+rule_trains_str,'post_train_'+post_train,seed)

                    print(m)

                    # plot_training(m)
                    # # variance.compute_variance(m)
                    # variance.compute_variance(m,rules = ['fdgo','delaygo', 'fdanti', 'delayanti'])
                    make_dendro(m,method = 'average')
                    # plot_stability(m)
                    # rotate_D(m)

                    model_params = {}
                    w_in, b_in, w_out, b_out  = get_model_params(m)
                    
                    model_params['w_in'] = w_in
                    model_params['b_in'] = b_in
                    model_params['w_out'] = w_out
                    model_params['b_out'] = b_out

                    np.savez(os.path.join(m,'model_params.npz'),**model_params)
