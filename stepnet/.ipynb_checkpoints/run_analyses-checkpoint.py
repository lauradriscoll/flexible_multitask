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

net = 'stepnet'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net) 
sys.path.insert(0, PATH_YANGNET)

from tools_lnd import plot_training, make_dendro, plot_stability, get_model_params, lesions
from analysis import variance
from varimax_rotation import rotate_D

rnn_type = 'LeakyRNN'
activation = 'softplus'
init = 'diag'
n_rnn = 128
l2w = -6
l2h = -6
l1w = 0
l1h = 0
seed = 0
lr = -8#6
sigma_rec = 1/20
sigma_x = 2/20
pop_rule = 5
ruleset = 'all'
w_rec_coeff  = 1#8/10 #1

rules_dict = \
    {'all' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'], #15
    'untrained' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'], 
    'arm' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
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
    'mem_motifs_small' : ['delaygo','delayanti'],
    'pro_small' : ['fdgo','delaygo'],
    'irrel_anti' : ['reactgo','dmcgo']} #2

rule_trains_set = {}
rule_set_keys = ['all',]
for key in rule_set_keys:
    rule_trains_set[key] = rules_dict[key]

for rnn_type in ['LeakyRNN',]: #'LeakyRNN',
    for activation in ['retanh',]: #,'retanh' 'tanh','retanh',
        for init in ['randgauss','diag']:# 'randgauss'
            for n_rnn in [128,]:#256,512,1024
                for seed in [0,1,2]:
                    for data_folder in ['all',]:#['untrained',]:

                        for rule_trains_label in rule_set_keys:
                            # for pop_rule in range(15):
                            rule_trains = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                            'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                            'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
                            # rule_trains.pop(pop_rule)

                            s = '_'
                            rule_trains_str = s.join(rule_trains)

                            # net_name = 'lr'+"{:.1f}".format(-lr)+'l2_w'+"{:.1f}".format(-l2w)+'_h'+"{:.1f}".format(-l2h)

                            # net_name2 = '_sig_rec'+str(sigma_rec)+'_sig_x'+str(sigma_x)+'_w_rec_coeff'+"{:.1f}".format(w_rec_coeff)+'_'+rule_trains_str

                            # m = os.path.join(p,'sherlock','multitask-nets',net,'data','all',rnn_type,activation,
                            #     init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name+'_'+rule_trains_str,str(seed))
                            

                            if data_folder=='untrained':
                                w_rec_coeff  = 1
                                lr = -8

                            elif (rnn_type == 'LeakyRNN') & (activation[-4:-1] == 'tan'):
                                w_rec_coeff  = 1
                                lr = -8

                            else:
                                w_rec_coeff  = 8/10
                                lr = -6

                            net_name = 'lr'+"{:.1f}".format(-lr)+'l2_w'+"{:.1f}".format(-l2w)+'_h'+"{:.1f}".format(-l2h) 
                            
                            m = os.path.join(p,'sherlock','multitask-nets',net,'data',data_folder,rnn_type,activation,init,
                                    str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name+'_'+rule_trains_str,str(seed))

                            print(m)
                            
                            plot_training(m)
                            variance.compute_variance(m)
                            # make_dendro(m,cel_max_d = 20,method = 'average',criterion = 'maxclust')
                            # make_dendro(m,method = 'single',cel_max_d = 14,criterion = 'maxclust')
                            # _,_ = lesions(m,rules=[],method = 'single',max_d = 14,criterion = 'maxclust')
                            # _,_ = lesions(m,rules=[],method = 'average',max_d = 20,criterion = 'maxclust')
                            make_dendro(m,method = 'ward',criterion = 'maxclust',cel_max_d = 0,max_d = 0)
                            # _,_ = lesions(m,rules=[],method = 'ward',criterion = 'maxclust',max_d = 0)
                            # plot_stability(m)
                            # rotate_D(m)

                            # model_params = {}
                            # w_in, b_in, w_out, b_out  = get_model_params(m)
                            
                            # model_params['w_in'] = w_in
                            # model_params['b_in'] = b_in
                            # model_params['w_out'] = w_out
                            # model_params['b_out'] = b_out

                            # np.savez(os.path.join(m,'model_params.npz'),**model_params)
