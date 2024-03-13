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
elif ui == 'lndrisco':
    p = '/home/users/lndrisco/'

net = 'stepnet'#'transfer_learn'#
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net) 
sys.path.insert(0, PATH_YANGNET)

from tools_lnd import get_model_params

# parse input arguments as:
rnn_type = 'LeakyRNN'
activation = 'softplus'
init = 'randgauss'
n_rnn = int(256)
l2w = float(-6)
l2h = float(-6)
l1w = float(0)
l1h = float(0)
lr = float(-7)
seed = str(0)

rule_trains_all = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

for pop_rule in range(len(rule_trains_all)):

    rule_trains_pre = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
    'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    rule_trains_pre.pop(pop_rule)

    post_train = rule_trains_all[pop_rule]

    s = '_'
    rule_trains_str = s.join(rule_trains_pre)

    m = os.path.join(p,'code','multitask-nets',net,'data',rnn_type,activation,init,str(len(rule_trains_pre))+'_tasks',
        str(n_rnn)+'_n_rnn','lr'+"{:.1f}".format(-lr)+'l2_w'+"{:.1f}".format(-l2w)+'_h'+"{:.1f}".format(-l2h)+'_'+rule_trains_str,seed)

    model_params = {}
    w_in, b_in, w_out, b_out  = get_model_params(m)
    
    model_params['w_in'] = w_in
    model_params['b_in'] = b_in
    model_params['w_out'] = w_out
    model_params['b_out'] = b_out

    np.savez(os.path.join(m,'model_params.npz'),**model_params)
