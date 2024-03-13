from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import numpy as np
import numpy.random as npr
import tensorflow as tf
import sys
import pdb
import getpass

ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'
elif ui == 'lndrisco':
    p = '/home/users/lndrisco'

net = 'transfer_learn'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net)

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map
from network import FixedPoint_Model
import tools
from tools_lnd import gen_trials_from_model_dir, align_output_inds

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

##################################################################
#Find right model dir
##################################################################
model_n = 1
# dir_specific_all = 'crystals_no_noise/softplus/l2w0001/'#'stepnet/crystals/softplus/l2w0001'#'crystals/softplus/no_reg'#''crystals/softplus/l2h00001'#'stepnet/crystals/softplus/'#grad_norm_both/'#'lowD/combos'#'stepnet/lowD/tanh'#'lowD/grad_norm_l2001' #' #'lowD/armnet_noreg'#lowD/combos' ##grad_norm_l2h000001' /Documents/data/rnn/multitask/varGo/lowD/most/

# dir_specific_all = 'stepnet/crystals/softplus/l2w0001'#'crystals_no_noise/softplus/l2w0001/'#'crystals/softplus/no_reg'#''crystals/softplus/l2h00001'#'stepnet/crystals/softplus/'#grad_norm_both/'#'lowD/combos'#'stepnet/lowD/tanh'#'lowD/grad_norm_l2001' #' #'lowD/armnet_noreg'#lowD/combos' ##grad_norm_l2h000001' /Documents/data/rnn/multitask/varGo/lowD/most/
dir_specific_all = 'stepnet/crystals/softplus/two_tasks/l2h00001_fdgo_fdanti_delaygo_delayanti/'
model_dir_all = os.path.join(p,'data/rnn/multitask/',dir_specific_all,str(model_n))

data_folder = 'data/rnn/multitask/stepnet/crystals/softplus'
file_spec = 'transfer_learn/start_from_best/14_to_1_tasks/256_n_rnn/l2_w4_h5_delayanti/1/'
model_dir_all = os.path.join(p,data_folder,file_spec)

task_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

task_list = ['reactgo','delayanti']
epoch = 'go1'

##################################################################
def project2d(x,axes):

    if x.ndim>1:
        n_steps = x.shape[1]
    else:
        n_steps = 1

    z = np.zeros((n_steps,2))
    z[:,0] = np.dot(axes[:,0],x)
    z[:,1] = np.dot(axes[:,1],x)
    return z

def add_unique_to_inputs_list(dict_list, key, value):
    for d in range(len(dict_list)):
        if (dict_list.values()[d]==value).all():
            return False, dict_list

    dict_list.update({key : value})
    return True, dict_list

def get_filename(task_list,t,step):
    filename = task_list[0]+'_'+task_list[1]+'_trial_'+str(t)+'_step_'+str(step)

    return filename


##################################################################
#Run fixed pt finder
##################################################################

'''Initial states are sampled from states observed during realistic behavior
of the network. Because a well-trained network transitions instantaneously
from one stable state to another, observed networks states spend little if any
time near the unstable fixed points. In order to identify ALL fixed points,
noise must be added to the initial states before handing them to the fixed
point finder.'''
NOISE_SCALE = 0.05 #0.01 #0.5 # Standard deviation of noise added to initial states
N_INITS = 1000 # The number of initial states to provide
n_interp = 20 # number of steps between input conditions

rule1 =  task_list[0]
rule2 =  task_list[1]

trial1 = gen_trials_from_model_dir(model_dir_all,rule1,mode='test',noise_on = False)
trial2 = gen_trials_from_model_dir(model_dir_all,rule2,mode='random',noise_on = False,batch_size = 2000)
trial2 = align_output_inds(trial1, trial2) 
trial1 = gen_trials_from_model_dir(model_dir_all,rule1,mode='test',noise_on = False)

model = FixedPoint_Model(model_dir_all)
with tf.Session() as sess:
    model.restore()
    model._sigma=0
    # get all connection weights and biases as tensorflow variables
    var_list = model.var_list
    # evaluate the parameters after training
    params = [sess.run(var) for var in var_list]
    # get hparams
    hparams = model.hp

    feed_dict1 = tools.gen_feed_dict(model, trial1, hparams)
    h_tf1, y_hat_tf1 = sess.run([model.h, model.y_hat], feed_dict=feed_dict1) #(n_time, n_condition, n_neuron) 

    feed_dict2 = tools.gen_feed_dict(model, trial2, hparams)
    h_tf2, y_hat_tf2 = sess.run([model.h, model.y_hat], feed_dict=feed_dict2) #(n_time, n_condition, n_neuron) 

    ##################################################################
    # get shapes   
    n_steps, n_trials, n_input_dim = np.shape(trial1.x)
    n_rnn = np.shape(h_tf1)[2]
    n_output = np.shape(y_hat_tf1)[2]

    # Fixed point finder hyperparameters
    # See FixedPointFinder.py for detailed descriptions of available
    # hyperparameters.
    fpf_hps = {'tol_q': 1e-9}
    alr_dict = ({'decrease_factor' : .95, 'initial_rate' : 1})

    trial_set = [trial1, trial2]
    e_lims = np.zeros((2,len(trial_set)))

    for ti in range(len(trial_set)):
        trial = trial_set[ti]
        e_start = max([0, trial.epochs[epoch][0]])
        e_lims[0,ti] = int(e_start)
        end_set = [np.shape(trial.x)[0], trial.epochs[epoch][1]]
        e_end = min(x for x in end_set if x is not None)
        e_lims[1,ti] = int(e_end)

    n_inputs = 0
    input_set = {str(n_inputs) : np.zeros((1,n_input_dim))}


    print(e_lims[1,0],e_lims[1,1])
    h_tf_cat1 = h_tf1[:int(e_lims[1,0]),:,:]
    h_tf_cat2 = h_tf2[:int(e_lims[1,1]),:,:]
    h_tf_cat = np.concatenate((h_tf_cat1,h_tf_cat2),axis = 0)

    example_predictions = {'state': np.transpose(h_tf_cat,(1,0,2))}

    fp_predictions = []

    for t in range(0,n_trials,int(n_trials/4)): #np.arange(0, 40, 8): #0,n_trials,40

        inputs_1 = trial1.x[int(e_lims[0,0]),t,:]
        inputs_2 = trial2.x[int(e_lims[0,1]),int(t+n_trials/2),:]
        del_inputs = inputs_2 - inputs_1

        for step_i in range(n_interp):

            step_inputs = inputs_1[np.newaxis,:]+del_inputs[np.newaxis,:]*(step_i/n_interp)
            inputs = step_inputs
            inputs_big = inputs[np.newaxis,:]

            unique_input, input_set = add_unique_to_inputs_list(input_set, str(n_inputs), inputs)
            
            # if unique_input:
            n_inputs+=1
            input_set[str(n_inputs)] = inputs

            fpf = []
            fpf = FixedPointFinder(model.cell, sess, alr_hps=alr_dict, method='joint', verbose = False, **fpf_hps) #do_compute_input_jacobians = True , q_tol = 1e-1, do_q_tol = True
            
            if np.shape(fp_predictions)[0]==0:
                fp_predictions = fpf.sample_states(example_predictions['state'],
                                            n_inits=N_INITS,
                                            noise_scale=NOISE_SCALE)

            unique_fps, all_fps = fpf.find_fixed_points(fp_predictions, inputs)


            if unique_fps.xstar.shape[0]>0:

                fp_predictions = fpf.sample_states(unique_fps.xstar[np.newaxis,:,:],
                                            n_inits=N_INITS,
                                            noise_scale=NOISE_SCALE)

                save_dir = os.path.join(model_dir_all,'fp_stepX_tasks_transfer',epoch,task_list[0]+'_'+task_list[1])#,'random_trials'
                filename = get_filename(task_list, t, step_i)

                all_fps = {}
                all_fps = {'xstar':unique_fps.xstar,
                    # 'J_inputs':unique_fps.J_inputs, 
                    'J_xstar':unique_fps.J_xstar, 
                    'qstar':unique_fps.qstar, 
                    'inputs':unique_fps.inputs, 
                    'epoch_inds':range(e_start,e_end),
                    'noise_var':NOISE_SCALE,
                    'trial_num':t,
                    'state_traj':example_predictions['state']}

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.savez(os.path.join(save_dir,filename+'.npz'),**all_fps)

                    # plt.title(trial.epochs.keys()[epoch])
                    # plt.savefig(os.path.join(save_dir,filename + trial.epochs.keys()[epoch] + '.png'))


                    # Visualize identified fixed points with overlaid RNN state trajectories
                    # All visualized in the 3D PCA space fit the the example RNN states.
                    # t_set = range(5*(t//5),-5*(-(t+1)//5)) #get multiples of 5 trials
                    #t_set = range(0,40)

                    #fpf.plot_summary(example_predictions['state'][t_set,e_start:e_end,:])
                    #plt.savefig(os.path.join(save_dir,filename +'3D.png'))
                    # pdb.set_trace()