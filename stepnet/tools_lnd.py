from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import re
import json
from datetime import datetime as datetime
from tensorflow.python.ops import parallel_for as pfor
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from numpy import linalg as LA
import numpy.random as npr
from scipy import stats
from sklearn.manifold import MDS
from scipy.spatial import distance
from tensorflow.python.ops import parallel_for as pfor
import absl
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

import task
from task import generate_trials, rule_name, rules_dict
from network import Model, FixedPoint_Model, get_perf
import tools
import train
from collections import OrderedDict
from analysis import clustering, standard_analysis, variance
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
import matplotlib.patches as mpatches

rule_set_names = ['DelayPro', 'ReactPro', 'MemoryPro', 'DelayAnti', 'ReactAnti', 'MemoryAnti',
              'IntegrationModality1', 'IntegrationModality2', 'CtxtIntModality1', 'CtxtIntModality2', 'IntegrationMultimodal',
              'ReactMatch2Sample', 'ReactNonMatch2Sample', 'ReactCategoryPro', 'ReactCategoryAnti']

task_name_dict = {}
task_name_dict['Dly Anti'] = rule_set_names[5]
task_name_dict['RT Go'] = rule_set_names[1]
task_name_dict['Dly Go'] = rule_set_names[2]
task_name_dict['RT Anti'] = rule_set_names[4]
task_name_dict['Anti'] = rule_set_names[3]
task_name_dict['Go'] = rule_set_names[0]
task_name_dict['DNMS'] = rule_set_names[12]
task_name_dict['DMS'] = rule_set_names[11]
task_name_dict['DMC'] = rule_set_names[13]
task_name_dict['DNMC'] = rule_set_names[14]
task_name_dict['Dly DM 2'] = rule_set_names[7]
task_name_dict['Dly DM 1'] = rule_set_names[6]
task_name_dict['Ctx Dly DM 2'] = rule_set_names[9]
task_name_dict['Ctx Dly DM 1'] = rule_set_names[8]
task_name_dict['MultSen Dly DM'] = rule_set_names[10]

def load_X_from_model_dir(model_dir,rule, mode = 'test',noise = False,alpha_mod1 = 0):    
    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode=mode, batch_size = 20, noise_on=noise, delay_fac =1,alpha_mod1 = alpha_mod1)
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
            
    return h_tf, y_hat, hparams, trial

def gen_trials_from_model_dir(model_dir,rule,mode='test',noise_on = True,batch_size = 500):
    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
#         params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial
        trial = generate_trials(rule, hparams, mode=mode, noise_on=noise_on, batch_size =batch_size)
    return trial  

def gen_X_from_model_dir(model_dir,trial,d = [],lesion_units_list = []):
    model = Model(model_dir)
    with tf.Session() as sess:

        if len(d)==0:
            model.restore()
        else:
            model.saver.restore(sess,d)

        if len(lesion_units_list)>0:
            model.lesion_units(sess, lesion_units_list)

        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        hparams = model.hp
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        x = np.transpose(h_tf,(2,1,0)) # h_tf[:,range(1,n_trials),:],(2,1,0))
        X = np.reshape(x,(x.shape[0],-1))
    return X, x    #return orthogonal complement of hidden unit activity to ouput projection matrix

def gen_X_from_model_dir_epoch(model_dir,trial,epoch,d = []):
    model = Model(model_dir)
    with tf.Session() as sess:

        if len(d)==0:
            model.restore()
        else:
            model.saver.restore(sess,d)

        model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial       
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)

        if trial.epochs[epoch][1] ==None:
            epoch_range = range(trial.epochs[epoch][0],np.shape(h_tf)[0])
        elif trial.epochs[epoch][0] ==None:
            epoch_range = range(0,trial.epochs[epoch][1])
        else:
            epoch_range = range(trial.epochs[epoch][0],trial.epochs[epoch][1])

        x = np.transpose(h_tf[epoch_range,:,:],(2,1,0)) #h_tf[:,range(1,n_trials),:],(2,1,0))
        X = np.reshape(x,(x.shape[0],-1))
    return X, x    #return hidden unit activity

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    
def get_ylim_diff(ax):
    [y1,y2] = ax.get_ylim()
    y1 = y1+.2*abs(y1)
    y2 = y2+.2*abs(y2)
    y_diff = y2-y1
    return y1,y2,y_diff

def get_xlim_diff(ax):
    [x1,x2] = ax.get_xlim()
    x1 = x1+.2*abs(x1)
    x2 = x2+.2*abs(x2)
    x_diff = x2-x1
    return x1,x2,x_diff

def get_zlim_diff(ax):
    [z1,z2] = ax.get_zlim()
    z1 = z1+.2*abs(z1)
    z2 = z2+.2*abs(z2)
    z_diff = z2-z1
    return z1,z2,z_diff

def make_lil_axes(ax,axes_labels,fontsize = 20,fac_len = 10):
    
    remove_spines(ax)

    x1,x2,x_diff = get_xlim_diff(ax)
    y1,y2,y_diff = get_ylim_diff(ax)
    if len(axes_labels)>2:
        z1,z2,z_diff = get_zlim_diff(ax)
        
        plt.plot([x1,x1+x_diff/fac_len],[y1,y1],[z1,z1],'-k')
        plt.plot([x1,x1],[y1,y1+y_diff/fac_len],[z1,z1],'-k')
        plt.plot([x1,x1],[y1,y1],[z1,z1+z_diff/fac_len],'-k')
        
        ax.text(x1,y1-y_diff/8,z1,axes_labels[0],(1,0,0),
            horizontalalignment='left',
            verticalalignment='top',
                 fontsize = fontsize)
        ax.text(x1-x_diff/8,y1+y_diff/5,z1,axes_labels[1],(0,1,0),
            horizontalalignment='left',
            verticalalignment='bottom',
                 fontsize = fontsize)
        ax.text(x1-x_diff/15,y1,z1+z_diff/5,axes_labels[2],
            horizontalalignment='right',
            verticalalignment='bottom',
                 fontsize = fontsize)
        
        ax.set_zticks([])
        
    else:
        plt.plot([x1,x1+x_diff/10],[y1,y1],'-k')
        plt.plot([x1,x1],[y1,y1+y_diff/10],'-k')

        plt.text(x1,y1-y_diff/50,axes_labels[0],
            horizontalalignment='left',
            verticalalignment='top',
                 fontsize = fontsize)
        plt.text(x1-x_diff/5,y1,axes_labels[1], #/6
            horizontalalignment='left',
            verticalalignment='bottom',
                 fontsize = fontsize,
                 rotation = 90)
        
    ax.set_yticks([])
    ax.set_xticks([])

def plot_training_history(m,rule_trains):

    fig = plt.figure(figsize=(10, 5))
    fname = os.path.join(m, 'log.json')

    with open(fname, 'r') as f:
        log_all = json.load(f)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Define a colormap for each task
    cmap = plt.cm.get_cmap('Greys')

    # Plot cost and performance for each rule
    for i, rule in enumerate(rule_trains):
        color = cmap(i+1 / (len(rule_trains) + 2))  # Normalize color index

        ax1.plot(np.log(log_all['cost_'+rule]), '-', color=color, alpha=0.5, label=rule)
        ax2.plot(log_all['perf_'+rule], '-', color=color, alpha=0.5, label=rule)

    # Set axis labels and title
    ax1.set_xlabel('Training Step (x 1000)')
    ax1.set_ylabel('Log Cost [for each task]')
    ax2.set_xlabel('Training Step (x 1000)')
    ax2.set_ylabel('Performance [for each task]')

    # Add legend
    ax2.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_var_x_epoch(m, tick_labels, figpath, n_components = 256, timing_sort=True, sum_dims = 2, cmap = 'cividis', fontsize = 14):
    var_test_sort, tick_labels, keys = var_x_epoch(m, n_components = n_components, timing_sort=timing_sort)
    
    fig = plt.figure(figsize=(4,2),tight_layout=True,facecolor='white')
    ax = plt.subplot(111)
    
    im = plt.imshow(np.sum(var_test_sort[:,:,:sum_dims],axis = 2),cmap = cmap)
    
    ax.set_yticks(range(len(var_test_sort)))
    ax.set_yticklabels(tick_labels, fontsize = fontsize)
    ax.set_xticks([])
    plt.title('Fraction var. in first '+ str(sum_dims) + ' dim', fontsize = fontsize)
    
    cbar = plt.colorbar(im, ticks=[ 0,.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize = fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.clim(0, 1) 
    
    ax.set_xlabel('Test', fontsize = fontsize)
    ax.set_ylabel('Fit', fontsize = fontsize)
    
    cbar.outline.set_visible(False)
    remove_spines(ax)
    
    plt.savefig(os.path.join(figpath,'var_test_sort_random'+'.pdf'),bbox_inches='tight')
    plt.show()

    return var_test_sort

def plot_var_cumsum(var_test_sort, tick_labels, figpath, 
                    n_dims = 11, 
                    cmap = ['k','orange','dodgerblue','mediumseagreen',],
                    fontsize = 14):

    fig = plt.figure(figsize=(4,2),tight_layout=True,facecolor='white')
    ax1 = plt.subplot(111)
    
    for x in range(0,len(var_test_sort)):
        ax1.plot(np.cumsum(var_test_sort[x,x,:]),'.-',label = tick_labels[x],
                 c = cmap[x],alpha = .6,linewidth = 2,markersize = 13)
           
    ax1.set_xlabel('N PCs', fontsize = fontsize)
    ax1.set_ylabel('Var Expl.', fontsize = fontsize)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.set_xlim([-1,n_dims-.5])
    ax1.set_ylim([-.1,1.1])
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    
    plt.legend(bbox_to_anchor=(1.05, .95), fontsize = fontsize)
    
    plt.savefig(os.path.join(figpath,'var_skree_random'+'.pdf'),bbox_inches='tight')
    plt.show()

    return n_dims

def make_axes_nice(ax,t_len,trial_f,x_ax_label=False,plot_divisions=True,ylims = [-1,1],label = [],label_y = -1):
    remove_ax_lines(ax)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    
    #all xlims are the same
    ax.set_xlim([0,t_len]) 
    
    #set ylims
    ax.spines['left'].set_bounds(ylims[0],ylims[1])
    ax.set_ylim((ylims[0]-.3,ylims[1]+.3))
    ax.set_yticks((ylims[0],ylims[1]))
    
    if x_ax_label==False:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([]) 
    else:
        ax.set_xlabel('Time (ms)')
        
    if plot_divisions==True:
        if label=='Fixation':
            plot_label = True
        else:
            plot_label = False
                
        plot_epoch_division_nolabel(ax, trial_f, ylims, plot_label=plot_label)
        
    ax.text(0,label_y,label, fontsize=14, fontweight='bold')

def plot_epoch_division_nolabel(ax, trial, y_lim, plot_label = False):
    
    for e in trial.epochs.keys():
        
        e_start = 0 if trial.epochs[e][0] is None else trial.epochs[e][0]
        e_end = -1 if trial.epochs[e][1] is None else trial.epochs[e][1]
    
        if plot_label ==True:
            e_name ,_ ,_ ,_ = take_names(e,'fdgo')
            ax.text(10*e_start+20,1.2,e_name,rotation = 0,fontsize = 14)
        if e_end>0:
            ax.plot([10*(trial.epochs[e][1]-1),10*(trial.epochs[e][1]-1)],[-1.8,1.8],'-k',linewidth = 1,alpha = .3)
    ax.set_ylim((y_lim[0]-1,y_lim[1]+1))
        

def adjust_ax(ax,lims,threeD,bifurc,plot_zero_plane,view_ang):
    if not bifurc:
        ax.set_xticks([])

    ax.set_yticks([])

    if threeD:

        if not plot_zero_plane:
            ax.set_zticks([])

        if len(view_ang)>0:
            ax.view_init(elev=view_ang[0], azim=view_ang[1])

        if len(lims)>0:
            ax.set_xlim([lims[0],lims[1]])
            ax.set_ylim([lims[2],lims[3]])
            ax.set_zlim([lims[4],lims[5]])
        else:
            [x1,x2] = ax.get_xlim()
            [y1,y2] = ax.get_ylim()
            [z1,z2] = ax.get_zlim()
            lims = [x1,x2,y1,y2,z1,z2]

        check_plot_zero_plane(ax,plot_zero_plane)
        
    else:

        if len(lims)>0:
            ax.set_xlim([lims[0],lims[1]])
            ax.set_ylim([lims[2],lims[3]])
        else:
            [x1,x2] = ax.get_xlim()
            [y1,y2] = ax.get_ylim()
            lims = [x1,x2,y1,y2]

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    ax.dist = 12
    return ax

def restore_ckpt(model_dir, ckpt_n):
    ckpt_n_dir = os.path.join(model_dir,'ckpts/model.ckpt-' + str(int(ckpt_n)) + '.meta')
    model = Model(model_dir)
    with tf.Session() as sess:
        model.saver.restore(sess,ckpt_n_dir)
    return model

def find_ckpts(model_dir):
    s_all = []
    ckpt_n_dir = os.path.join(model_dir,'ckpts/')
    for file in os.listdir(ckpt_n_dir):
        if file.endswith('.meta'):
            m = re.search('model.ckpt(.+?).meta', file)
            if m:
                found = m.group(1)
            s_all = np.concatenate((s_all,np.expand_dims(abs(int(found)),axis=0)),axis = 0)
    return s_all.astype(int)

def name_best_ckpt(model_dir,rule):
    s_all = find_ckpts(model_dir)
    s_all_inds = np.sort(s_all)
    s_all_inds = s_all_inds.astype(int)
    fname = os.path.join(model_dir, 'log.json')
    
    with open(fname, 'r') as f:
        log_all = json.load(f)
        x = log_all['cost_'+rule]
           
    y = [x[int(j/1000)] for j in s_all_inds[:-1]]
    ind = int(s_all_inds[np.argmin(y)])
    return ind

def get_model_params(model_dir,ckpt_n_dir = []):

    model = Model(model_dir)
    with tf.Session() as sess:
        if len(ckpt_n_dir)==0:
            model.restore()
        else:
            model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]

    w_in = params[0]
    b_in = params[1]
    w_out = params[2]
    b_out = params[3]

    return w_in, b_in, w_out, b_out

# def get_path_names():
#     import getpass
#     ui = getpass.getuser()
#     if ui == 'laura':
#         p = '/home/laura'
#     elif ui == 'lauradriscoll':
#         p = '/Users/lauradriscoll/Documents'
#     return p

def plot_N(X, D, clist, linewidth = 1, alpha = .5, linestyle = '-', cmap_c = 'hsv',
    markersize = 10, edgecolors = []):
    """Plot activity is some 2D space.

        Args:
            X: neural activity in Trials x Time x Neurons
            D: Neurons x 2 plotting dims
        """

    cmap=plt.get_cmap(cmap_c)
    S = np.shape(X)[0]
    
    print(len(clist))
    
    for s in range(S):

        if len(clist)<2:
            c = clist[0]
        elif len(clist)==4:
            c = clist
        else:
            c = cmap(clist[s]/max(clist))

        X_trial = np.dot(X[s,:,:],D.T)
        plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, linewidth = linewidth, alpha = alpha, markersize = markersize,markeredgewidth = linewidth)
        plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = c, linewidth = linewidth, alpha = alpha, markersize = markersize)
        plt.plot(X_trial[0,0],X_trial[0,1],'x',c = c, linewidth = linewidth, alpha = alpha,markersize = markersize,markeredgewidth = linewidth)
        if len(edgecolors)>0:
            plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = edgecolors, linewidth = linewidth/2, alpha = alpha, markersize = markersize)


def plot_N3D(ax, X, D, clist, linewidth = 1, alpha = .5, linestyle = '-', cmap_c = 'hsv',markersize = 10, edgecolors = []):
    """Plot activity is some 2D space.

        Args:
            X: neural activity in Trials x Time x Neurons
            D: Neurons x 2 plotting dims
        """

    cmap=plt.get_cmap(cmap_c)
    S = np.shape(X)[0]
    
    for s in range(S):

        if isinstance(clist, str) :
            c = clist
        elif len(clist)==1:
            c = clist[0]
        else:
            c = cmap(clist[s]/max(clist))

        X_trial = np.dot(X[s,:,:],D.T)
        ax.plot3D(X_trial[:,0],X_trial[:,1],X_trial[:,2],linestyle,
                c = c, linewidth = linewidth, alpha = alpha)
        ax.scatter(X_trial[0,0],X_trial[0,1],X_trial[0,2],s = markersize,marker = 'x',
                c = c, linewidth = linewidth, alpha = alpha)

        if len(edgecolors) == 0:
            ax.scatter(X_trial[-1,0],X_trial[-1,1],X_trial[-1,2],s = markersize,marker ='^',
                c = c, linewidth = linewidth, alpha = alpha)
        else:
            ax.plot3D(X_trial[:,0],X_trial[:,1],X_trial[:,2],linestyle,
                c = edgecolors, linewidth = linewidth/2, alpha = alpha)
            ax.scatter(X_trial[-1,0],X_trial[-1,1],X_trial[-1,2],s = markersize,marker ='^',
                edgecolors = c, c = edgecolors, linewidth = linewidth/2, alpha = alpha)


# def plot_FP(X, D, eig_decomps, c='k', al = .2, lw = .5):

#     """Plot activity is some 2D space.

#         Args:
#             X: Fixed points in #Fps x Neurons
#             D: Neurons x 2 plotting dims
    
#         """
#     S = np.shape(X)[0]
#     lf = 7
#     rf = 7
#     D = D[:2,:] #reduce dim in >2
    
#     for s in range(S):
        
#         X_trial = np.dot(X[s,:],D.T)
        
#         n_arg = np.argwhere(eig_decomps[s]['evals']>1)+1
#         if len(n_arg)>0:
#             for arg in range(np.max(n_arg)):
#                 rdots = np.dot(np.real(eig_decomps[s]['R'][:, arg]).T,D.T)
#                 ldots = np.dot(np.real(eig_decomps[s]['L'][:, arg]).T,D.T)
#                 overlap = np.dot(rdots,ldots.T)
#                 r = np.concatenate((X_trial - rf*overlap*rdots, X_trial + rf*overlap*rdots),0)
#                 plt.plot(r[0:4:2],r[1:4:2], c = 'k' ,alpha = .2,linewidth = .5)
        
#         n_arg = np.argwhere(eig_decomps[s]['evals']<.3)
#         if len(n_arg)>0:
#             for arg in range(np.min(n_arg),len(eig_decomps[s]['evals'])):
#                 rdots = np.dot(np.real(eig_decomps[s]['R'][:, arg]).T,D.T)
#                 ldots = np.dot(np.real(eig_decomps[s]['L'][:, arg]).T,D.T)
#                 overlap = np.dot(rdots,ldots.T)
#                 r = np.concatenate((X_trial - rf*overlap*rdots, X_trial + rf*overlap*rdots),0)
#                 plt.plot(r[0:4:2],r[1:4:2],'b',alpha = .2,linewidth = .5)

#         if np.max(eig_decomps[s]['evals'].real)<1:
#             markerfacecolor = c
#         else:
#             markerfacecolor = 'None'
            
#         plt.plot(X_trial[0], X_trial[1], 'o', markerfacecolor = markerfacecolor, markeredgecolor = c, markersize = 10, alpha = .5)

        
def plot_FP_jitter_3D(m,D_use,rule,t_num,fp_epoch,sorted_fps,fp_inds,jit_fps=True,
                   xlabel = 'FP set PC1',ylabel = 'FP set PC2',rand_step_coef = 0.1,n_steps = 100,
                   lw = 3,al = .6,linestyle = '-',n_jit = 0,unstable_qlim = -6,c = 'k'):

    cmap=plt.get_cmap('rainbow')
        
    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        hparams = model.hp
        alpha = hparams['dt']/hparams['tau']
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]

        trial = generate_trials(rule, hparams, mode='test',noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf, _ = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        T,S,N = np.shape(h_tf)
        T_inds = get_T_inds(trial,fp_epoch) # grab epoch time indexing
        x_t = np.matlib.repmat(trial.x[T_inds[1],t_num,:],n_steps,1)
        
        if jit_fps==True:
            for fp_ind in fp_inds:
                for jit in range(n_jit):
                    h0 = sorted_fps[fp_ind,:] + rand_step_coef*npr.randn(N)
                    h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
                    jitter = np.dot(h_t,D_use)
                    ax.plot3D(jitter[:,0],jitter[:,1],jitter[:,2],'-',c = 'k',linewidth = .1)
                
        
        for jit in range(1):
            h0 = h_tf[T_inds[0],t_num,:]
            h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
            jitter = np.dot(h_t,D_use)
            ax.plot3D(jitter[:,0],jitter[:,1],jitter[:,2],'-',c = c,linewidth = 3)

def dst_to_h(h,sorted_fps):
    X = np.squeeze(sorted_fps).astype(np.float64)
    dst = np.zeros((np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
            dst[xi] = distance.euclidean(h, X[xi,:])
    return dst

def proximate_fp(h,fps):
    if len(fps)==1:
        proximate_fps = 0
    else:
        sorted_fps_list = np.argsort(dst_to_h(h,fps))
        proximate_fps = sorted_fps_list[0]
    return proximate_fps

def load_proximal_fp_from_f(f,h):
    fp_struct = np.load(f)
    xstar = fp_struct['xstar']
    
    proximate_fps = proximate_fp(h,xstar)
    fp_num = proximate_fps
    evals, _ = LA.eig(fp_struct['J_xstar'][fp_num,:,:])
    
    return xstar, fp_struct['J_xstar'], fp_num, evals

class rnn_obj(object):

    default_hps = {
        'tf_dtype': 'float32',
        'random_seed': 0, #IS THIS RIGHT? FIX
        'rnn_cell_feed_dict': {},
    }

    def __init__(self, rnn_cell, sess,
        random_seed=default_hps['random_seed'],
        tf_dtype=default_hps['tf_dtype'],
        rnn_cell_feed_dict=default_hps['rnn_cell_feed_dict']):
        
        self.rnn_cell = rnn_cell
        self.rnn_cell_feed_dict = rnn_cell_feed_dict
        self.session = sess
        self.tf_dtype = getattr(tf, tf_dtype)
        
    def _grab_RNN(self, initial_states, inputs):

        x = tf.Variable(initial_states, dtype=self.tf_dtype)
        x_rnncell = x

        inputs_tf = tf.constant(inputs, dtype=self.tf_dtype)

        output, F_rnncell = self.rnn_cell(inputs_tf, x_rnncell)
        F = F_rnncell
        
        init = tf.variables_initializer(var_list=[x,])
        self.session.run(init)

        return x, F

def calc_Jac(sess,rnn_o,states,inputs):
    x_tf, F_tf = rnn_o._grab_RNN(states, inputs)
    try:
       dFdx_tf = pfor.batch_jacobian(F_tf, x_tf)
    except absl.flags._exceptions.UnparsedFlagAccessError:
       dFdx_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor=False)
    dFdx_np = sess.run(dFdx_tf)
    return dFdx_np

def calc_relevant_jacobian(m,fp_struct,fp_num,state_diff = []):
    
    model = FixedPoint_Model(m)
    with tf.Session() as sess:
        model.restore()
        hp = model.hp

        rnn_o_hps = {}
        rnn_o = []
        rnn_o = rnn_obj(model.cell,sess, **rnn_o_hps) 

        if len(state_diff)==0:
            state = fp_struct['xstar'][fp_num][np.newaxis,:]
        else:
            state = state_diff[np.newaxis,:]

        inputs = fp_struct['inputs'][fp_num][np.newaxis,:]

        dFdx = calc_Jac(sess,rnn_o,state,inputs)
    return dFdx

def calc_jacobian_at(m,inputs,state,lesion_units = []):
    
    model = FixedPoint_Model(m)
    with tf.Session() as sess:

        model.restore()

        if len(lesion_units)>0:
            model.lesion_units(sess, lesion_units)

        hp = model.hp

        rnn_o_hps = {}
        rnn_o = []
        rnn_o = rnn_obj(model.cell,sess, **rnn_o_hps) 

        dFdx = calc_Jac(sess,rnn_o,state,inputs)
    return dFdx

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rot_mat(theta):
    R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    return R

def calc_R_angle(R):
    return np.arccos((np.trace(R)-1)/2)

def tranform_in_rPC(X,R,X_ss):
    Xr_ss = np.dot(R,X_ss.T).T
    Xr = np.dot(R,X.T).T
    if Xr_ss[1,1]>0:
        Xr = np.dot(Xr,np.array(((1,0),(0,-1))))
    return Xr

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def make_Jac_u_dot_delu(model_dir_all,rule,task_set,trial_set):
    n_tasks = len(task_set)
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.restore()
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode='test', noise_on=False)

        time_set = (trial.epochs['stim1'][0],)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - len(hparams['rules'])
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),len(time_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),len(time_set),N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]

            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_u = pfor.batch_jacobian(F_context, inputs_tf_context, use_pfor=False)
                    J_np_u[r,trial_i,time_i,:,:] = sess.run(J_tf_u)
                    J_np_u_dot_delu[r,trial_i,time_i,:] = np.squeeze(np.dot(J_np_u[r,trial_i,time_i,:,:],delta_inputs.T))
                    
    return J_np_u_dot_delu

def make_Jac_x(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):
    n_tasks = len(task_set)
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode='test', noise_on=False)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_x = np.zeros((n_tasks,len(trial_set),len(time_set),N,N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_x = pfor.batch_jacobian(F_context, states_tf, use_pfor=False)
                    J_np_x[r,trial_i,time_i,:,:] = sess.run(J_tf_x)
                    
    return J_np_x

def make_h_and_Jac(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):

    h_context_combined = []
    h_stim_early_combined = []
    h_stim_late_combined = []
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials('delaygo', hparams, mode='test', noise_on=False)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        time_set = [trial.epochs['stim1'][0]] #beginning of stim period

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),len(time_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),len(time_set),N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            # comparing Jacobians to proximity of hidden state across tasks
            # we focus on end of the context period, early, and late in the stim period
            h_context = np.reshape(h_tf[trial.epochs['stim1'][0]-1,trial_set,:],(1,-1)) # h @ end of context period
            h_stim_early = np.reshape(h_tf[trial.epochs['stim1'][0]+n_steps_early,trial_set,:],(1,-1)) # h @ 5 steps into stim
            h_stim_late = np.reshape(h_tf[trial.epochs['stim1'][1],trial_set,:],(1,-1)) # h @ end of stim period

            #concatenate activity states across tasks
            if h_context_combined == []:
                h_context_combined = h_context[np.newaxis,:]
                h_stim_late_combined = h_stim_late[np.newaxis,:]
                h_stim_early_combined = h_stim_early[np.newaxis,:]
            else:
                h_context_combined = np.concatenate((h_context_combined, h_context[np.newaxis,:]), axis=0)
                h_stim_late_combined = np.concatenate((h_stim_late_combined, h_stim_late[np.newaxis,:]), axis=0)
                h_stim_early_combined = np.concatenate((h_stim_early_combined, h_stim_early[np.newaxis,:]), axis=0)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_u = pfor.batch_jacobian(F_context, inputs_tf_context, use_pfor=False)
                    J_np_u[r,trial_i,time_i,:,:] = sess.run(J_tf_u)
                    J_np_u_dot_delu[r,trial_i,time_i,:] = np.squeeze(np.dot(J_np_u[r,trial_i,time_i,:,:],delta_inputs.T))
                    
    return J_np_u_dot_delu, h_context_combined, h_stim_late_combined, h_stim_early_combined

def prep_procrustes(data1, data2):
    """Procrustes analysis, a similarity test for two data sets.
    
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
  
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2
    
    return mtx1,mtx2

def same_stim_trial(trial_master, task_num):
    ins = np.shape(trial_master.x)[2]
    n_stim_per_ring = int(np.shape(trial_master.y)[2]-1)
    stim_rep_size = int(2*n_stim_per_ring+1)
    n_tasks = ins - stim_rep_size
    trial_task_num = trial_master
    trial_task_num.x[:,:,stim_rep_size:] = 0
    trial_task_num.x[:,:,stim_rep_size+task_num] = 1
    #reaction tasks dont have a fixation input
    react_tasks = [1,4,n_tasks-4,n_tasks-3,n_tasks-2,n_tasks-1]
    anti_tasks = [3,4,5,12,14]
    if any(task_num == tn for tn in react_tasks):
        trial_task_num.x[:,:,0] = 1
    if any(task_num == tn for tn in anti_tasks):
        rep_inds = trial_task_num.y_loc>0
        trial_task_num.y_loc[rep_inds] = (trial_task_num.y_loc[rep_inds]+np.pi)%(2*np.pi)
    return trial_task_num

def pca_denoise(X1,X2,nD):
    pca = PCA(n_components = nD)
    X12 = np.concatenate((X1,X2),axis=1)
    _ = pca.fit_transform(X12.T)
    X1_pca = pca.transform(X1.T)
    X2_pca = pca.transform(X2.T)
    
    return X1_pca, X2_pca

def same_mov_inds(trial_master, trial_temp): #used to be called align_output_inds

    indices = range(np.shape(trial_master.y_loc)[1])
    n_out = np.shape(trial_master.y)[2]-1

    for ii in range(np.shape(trial_master.y_loc)[1]):
        if np.max(np.sum(abs(trial_master.x[:,ii,1:(1+n_out)]),axis = 1),axis = 0)>0:
            ind_use = np.max(np.sum(abs(trial_temp.x[:,:,1:(1+n_out)]),axis = 2),axis = 0)>0
        else:
            ind_use = np.max(np.sum(abs(trial_temp.x[:,:,(1+n_out):(1+2*n_out)]),axis = 2),axis = 0)>0

        loc_diff = abs(trial_temp.y_loc[-1,:]-trial_master.y_loc[-1,ii])%(2*np.pi)
        align_ind = [int(i) for i, x in enumerate(loc_diff) if x == min(loc_diff)]
        align_ind_choosey = [x for i, x in enumerate(align_ind) if ind_use[x]]
        if len(align_ind_choosey)==0:
            align_ind_choosey = align_ind
        indices[ii] = align_ind_choosey[npr.randint(len(align_ind_choosey))]
    
    trial_temp_new = trial_temp
    trial_temp_new.x = trial_temp_new.x[:,indices,:]
    trial_temp_new.y = trial_temp_new.y[:,indices,:]
    trial_temp_new.y_loc = trial_temp_new.y_loc[:,indices]
    return trial_temp_new

def same_mov_inds_py3(trial_master, trial_temp):
    n_trials = np.shape(trial_master.y_loc)[1]
    n_out = np.shape(trial_master.y)[2] - 1

    indices = [None] * n_trials

    for ii in range(n_trials):
        if np.max(np.sum(abs(trial_master.x[:, ii, 1:(1 + n_out)]), axis=1), axis=0) > 0:
            ind_use = np.max(np.sum(abs(trial_temp.x[:, :, 1:(1 + n_out)]), axis=2), axis=0) > 0
        else:
            ind_use = np.max(np.sum(abs(trial_temp.x[:, :, (1 + n_out):(1 + 2 * n_out)]), axis=2), axis=0) > 0

        loc_diff = abs(trial_temp.y_loc[-1, :] - trial_master.y_loc[-1, ii]) % (2 * np.pi)
        align_ind = [int(i) for i, x in enumerate(loc_diff) if x == min(loc_diff)]
        align_ind_choosey = [x for i, x in enumerate(align_ind) if ind_use[x]]

        if not align_ind_choosey:
            align_ind_choosey = align_ind

        if align_ind_choosey:
            indices[ii] = npr.choice(align_ind_choosey)

    trial_temp_new = trial_temp
    trial_temp_new.x = trial_temp_new.x[:, indices, :]
    trial_temp_new.y = trial_temp_new.y[:, indices, :]
    trial_temp_new.y_loc = trial_temp_new.y_loc[:, indices]

    return trial_temp_new

# def out_affine(params, h):
#     offset = np.repeat(params[3][:,np.newaxis],np.shape(h)[1],axis = 1)
#     return np.dot(params[2].T,h)+offset

def init_from_other_task(m,ri_set,init_from):

    rule1 = rules_dict['all'][ri_set[0]]
    rule2 = rules_dict['all'][ri_set[1]]
    trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    trial2 = gen_trials_from_model_dir(m,rule2,mode='test',noise_on = False)
    trial2 = same_mov_inds_py3(trial1, trial2) 
    trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)

    trial_set = [trial1,trial2]
    
    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]
        hparams = model.hp

    epoch = 'go1'
    B = np.shape(trial1.x)[1]
    N = hparams['n_rnn']

    _,h = gen_X_from_model_dir(m,trial_set[init_from])
    T1_inds = get_T_inds(trial1,epoch)
    T2_inds = get_T_inds(trial2,epoch)

    y_hat = np.zeros((len(T1_inds),B,3))
    for ti in range(B):

        h0 = h[:,ti,T2_inds[0]-2]
        x_t = trial1.x[T1_inds[1:],ti,:]
        h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
        y_hat[:,ti,:] = out_affine(params, h_t.T).T

    y_loc = trial1.y_loc[T1_inds,:]
    return get_perf(y_hat, y_loc)

def project_to_output(model_dir_all,X):
    w_in, b_in, w_out, b_out = get_model_params(model_dir_all)
    y = np.dot(X.T, w_out) + b_out
    return y

def gen_mov_x(model_dir_all,ckpt_n_dir,rule,trial_master):
    trial = gen_trials_from_model_dir(model_dir_all,rule)
    trial = align_output_inds(trial_master, trial)
    _,x = gen_X_from_model_dir_epoch(model_dir_all,ckpt_n_dir,trial,'go1')
    x_out = project_to_output(model_dir_all,x[:,:,-1])
    err = np.sum(np.square(x_out[:,1:] - trial.y[-1,:,1:]),axis=1)
    return err, x

def make_procrustes_mat_mov(model_dir_all,ckpt_n_dir,tasks,nD = 10,err_lim = .2):
    
    procrust = {}
    procrust['Disparity'] = np.zeros((len(tasks),len(tasks)))
    procrust['Scaling'] = np.zeros((len(tasks),len(tasks)))
    procrust['R']= np.zeros((len(tasks),len(tasks)))
    
    rule_master = 'delaygo'
    trial_master = gen_trials_from_model_dir(model_dir_all,rule_master)
    trial_master_test = gen_trials_from_model_dir(model_dir_all,rule_master)

    for t1_ind in range(len(tasks)):
        t1 = tasks[t1_ind]
        err1, x1 = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t1],trial_master)
        _, x1_test = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t1],trial_master_test)

        for t2_ind in range(len(tasks)):
            if t1_ind !=t2_ind:
                t2 = tasks[t2_ind]
                err2, x2 = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t2],trial_master)
                _, x2_test = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t2],trial_master_test)

                if ckpt_n_dir[-1]==str(1):
                    use_trials = np.multiply(err1>-err_lim,err2>-err_lim) #keep all trials
                else:
                    use_trials = np.multiply(err1<err_lim,err2<err_lim)

                if np.sum(use_trials)>8:
                    X2 = np.reshape(x2[:,use_trials,:15],(np.shape(x2)[0],-1))
                    X1 = np.reshape(x1[:,use_trials,:15],(np.shape(x1)[0],-1))

                    X2_test = np.reshape(x2_test[:,use_trials,:15],(np.shape(x2)[0],-1))
                    X1_test = np.reshape(x1_test[:,use_trials,:15],(np.shape(x1)[0],-1))
                else:
                    raise ValueError('Less than ' + str(np.sum(use_trials)) + ' trials to compare.')

                X1_pca,X2_pca = pca_denoise(X1,X2,nD)
                prep_mtx1, prep_mtx2 = prep_procrustes(X1_pca,X2_pca)
                _, _, disparity_train, R, s = procrustes_fit(prep_mtx1, prep_mtx2)

                X1_pca_test,X2_pca_test = pca_denoise(X1_test,X2_test,nD)
                prep_mtx1_test, prep_mtx2_test = prep_procrustes(X1_pca_test,X2_pca_test)
                mtx1, mtx2, disparity_test = procrustes_test(prep_mtx1_test, prep_mtx2_test, R, s)

                procrust['Disparity'][t1_ind,t2_ind] = disparity_test
                procrust['Scaling'][t1_ind,t2_ind] = s
                procrust['R'][t1_ind,t2_ind] = calc_R_angle(R)
    return procrust

def make_h_combined(model_dir_all,tasks,trial_set,epoch,ind = -1,ckpt_n_dir = []):

    # Initialize h_combined with a compatible shape for concatenation
    h_combined = np.zeros((0, np.shape(trial_set)[0], 128))  # Assuming h_temp has 128 features

    model = Model(model_dir_all)
    with tf.Session() as sess:

        rule = 'delaygo'
        if len(ckpt_n_dir)>0:
            model.saver.restore(sess,ckpt_n_dir)
        else:
            model.restore()
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial_master = generate_trials(rule, hparams, mode='random', batch_size = 100,noise_on=True)

        #get size of relevant variables to init mats
        n_inputs = np.shape(trial_master.x)[2]
        N = np.shape(params[0])[1]
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        n_stim_dims = np.shape(trial_master.x)[2]-len(hparams['rules'])
        T_inds = get_T_inds(trial_master,epoch)

        for r in range(len(tasks)):
            r_all_tasks_ind = tasks[r]
            trial = same_stim_trial(trial_master, r_all_tasks_ind)

            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            h_epoch = h_tf[T_inds,:,:]
            h_temp = h_epoch[ind,trial_set,:]

            #concatenate activity states across tasks
            h_combined = np.concatenate((h_combined, h_temp[np.newaxis, :, :]), axis=0)
        h_combined = h_combined.reshape((-1, 128)) 
            
    return h_combined

def make_h_all(m,mode = 'test',rules = []):
    
    model = FixedPoint_Model(m, sigma_rec=0)

    with tf.Session() as sess:
        model.restore()
        model._sigma=0

        h_all_byrule = OrderedDict()
        h_all_byepoch = OrderedDict()
        y_loc_byrule = OrderedDict()
        y_loc_byepoch = OrderedDict()
        h_all_timing = []

        hp = model.hp
        n_hidden = hp['n_rnn']

        if len(rules)==0:
            rules = hp['rules']

        counter = 0

        for rule in rules:
            trial = generate_trials(rule, hp, mode = mode, noise_on=False, batch_size = 1024)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)

            for e_name, e_time in trial.epochs.items():

                time_diff = len(h[e_time[0]:e_time[1],:,:])

                counter += 1 
                h_all_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1],:,:]
                h_all_byrule[rule] = h
                y_loc_byepoch[(rule, e_name)] = np.tile(trial.y_loc[-1,:],(time_diff,1))
                y_loc_byrule[rule] = trial.y_loc

                timing = e_time[0]
                if not e_time[0]:
                    timing = 0
                h_all_timing.append(timing)

    return h_all_byepoch, h_all_byrule, rules, h_all_timing, y_loc_byepoch, y_loc_byrule

def generate_Beta_epoch(h_tf,trial,ind = -1,mod = 'either', ind_adjust = 0):
    Beta_epoch = {}

    for epoch in trial.epochs.keys():

        T_inds = get_T_inds(trial,epoch)
        T_use = T_inds[ind]
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        # X = h_tf[T_use,inds_use,:].T
        # X_zscore = stats.zscore(X, axis=1)
        # X_zscore_nonan = X_zscore
        # X_zscore_nonan[np.isnan(X_zscore)] = 0
        # r = X_zscore_nonan

        r = h_tf[T_use,inds_use,:].T

        if mod == 'either':
            stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
            stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        elif mod==1:
            stim1_locs = trial.stim_locs[:,0]
            stim2_locs = trial.stim_locs[:,1]
        elif mod==2:
            stim1_locs = trial.stim_locs[:,2]
            stim2_locs = trial.stim_locs[:,3]

        y_loc = trial.y_loc[-1,:]

        if epoch == 'stim1' or epoch == 'delay1':
            angle_var = stim1_locs[inds_use]
        elif epoch =='stim2' or epoch == 'delay2':
            angle_var = stim2_locs[inds_use]
        elif epoch =='go1' or epoch == 'fix1':
            angle_var = stim1_locs[inds_use]

        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_epoch[epoch],_ = LA.qr(Beta)

        #Make sure vectors are oriented appropriately
        #first identify a trial that should be in quadrant 1
        quad1_arg = np.argmin((angle_var - np.pi/4)%(2*np.pi))
        quad1_x = h_tf[T_use,quad1_arg,:]
        dr_loc = np.dot(quad1_x,Beta_epoch[epoch])

        #flip vectors so that point is actually in quadrant 1
        if dr_loc[0]<0:
            Beta_epoch[epoch][:,0] = -Beta_epoch[epoch][:,0]
            
        if dr_loc[1]<0:
            Beta_epoch[epoch][:,1] = -Beta_epoch[epoch][:,1]

    return Beta_epoch


def make_Beta(m,task_list,fp_epoch,ind=-1):
    h_all,trial_all,_ = make_h_trial_rule(m,mode = 'test',noise_on = False)
    T_inds = get_T_inds(trial_all[task_list[0]],fp_epoch)
    B,N = np.shape(h_all[task_list[0]][T_inds[ind],:,:])
    R = np.zeros((len(task_list)*B*len(T_inds),N))
    Y = np.zeros((9,len(task_list)*B*len(T_inds)))

    for rule_i in range(len(task_list)):
        rule = task_list[rule_i]
        T_inds = [get_T_inds(trial_all[rule],fp_epoch)[ind],]
        trial_inds = range(0,np.shape(h_all[rule])[1],int(np.shape(h_all[rule])[1]/B))
        h_subselect = h_all[rule][:,trial_inds,:]
        r = np.reshape(h_subselect[T_inds,:,:],(len(T_inds)*B,N))
        
        isanti = 'anti' in task_list[rule_i]
        ismemory = 'delay' in task_list[rule_i]
        y_anti = isanti*np.ones((np.shape(r)[0],1))
        y_memory = ismemory*np.ones((np.shape(r)[0],1))
        y_stim_all = np.repeat(np.min(trial_all[rule].stim_locs[:,[0,2]],axis=1),len(T_inds))
        y_stim = y_stim_all[trial_inds]
        y_out = np.repeat(trial_all[rule].y_loc[-1,trial_inds],len(T_inds))
        
        inds = range((B*len(T_inds))*rule_i,(B*len(T_inds))*(rule_i+1))
        
        R[inds,:] = r
        Y[0,inds] = np.expand_dims(isanti,axis = 0)
        Y[1,inds] = np.expand_dims(ismemory,axis = 0)
        Y[2,inds] = np.expand_dims(np.sin(y_stim),axis = 0)
        Y[3,inds] = np.expand_dims(np.cos(y_stim),axis = 0)
        Y[4,inds] = np.expand_dims(np.sin(y_out),axis = 0)
        Y[5,inds] = np.expand_dims(np.cos(y_out),axis = 0)
        Y[6,inds] = np.repeat(T_inds,B)
        
        Y_labels = ['ANTI','MEMORY','STIM '+ r' $\cos{\theta}$','STIM '+ r' $\sin{\theta}$',
                    'OUT '+ r' $\cos{\theta}$','OUT '+ r' $\sin{\theta}$','TIME']
        
    lm = linear_model.LinearRegression()
    model = lm.fit(Y.T,R)
    Beta = model.coef_
    Beta_qr,_ = LA.qr(Beta)
    return Beta_qr,Y_labels

def make_axes(model_dir_all,rule_master,epoch,ind = -1,mod = 'either',ckpt_n_dir=[]):

    model = Model(model_dir_all)
    with tf.Session() as sess:

        if len(ckpt_n_dir)==0:
            model.restore()
        else:
            model.saver.restore(sess,ckpt_n_dir)

        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial_master = generate_trials(rule_master, hparams, mode = 'test', batch_size = 400, noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial_master, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

    # print(mod)
    Beta_epoch = generate_Beta_epoch(h_tf,trial_master,ind,mod = mod)
    X_pca = Beta_epoch[epoch]    
    D = np.concatenate((np.expand_dims(X_pca[:,0],axis=1),np.expand_dims(X_pca[:,1],axis=1)),axis = 1)
    return D

def get_D(dims,h,trial,tasks,epoch,ind = -1):
    D = {}

    if dims=='pca':
        for ri in range(len(tasks)):
            rule = tasks[ri]
            pca = PCA(n_components = 100)
            X = np.reshape(h[rule],(-1,N))
            _ = pca.fit_transform(X)
            D[rule] = pca.components_
    elif dims=='tdr':
        for ri in range(len(tasks)):
            rule = tasks[ri]
            Beta_temp = generate_Beta_epoch(h[rule],trial[rule],ind = ind)
            if (rule[:2] == 'fd') & (epoch == 'delay1'):
                D[rule] = Beta_temp['stim1'].T
            else:
                D[rule] = Beta_temp[epoch].T
    return D

def get_T_inds(trial,epoch):

    T_end = trial.epochs[epoch][1] 
    if T_end == None:
        T_end = np.shape(trial.x)[0]

    T_start = trial.epochs[epoch][0]
    if T_start == None:
        T_start = 0

    T_inds = range(T_start,T_end)

    return T_inds

def generate_Beta_timeseries(h_tf,trial,T_inds,align_group):
    T,S,N = np.shape(h_tf)
    Beta_timeseries = np.empty((N,2,len(T_inds)))

    for t in T_inds:
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        X = h_tf[t,inds_use,:].T
        X_zscore = stats.zscore(X, axis=1)
        X_zscore_nonan = X_zscore
        X_zscore_nonan[np.isnan(X_zscore)] = 0
        r = X_zscore_nonan

        stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
        stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        y_loc = trial.y_loc[-1,:]
        
        if align_group == 'stim1':
            angle_var = stim1_locs[inds_use]
        elif align_group =='stim2':
            angle_var = stim2_locs[inds_use]
        elif align_group =='go1':
            angle_var = y_loc[inds_use]
        
#         if t<trial.epochs['stim2'][0]:
#             angle_var = stim1_locs[inds_use]
#         elif t<trial.epochs['go1'][0]:
#             angle_var = stim2_locs[inds_use]
#         else:
#             angle_var = y_loc[inds_use]

        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_timeseries[:,:,t],_ = LA.qr(Beta)

    return Beta_timeseries

def get_stim_cats(trial):
    #stim locations and category ids
    stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
    stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    stim2_cats = stim2_locs<np.pi # Category of stimulus 2
    matchs = stim1_cats == stim2_cats
    
    return stim1_locs, stim2_locs, stim1_cats, stim2_cats

def get_Jacs(model_dir_all, ckpt_n_dir, rule_num, trial_master):

    fpf = []
    J_np = {}
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        
        trial = same_stim_trial(trial_master, rule_num)
        
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)
        
        stim1_locs, stim2_locs, stim1_cats, stim2_cats = get_stim_cats(trial)

        stim1 = trial.epochs['stim1']
        stim2 = trial.epochs['go1']

        inputs_np = []
        inputs_np.append(trial.x[stim1[0],:,:])
        inputs_np.append(trial.x[stim2[0],stim1_cats,:])
        inputs_np.append(trial.x[stim2[0],stim1_cats==0,:])

        states_np = []
        states_np.append(h_tf[stim1[0]-1,:,:])
        states_np.append(h_tf[stim2[0]-1,stim1_cats,:])
        states_np.append(h_tf[stim2[0]-1,stim1_cats==0,:])

        fpf = FixedPointFinder(model.cell,sess)
        
        for bi in range(len(states_np)):
            x_tf, F_tf = fpf._grab_RNN(states_np[bi], inputs_np[bi])
            J_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor = False)
            J_np[bi] = fpf.session.run(J_tf)
    
    return J_np

def make_h_trial_rule(model_dir_all,mode = 'random',noise_on = False,task_set = []):
    
    trial = {}
    h = {}
    
    model = FixedPoint_Model(model_dir_all)
    with tf.Session() as sess:

        model.restore()#model.saver.restore(sess,ckpt_n_dir)
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hp = model.hp

        if len(task_set)==0:
            task_set = hp['rule_trains']

        for rule in task_set:
            trial[rule] = generate_trials(rule, hp, mode,
                            batch_size=400,noise_on = noise_on)

            # Generating feed_dict.
            feed_dict = tools.gen_feed_dict(model, trial[rule], hp)
            h[rule] = sess.run(model.h, feed_dict=feed_dict)
            
    return h, trial, hp['rule_trains']

def make_cat_h_rules(model_dir_all,mode = 'random',noise_on = False, task_set = []):
    
    cat_h = []
    
    model = FixedPoint_Model(model_dir_all)
    with tf.Session() as sess:

        model.restore()#model.saver.restore(sess,ckpt_n_dir)
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hp = model.hp
        n_rnn = hp['n_rnn']

        if len(task_set)<1:
            task_set = hp['rule_trains']

        for rule in task_set:
            trial = generate_trials(rule, hp, mode,
                            batch_size=400,noise_on = noise_on)

            # Generating feed_dict.
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)

            h_temp = np.reshape(np.transpose(h,(2,0,1)),(n_rnn,-1))

            if len(cat_h)==0:
                cat_h = h_temp
            else:
                cat_h = np.concatenate((cat_h,h_temp),axis = 1)
            
    return cat_h

def get_fp_filename(trial, epoch,t):
    ind_stim_loc  = str(int(180*trial.y_loc[-1,t]/np.pi))
    nonzero_stim = trial.stim_locs[t,:]<100
    stim_names = '_'.join(str(int(180*x/np.pi)) for x in trial.stim_locs[t,nonzero_stim])
    filename = epoch+'_trial'+str(t)+'_x'+stim_names+'_y'+ind_stim_loc

    return filename, ind_stim_loc

def make_fp_struct(m,trial,fp_file,rule,epoch,ind_stim_loc,trial_set = range(0,360,36)):

    fps = []
    J_xstar = []
    trial_label = []

    if (rule[:2]=='fd') & (epoch=='delay1'):
        epoch_temp = 'stim1'

        for ti in trial_set:
            filename,_ = get_fp_filename(trial, epoch,ti)
            fp_struct = np.load(os.path.join(m,fp_file,rule,filename+'.npz'))
            fp_num = np.argsort(np.log10(fp_struct['qstar']))

            fps_temp = fp_struct['xstar'][fp_num,:]
            J_xstar_temp = fp_struct['J_xstar'][fp_num,:,:]
            qstar_temp = fp_struct['qstar'][fp_num]

            if len(np.shape(fps_temp))==1:
                fps = fps_temp[np.newaxis,:]
                J_xstar = J_xstar_temp[np.newaxis,:,:]
                qstar = qstar_temp
                trial_label = ti*np.ones(len(fp_num))
            else:
                fps = np.concatenate((fps,fps_temp[np.newaxis,:]),axis = 0)
                J_xstar = np.concatenate((J_xstar,J_xstar_temp[np.newaxis,:,:]),axis = 0)
                qstar = np.concatenate((qstar,qstar_temp),axis = 0)
                trial_label = np.concatenate((trial_label,ti*np.ones(len(fp_num))),axis = 0)

    else:
        filename,_ = get_fp_filename(trial,epoch,ind_stim_loc) #ti = 0
        fp_struct = np.load(os.path.join(m,fp_file,rule,filename+'.npz'))
        fp_num = np.argsort(np.log10(fp_struct['qstar']))

        if len(np.shape(fp_struct['xstar'][fp_num,:]))==1:
            fps = fp_struct['xstar'][fp_num,:][np.newaxis,:]
            J_xstar = fp_struct['J_xstar'][fp_num,:,:][np.newaxis,:,:]
        else:
            fps = fp_struct['xstar'][fp_num,:]
            J_xstar = fp_struct['J_xstar'][fp_num,:,:]
        qstar = fp_struct['qstar'][fp_num]
        
    return fps, J_xstar, qstar

def make_fp_tdr_fig(m,fp_file,rule1,rule2,epoch,ind_stim_loc,tit,trial_set = range(0,360,36),dims = 'tdr'):
    
    nr = 1
    nc = 1
    ms = 10
    
    h,trial,tasks = make_h_trial_rule(m)
    D = get_D(dims,h,trial,[rule1,],epoch,ind = -1)

    fig = plt.figure(figsize=(5.5*nc,4.5*nr),tight_layout=True,facecolor='white')
    cmap=plt.get_cmap('hsv')
    
    for ind_stim_loc in trial_set:
        ind_stim_loc_anti = (ind_stim_loc+180)%360

        if rule1[-4:]=='anti':
            if (rule1 == 'delayanti') & (epoch!='stim1'):
                ind_stim_loc_anti=180
            fps, J_xstar = make_fp_struct(m,fp_file,rule1,epoch,ind_stim_loc_anti,trial_set = trial_set)
        else:
            if (rule1 == 'delaygo') & (epoch!='stim1'):
                ind_stim_loc=0
            fps, J_xstar = make_fp_struct(m,fp_file,rule1,epoch,ind_stim_loc,trial_set = trial_set)

        plt.subplot(nr,nc,1)
        fp_tdr = np.dot(fps,D[rule1].T)
        if (epoch=='delay1') or (epoch=='go1'):
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = 'dodgerblue',markersize = ms)
        else:
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = cmap(ind_stim_loc/360),markersize = ms)

        if rule2[-4:]=='anti':
            if (rule2 == 'delayanti') & (epoch!='stim1'):
                ind_stim_loc_anti=180
            fps, J_xstar = make_fp_struct(m,fp_file,rule2,epoch,ind_stim_loc_anti,trial_set = trial_set)
        else:
            if (rule2 == 'delaygo') & (epoch!='stim1'):
                ind_stim_loc=0
            fps, J_xstar = make_fp_struct(m,fp_file,rule2,epoch,ind_stim_loc,trial_set = trial_set)

        plt.subplot(nr,nc,1)
        fp_tdr = np.dot(fps,D[rule1].T)
        if (epoch=='delay1') or (epoch=='go1'):
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = 'orangered',markersize = ms)
        else:
            print(epoch)
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = cmap(ind_stim_loc/360),markerfacecolor = 'w',markersize = ms)
        
    ax = plt.subplot(nr,nc,1)
    if dims == 'tdr':
        plt.xlabel(rule1 + ' TDR input 1')
        plt.ylabel(rule1 + ' TDR input 1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Fixed Points : ' + tit)
    plt.legend((rule1,rule2))
    return ax

def plot_epoch_dynamics(m,fp_file,epoch,h,trial,rule,D_use,
                        y_set = range(0,360,36),h_epoch = [],plot_eigenspect = True,lim=4, ax_type = 'tdr',
                        epoch_axes =[],stim_loc_fp = 0,q_thresh = -7, fig_scale = 6, al = .5, print_title = True):


    xs = np.linspace(-1, 1, 1000)
    ys = np.sqrt(1 - xs**2)
    
    epoch_name, rule_name, epoch_axes_name, h_epoch = take_names(epoch,rule,epoch_axes = epoch_axes,
                                                                 h_epoch = h_epoch)
    if plot_eigenspect:
        nc = 2
    else:
        nc = 1
    nr = 1    
    
    
    fig = plt.figure(figsize=(fig_scale*nc,fig_scale*nr),tight_layout=True,facecolor='white')
    cmap = plt.get_cmap('hsv')
    
    stim1_locs = np.min(trial[rule].stim_locs[:,[0,2]],axis=1)
    y_locs = trial[rule].y_loc[-1,:]
    
    ax = plt.subplot(nr,nc,1)
    T_inds = get_T_inds(trial[rule],h_epoch)
    h_tdr = np.empty((len(T_inds),np.shape(h[rule])[1]))
    for t in range(0,np.shape(h[rule])[1],2):
        h_tdr_temp = np.dot(h[rule][T_inds,t,:],D_use)
#         print(stim1_locs[t])
        if stim1_locs[t]==stim_loc_fp:
            plt.plot(h_tdr_temp[:,0],h_tdr_temp[:,1],c = cmap(stim1_locs[t]/(2*np.pi)),alpha = 1,linewidth = 3)
            plt.plot(h_tdr_temp[0,0],h_tdr_temp[0,1],'x',c = cmap(stim1_locs[t]/(2*np.pi)),alpha = 1,
                     markersize = 10,linewidth = 2)
        else:
            plt.plot(h_tdr_temp[:,0],h_tdr_temp[:,1],c = cmap(stim1_locs[t]/(2*np.pi)),alpha = al,linewidth = 2)
            plt.plot(h_tdr_temp[0,0],h_tdr_temp[0,1],'x',c = cmap(stim1_locs[t]/(2*np.pi)),alpha = al,
                     markersize = 10,linewidth = 2)
        
    lim = lim
    ax = add_ax_labels(ax,ax_type,lim,epoch_axes_name,rule_name)
        
    if print_title == True:
        plt.title(r"$\bf{" + rule_name + "}$"+ '\n '+epoch_name+' dynamics',y = .9)

    if ax_type!='mix':
        ax.set_aspect('equal')
    
    for ind_stim_loc_anti in y_set:
        fps_anti, J_xstar, q = make_fp_struct(m,trial[rule],fp_file,rule,epoch,ind_stim_loc_anti)

        #remove pts that aren't that slow
        fps_anti = fps_anti[np.log10(q)<q_thresh,:]
        J_xstar = J_xstar[np.log10(q)<q_thresh,:,:] 

        eig_decomps = comp_eig_decomp(J_xstar)
        fps_tdr_anti = np.dot(fps_anti,D_use)
        fp_c = cmap(ind_stim_loc_anti/(360))
        plt.plot(fps_tdr_anti[:,0],fps_tdr_anti[:,1],'o',c = 'k',alpha = .5,markersize=10)
        # plot_FP(fps_anti, D_use.T, eig_decomps, c='k')

        if plot_eigenspect:
            ax2 = fig.add_axes([.45, .45, .2, .2])
#             ax = plt.subplot(2*nr,2*nc,7)
            for fp_num in range(np.shape(J_xstar)[0]):
                evals, _ = LA.eig(J_xstar[fp_num,:,:]) 
                ax2.plot(evals.real,evals.imag,'.k',alpha = .3,markerfacecolor = 'k')

            ax2.plot(xs, ys,':k',linewidth = 1)
            ax2.plot(xs, -ys,':k',linewidth = 1)
            # plt.xlim((.7,1.1))
            # plt.ylim((-.15,.15))
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            eigenspectrum_axes(epoch,ax2)
            ax2.set_aspect('equal')   
            
    return ax

def plot_epoch_dynamics_3D(m,fp_file,epoch,h,trial,rule,D_use,
                        y_set = range(0,360,36),plot_eigenspect = True,lim=4,epoch_axes =[],h_epoch = [],
                        stim_loc_fp = 0,lil_axes = True,plot_unstable = True, ax_labels = [],nr = 1,
                        nc = 1,al = .5,markersize = 100,fp_size = 100,lw = 5,cmap = 'hsv',q_thresh = -10,
                        fig_scale = 6,plot_title=False, color_by_q = False):
    

    '''
    Plotting trajectories and FPs in 3D space
    m: model directory
    fp_file: where to find fps
    epoch: which epoch to plot
    h: use make_h_trial_rule, makes directoory of h for each rule
    D_use: dimensions to plot it
    y_set: which trials to plot fps for
    plot_eigenspect = whether to plot
    lim= xy 
    epoch_axes =[],
    stim_loc_fp = 0
    lil_axes = whether to make lil axes labels
    plot_unstable = plot unstable fps? true/false
    ax_labels = list of strings,
    nr/nc = num rows and columns of subplots
    al = transparency of lines
    markersize = 100,
    fp_size = 300,
    lw = linewidth,
    cmap = 'hsv',
    q_thresh = -10
    '''

    xs = np.linspace(-1, 1, 1000)
    ys = np.sqrt(1 - xs**2)
    
    epoch_name, rule_name, epoch_axes_name, h_epoch = take_names(epoch,rule,epoch_axes = epoch_axes,
                                                                 h_epoch = h_epoch)
    
    fig = plt.figure(figsize=(fig_scale*nc,fig_scale*nr),tight_layout=True,facecolor='white')
    ax = fig.add_axes([0,0,1,1], projection='3d');
    cmap = plt.get_cmap(cmap)
    
    T_inds = get_T_inds(trial[rule],h_epoch)
    if T_inds[0]>0:
        T_inds = range(T_inds[0]-1,T_inds[-1]) #start from the timestep before stim onset etc
        
    stim1_locs = np.min(trial[rule].stim_locs[:,[0,2]],axis=1) #only works for single modality tasks
    y_locs = trial[rule].y_loc[-1,:]

    for t in range(y_set[0],np.shape(h[rule])[1],int(np.shape(h[rule])[1]/10)):
        h_D = np.dot(h[rule][T_inds,t,:],D_use)

        if epoch == 'fix1':
            c = cmap(stim_loc_fp/(2*np.pi))
            al_use = 1
        elif (stim1_locs[t]==stim_loc_fp):
            al_use = 1
            c = cmap(stim1_locs[t]/(2*np.pi))
        else:
            al_use = al/3
            c = cmap(stim1_locs[t]/(2*np.pi))

        ax.plot3D(h_D[:,0],h_D[:,1],h_D[:,2],
                  c = c,
                  alpha = al_use,linewidth = lw)

        ax.scatter(h_D[0,0],h_D[0,1],h_D[0,2],
                   s = markersize*1.5,
                   marker = 'x',
                   c = c,
                   alpha = al_use,
                   linewidth = lw)

        ax.scatter(h_D[-1,0],h_D[-1,1],h_D[-1,2],
                   s = markersize*1.5,
                   marker = '^',
                   c = c,
                   alpha = al_use,
                   linewidth = lw)
        
        
        if stim1_locs[t]==stim_loc_fp or epoch=='fix1':
        
            ax.plot3D(h_D[:,0],h_D[:,1],h_D[:,2],
                      c = 'w',
                      alpha = 1,
                      linewidth = lw/2)
            
            ax.scatter(h_D[0,0],h_D[0,1],h_D[0,2],
                       s = markersize,
                       marker = 'x',
                       c = 'w',
                       alpha = 1,
                       linewidth = lw/2)
                       # edgecolors = cmap(stim1_locs[t]/(2*np.pi)))
                       # markeredgewidth=3)
            
            ax.scatter(h_D[-1,0],h_D[-1,1],h_D[-1,2],
                       s = markersize,
                       marker = '^',
                       c = 'w',
                       alpha = 1,
                       linewidth = lw/2)
                       # edgecolors = cmap(stim1_locs[t]/(2*np.pi)))
                       # markeredgewidth=3)
        
    lim = lim
    ax.set_zlim([-1.2,1.2])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    
    if lil_axes ==True:
        make_lil_axes(ax,ax_labels,fontsize = 20,fac_len = 2)
    
    if plot_title:
        plt.title('Single Task : '+r"$\bf{" + rule_name + "}$"+ '\n '+epoch_name+' dynamics',y = .9,fontsize = 16)

    for ind_stim_loc_anti in y_set:
        fps_anti, J_xstar, q_star = make_fp_struct(m,trial[rule],fp_file,rule,epoch,ind_stim_loc_anti)
        
        #remove pts that aren't that slow
        fps_anti = fps_anti[np.log10(q_star)<q_thresh,:]
        J_xstar = J_xstar[np.log10(q_star)<q_thresh,:,:]
        
        eig_decomps = comp_eig_decomp(J_xstar)
        fps_tdr_anti = np.dot(fps_anti,D_use)
        fp_c = cmap(ind_stim_loc_anti/(360))
        
        for s in range(len(eig_decomps)):
            
            X_trial = np.dot(fps_anti[s,:],D_use)
            
            stability_metric = 1.4 - np.max(eig_decomps[s]['evals'].real)
            
            if color_by_q:
                qcmap=plt.get_cmap('viridis')
                q_star_metric = np.log10(q_star) - np.max(np.log10(q_star))
                color_q = 1-q_star_metric[s]/np.min(q_star_metric)
                ax.scatter(X_trial[0], X_trial[1], X_trial[2], s = fp_size,marker = 'o',
                           edgecolors = 'w',facecolors = qcmap(color_q),alpha = stability_metric)
            
            else:
                ax.scatter(X_trial[0], X_trial[1],X_trial[2], s = fp_size,marker = 'o',
                           edgecolors = 'w',facecolors = 'k',alpha = stability_metric)
            
#             if np.max(eig_decomps[s]['evals'].real)<1:
#                 ax.scatter(X_trial[0], X_trial[1],X_trial[2],s = fp_size,marker = 'o',
#                            edgecolors = 'w',facecolors = 'k',alpha = .9)
#             elif plot_unstable:
#                 ax.scatter(X_trial[0], X_trial[1],X_trial[2],s = 100,marker = 'o',
#                            edgecolors = 'k',facecolors = 'w',alpha = .5)

        if plot_eigenspect:
            ax2 = fig.add_axes([.8, .3, .2, .2])
#             ax = plt.subplot(2*nr,2*nc,7)
            for fp_num in range(np.shape(J_xstar)[0]):
                evals, _ = LA.eig(J_xstar[fp_num,:,:]) 
                ax2.plot(evals.real,evals.imag,'.k',alpha = al,markerfacecolor = 'k')

            ax2.plot(xs, ys,':k',linewidth = 1)
            ax2.plot(xs, -ys,':k',linewidth = 1)
            plt.xlim((.5,1.3))
            plt.ylim((-.35,.35))
            plt.xticks(fontsize = 16)
            plt.yticks(fontsize = 16)
            eigenspectrum_axes(epoch,ax2)
            ax2.set_aspect('equal')   
            
    return ax

def remove_ax_lines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def make_FP_axs(f,m_all,rule_in,fp_epoch,n_fps = 5,axs ='pca_fp', clust = 'False',n_components = 3):
        
    if axs == 'tdr':
        D_use = make_axes(m_all,rule_in,fp_epoch)
    elif axs =='pca_fp':

        fp_struct = np.load(f)
        sorted_fps = fp_struct['xstar']

        if (clust == 'True') & (len(sorted_fps)>2):
            kmeans = KMeans(n_clusters=np.min((n_fps,np.shape(fp_struct['xstar'])[0])), random_state=0).fit(sorted_fps)
            _,fp_inds = np.unique(kmeans.labels_,return_index=True)
        
        else:
            fp_inds = range(len(sorted_fps))

        pca = PCA(n_components = n_components)
        fp_pca = pca.fit_transform(sorted_fps[fp_inds,:])
        D_use = pca.components_.T

    elif axs =='pca_h':
        trial = gen_trials_from_model_dir(m_all,rule_in,mode='random',noise_on = False)
        X, _ = gen_X_from_model_dir_epoch(m_all,trial,fp_epoch)
        pca = PCA(n_components = n_components)
        fp_pca = pca.fit_transform(X.T)
        D_use = pca.components_.T

    elif axs =='out':
        w_in, b_in, w_out, b_out = get_model_params(m_all)
        D_use = w_out[:,1:]

    elif axs =='pca_h_cat':

        trial = gen_trials_from_model_dir(m_all,rule_in[0],mode='random',noise_on = False)
        X1, _ = gen_X_from_model_dir_epoch(m_all,trial,fp_epoch[0])

        trial = gen_trials_from_model_dir(m_all,rule_in[1],mode='random',noise_on = False)
        X2, _ = gen_X_from_model_dir_epoch(m_all,trial,fp_epoch[1])
        X = np.concatenate((X1,X2),axis = 1)

        pca = PCA(n_components = n_components)
        fp_pca = pca.fit_transform(X.T)
        D_use = pca.components_.T

    return D_use
    
def add_ax_labels(ax,ax_type,lim,epoch_axes_name,rule_name):
    if ax_type == 'out':
        (ax)
    elif ax_type == 'pca':
        PC_axes(ax)
        plt.ylim((-lim,lim))
        plt.xlim((-lim,lim))
    elif ax_type == 'tdr':
        TDR_axes(epoch_axes_name,ax,rule_name)
        plt.ylim((-lim,lim))
        plt.xlim((-lim,lim))
    elif ax_type == 'mix':
        plt.ylim((-1.5,1.5))
        plt.xlim((-lim,lim))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
#         plt.xlabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+ epoch_axes_name +r' $\cos{\theta}$',fontsize = 20)
        plt.xlabel('TDR : '+r' $\sin{\theta}$',fontsize = 20)
        plt.ylabel('Output ' + r' $\sin{\theta}$',fontsize = 20)
    return (ax)

def TDR_axes(epoch,ax,rule_name):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
#     plt.xlabel('TDR : '+epoch+ r' $\cos{\theta}$')
#     plt.ylabel('TDR : ' +epoch+ r' $\sin{\theta}$')
    plt.xlabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+r' $\cos{\theta}$',fontsize = 20) #+r"$\bf{" + rule_name + "}$"+ ' '
    plt.ylabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+r' $\sin{\theta}$',fontsize = 20) #+r"$\bf{" + rule_name + "}$"+ ' '
    
def eigenspectrum_axes(epoch,ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Real Part',fontsize = 20)
    plt.ylabel('Imaginary Part',fontsize = 20,labelpad=1)
    
def out_axes(ax):

    plt.ylim((-1.5,1.5))    
    plt.xlim((-1.5,1.5))  
    
    plt.xlabel('Output ' + r' $\cos{\theta}$',fontsize = 20)
    plt.ylabel('Output ' + r' $\sin{\theta}$',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    
def PC_axes(ax):

    ax.set_xlabel('PC1',fontsize = 20)
    ax.set_ylabel('PC2',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def FP_PC_axes(ax):

    ax.set_xlabel('FP PC1',fontsize = 20)
    ax.set_ylabel('FP PC2',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 

def get_color_inds():
    tab20 = plt.get_cmap('tab20')
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')
    color_inds = (tab20c(8/20),tab20c(9/20),tab20c(10/20),tab20c(4/20),tab20c(5/20),tab20c(6/20),
                 tab20b(0/20),tab20b(12/20),tab20b(1/20),tab20b(13/20),tab20b(16/20),
                 tab20b(2/20),tab20b(14/20),tab20b(3/20),tab20b(15/20),tab20b(17/20),
                 tab20(18/20),tab20(19/20),tab20(16/20),tab20(17/20))


    color_inds_nu = (tab20c(8/20),tab20c(9/20),tab20b(0/20),tab20c(4/20),tab20c(5/20),tab20b(12/20),
                 tab20b(0/20),tab20b(12/20),tab20b(1/20),tab20b(13/20),tab20b(16/20),
                 tab20b(2/20),tab20b(14/20),tab20b(3/20),tab20b(15/20),tab20b(17/20),
                 tab20(18/20),tab20(19/20),tab20(16/20),tab20(17/20))

    return color_inds, color_inds_nu

def plot_FP_jitter(m,D_use,rule_master,t_num,fp_epoch,sorted_fps,fp_inds,eig_decomps,rule_set,
                   xlabel = 'FP set PC1',ylabel = 'FP set PC2',rand_step_coef = 0.1,n_steps = 100,
                   lw = 3,al = .6,linestyle = '-',n_jit = 0,c_master = 'k',lesion_units = []):

    cmap=plt.get_cmap('rainbow')
        
    model = FixedPoint_Model(m)
    with tf.Session() as sess:
        model.restore()
        if len(lesion_units)>0:
            model.lesion_units(sess, lesion_units)
        model._sigma=0
        hparams = model.hp
        alpha = hparams['dt']/hparams['tau']
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]

        for rule in rule_set:
            trial = generate_trials(rule, hparams, mode='test',noise_on=False)
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf, _ = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
            T,S,N = np.shape(h_tf)
            T_inds = get_T_inds(trial,fp_epoch) # grab epoch time indexing

            for s in range(0,S,int(S/8)):

                if c_master == 'y_locs':
                    c_inds = trial.y_loc[-1,:]
                    c = cmap(c_inds[s]/(2*np.pi))
                elif c_master == 'x_locs':
                    stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
                    c = cmap(stim1_locs[s]/(2*np.pi))
                else:
                    c = c_master

                X_trial = np.dot(h_tf[T_inds,s,:],D_use)
                plt.plot(X_trial[0,0],X_trial[0,1],'x',c = c, alpha = al, linewidth = lw)
                plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = c, alpha = al, linewidth = lw)
                plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, alpha = al, linewidth = lw)

        trial = generate_trials(rule_master, hparams, mode='test',noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        T_inds = get_T_inds(trial,fp_epoch)

        if c_master == 'y_locs':
            c_inds = trial.y_loc[-1,:]
            c = cmap(c_inds[t_num]/(2*np.pi))
        elif c_master == 'x_locs':
            stim1_locs = np.min(trial.stim_locs,axis=1)
            c = cmap(stim1_locs[t_num]/(2*np.pi))
        else:
            c = c_master

        X_trial = np.dot(h_tf[T_inds,t_num,:],D_use)
        plt.plot(X_trial[0,0],X_trial[0,1],'x',c = c, alpha = .8, linewidth = 5)
        plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = c, alpha = .8, linewidth = 5)
        plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, alpha = .8, linewidth = 5, markersize = 10)

        for fp_ind in fp_inds:

            if np.max(eig_decomps[fp_ind]['evals'])>.99:
                markerfacecolor = 'None'

            else:
                markerfacecolor = c

            x_t = np.matlib.repmat(trial.x[T_inds[0],t_num,:],n_steps,1)
            for jit in range(n_jit):
                h0 = sorted_fps[fp_ind,:] + rand_step_coef*npr.randn(N)
                h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
                jitter = np.dot(h_t,D_use)
                plt.plot(jitter[-1,0],jitter[-1,1],'^k',linewidth = .1,alpha = .1)
                plt.plot(jitter[:,0],jitter[:,1],'-k',linewidth = .1)

            fp = np.dot(sorted_fps[fp_ind,:],D_use)
            # print(np.shape(fp))
            plt.plot(fp[0],fp[1],'o',c = c,linewidth = 5,markersize = 6,markerfacecolor = markerfacecolor)

            h_t = vanilla_run_with_h0(params, x_t, h_tf[T_inds[0],t_num,:], hparams)
            jitter = np.dot(h_t,D_use)
            plt.plot(jitter[-1,0],jitter[-1,1],'^k',c = c)
            plt.plot(jitter[:,0],jitter[:,1],'-k',c = c,linewidth = 1)
            plt.plot(jitter[0,0],jitter[0,1],'xk',c = c,linewidth = 5,markersize = 10)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(rule_master)

def out_affine(params, h):
    return np.dot(params[2].T,h)+params[3]

def relu(x):
    f = x * (x > 0)
    return f

def rnn_vanilla(params, h, x, alpha, activation):

    if activation == 'softplus':
        _activation = lambda x: np.log(np.exp(x) + 1)
    elif activation == 'tanh':
        _activation = lambda x: np.tanh(x)
    elif activation == 'relu':
        _activation = lambda x: x * (x > 0)
    elif activation == 'power':
        _activation = lambda x: (x * (x > 0))**2
    elif activation == 'retanh':
        _activation = lambda x: np.tanh(x * (x > 0))
    
    xh = np.concatenate([x,h], axis=0)
    gate_inputs = np.dot(params[0].T,xh)+params[1]
    noise = 0
    output = _activation(gate_inputs) # + noise

    h_new = (1-alpha) * h + alpha * output
    
    return h_new

def vanilla_run_with_h0(params, x_t, h0, hparams):

    dt = hparams['dt']
    tau = hparams['tau']
    alpha = dt/tau
    activation = hparams['activation']

    h = h0
    h_t = []
    h_t.append(np.expand_dims(h0,axis=1))
    for x in x_t:
        h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T), alpha, activation)
        h_t.append(np.expand_dims(h,axis=1))

    h_t = np.squeeze(np.array(h_t))  
    return h_t

def vanilla_run_at_fp(params, x_t, h0, hparams):

    dt = hparams['dt']
    tau = hparams['tau']
    alpha = dt/tau
    activation = hparams['activation']

    h = h0
    h_t = []
    h_t.append(h)
    for x in x_t:
        h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T), alpha, activation)
        h_t.append(np.expand_dims(h,axis=1))
        
    return h_t

def make_dst_mat(X):
    X = np.squeeze(X).astype(np.float64)
    dst = np.zeros((np.shape(X)[0],np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
        for yi in range(np.shape(X)[0]):
            dst[xi,yi] = distance.euclidean(X[xi,:], X[yi,:])
    return dst

def make_angle_mat(X):
    X = np.squeeze(X).astype(np.float64)

    theta_mat = np.zeros((np.shape(X)[0],np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
        v1 = X[xi,:]
        for yi in range(np.shape(X)[0]):
            v2 = X[yi,:]
            theta_mat[xi,yi] = angle_between(v1, v2)

    return theta_mat

    

def make_MDS_dst(h_out):
    X = np.squeeze(h_out).astype(np.float64)
    dst = np.zeros((np.shape(X)[0],np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
        for yi in range(np.shape(X)[0]):
            dst[xi,yi] = distance.euclidean(X[xi,:], X[yi,:])
            
    embedding = MDS(n_components = 2,dissimilarity = 'precomputed')
    X_out = embedding.fit_transform(dst)
    return X_out,dst
    
def comp_eig_decomp(Ms, sort_by='real',do_compute_lefts=True):
  """Compute the eigenvalues of the matrix M. No assumptions are made on M.

  Arguments: 
    M: 3D np.array nmatrices x dim x dim matrix
    do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse 
      call.

  Returns: 
    list of dictionaries with eigenvalues components: sorted 
      eigenvalues, sorted right eigenvectors, and sored left eigenvectors 
      (as column vectors).
  """
  if sort_by == 'magnitude':
    sort_fun = np.abs
  elif sort_by == 'real':
    sort_fun = np.real
  else:
    assert False, "Not implemented yet."      
  
  decomps = []
  L = None  
  for M in Ms:
    evals, R = LA.eig(M)    
    indices = np.flipud(np.argsort(sort_fun(evals)))
    if do_compute_lefts:
      L = LA.pinv(R).T  # as columns      
      L = L[:, indices]
    decomps.append({'evals' : evals[indices], 'R' : R[:, indices],  'L' : L})
  
  return decomps

def take_names(epoch,rule,epoch_axes = [],h_epoch = []):
    epochs = ['fix1','stim1','delay1','stim2','delay2','go1']
    # epoch_names = ['Cntxt','Stim1','Memory1','Stim2','Mmry2','Respns']
    epoch_names = ['Context','Stimulus','Memory','Stimulus2','Memory','Response']
    # epoch_names = ['Context','Stim1','Memory1','Stim2','Memory2','Response']

    ei = [i for i,e in enumerate(epochs) if e==epoch]
    epoch_name = epoch_names[ei[0]]

    ri = [i for i,e in enumerate(rules_dict['all']) if e==rule]
    rule_name = rule_set_names[ri[0]]
    
    if len(epoch_axes)<1:
        epoch_axes_name = epoch_names[ei[0]]
    else:
        ei = [i for i,e in enumerate(epochs) if e==epoch_axes]
        epoch_axes_name = epoch_names[ei[0]]
    
    if len(h_epoch)==0:
        h_epoch = epoch
    
    return epoch_name, rule_name, epoch_axes_name, h_epoch

import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

def find_opt_clust_num(D,Y):
    # Choose number of clusters that maximize silhouette score
    n_clusters = range(3, np.min([len(Y),40]))
    scores = list()
    labels_list = list()
    for n_cluster in n_clusters:
        clusters = fcluster(Y, n_cluster, criterion='maxclust')

        score = metrics.silhouette_score(D, clusters)

        scores.append(score)
        labels_list.append(clusters)

    scores = np.array(scores)
    max_d = np.argmax(scores)
    opt_clust = n_clusters[max_d]
    return opt_clust

def make_dendro(m,method = 'ward',criterion = 'distance',cel_max_d = 0,max_d = 0, normalization_method='max'):

    CA = clustering.Analysis(m, data_type='epoch', normalization_method=normalization_method)
    tick_names = [rule_name[key[0]]+' '+key[1] for key in CA.keys]

    # Generate features and distance matrix.
    D  = CA.h_normvar_all.T

    # Compute and plot dendrogram.
    fig = pylab.figure(figsize=(24, 15))
    axdendro = fig.add_axes([0.09,0.1,0.05,0.75])

    Y = sch.linkage(D, method=method)

    if max_d==0:
        max_d = find_opt_clust_num(D,Y)
        clusters = fcluster(Y, max_d, criterion='maxclust')
    else:
        clusters = fcluster(Y, max_d, criterion=criterion)
    Z = sch.dendrogram(Y, orientation='left',labels = tick_names,
                       leaf_font_size = 11,color_threshold=max_d)

    axdendro.set_xticks([])
    axdendro.spines['top'].set_visible(False)
    axdendro.spines['right'].set_visible(False)
    axdendro.spines['bottom'].set_visible(False)
    axdendro.spines['left'].set_visible(False)

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.22,0.1,0.75,0.75])
    index_left = Z['leaves']
    tick_names_sorted = [tick_names[i] for i in index_left]
    D = D[index_left,:]

    # cel_num = [CA.ind_active[x] for x in index_top]
    axdendro_top = fig.add_axes([0.22,.9,0.75,0.1])
    Y = sch.linkage(D.T, method=method)
    if cel_max_d==0:
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_opt_clust'
        cel_max_d = find_opt_clust_num(D.T,Y)
        clusters = fcluster(Y, cel_max_d, criterion='maxclust')
    else:
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_max_d'+str(cel_max_d)
        clusters = fcluster(Y, cel_max_d, criterion=criterion)
    Z = sch.dendrogram(Y, orientation='top',labels = clusters, #CA.ind_active #clusters
                   leaf_font_size = 11,color_threshold=cel_max_d)

    # axdendro_top.set_xticks([])
    axdendro_top.set_yticks([])

    index_top = Z['leaves']
    D = D[:,index_top]
    clusters_sorted = clusters[index_top]
    im = axmatrix.matshow(D, aspect='auto', origin='lower',cmap='magma')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.975,0.1,0.01,0.8])
    pylab.colorbar(im, cax=axcolor)

    lesion_units_list = [None]
    for il, l in enumerate(np.unique(clusters)):
        ind_l = np.where(clusters == l)[0]
        # In original indices
        lesion_units_list += [CA.ind_active[ind_l]]

    # save cluster variables
    cluster_var = {'D':D,'Z':Z,
                'index_top':index_top,
                'index_left':index_left,
                'tick_names':tick_names_sorted,
                'clusters':clusters_sorted,
                'lesion_units_list':lesion_units_list,
                'max_d':cel_max_d,
                'criterion':criterion,
                'method':method}

    save_dir = os.path.join(m,lesion_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(os.path.join(save_dir,'cluster_var.npz'),**cluster_var)
    plt.savefig(os.path.join(save_dir,'dynamic_modules_atlas'+'.pdf'))
    plt.savefig(os.path.join(save_dir,'dynamic_modules_atlas'+'.png'))

def plot_stability(m,rule = 'delaygo', epoch = 'stim1', trial_n = 0, trained = True):

    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        hp = model.hp
        trial = generate_trials(rule, hp, mode='random', noise_on=False, batch_size =100)
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

    fig = plt.figure(figsize=(17, 3))

    # Plot during task
    ax1 = plt.subplot(1,3,1)
    plt.plot(h_tf[:,0,:50],alpha = .5)
    plt.xlabel('time in trial',fontsize = 18)
    plt.ylabel('activations',fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    #Plot evolution of hidden state for extended time
    n_steps = 500
    x_t = np.matlib.repmat(trial.x[-1,trial_n,:],n_steps,1)
    h0 = h_tf[-1,trial_n,:]
    h_t = vanilla_run_with_h0(params, x_t, h0, hp)

    ax1 = plt.subplot(1,3,2)
    ax1.plot(h_t[:,:50],alpha = .5)
    plt.xlabel('timesteps',fontsize = 18)
    plt.ylabel('activations',fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    #Plot eigenspectrum
    inputs = x_t[:1,:]
    state = h_tf[-1:,trial_n,:]

    dFdx = calc_jacobian_at(m,inputs,state,lesion_units = [])
    evals, _ = LA.eig(dFdx[0])

    xs = np.linspace(-1, 1, 1000)
    ys = np.sqrt(1 - xs**2)

    ax2 = plt.subplot(1,3,3)

    ax2.plot(evals.real,evals.imag,'.k',alpha = .3,markerfacecolor = 'k')
    ax2.plot(xs, ys,':k',linewidth = 1)
    ax2.plot(xs, -ys,':k',linewidth = 1)
    ax2.plot([1,1],[-1,1],'-k',alpha = .3)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    eigenspectrum_axes(epoch,ax2)
    ax2.set_aspect('equal')
    tit = hp['activation'] + '_' + hp['w_rec_init'] + '_alpha_' + str(hp['alpha'])
    plt.title(tit,fontsize = 18)

    save_dir = os.path.join(m,'stability_figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir,'eigenspec.png'))

from network import get_perf
from task import generate_trials
        
def lesions(m,rules=[],method = 'ward',max_d = 0,criterion = 'distance',other = False, mode = 'random'):


    if max_d==0:
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_opt_clust'
        lesion_file = 'cluster_var.npz'
        save_f = 'lesion_var'
    else:
        lesion_folder = 'lesion_dendro'
        lesion_file = 'cluster_dendro_max_d'+str(max_d)+'.npz'
        save_f = 'lesion_var'+str(max_d)

    save_dir = os.path.join(m,lesion_folder)
    cluster_var = np.load(os.path.join(save_dir,lesion_file))
    lesion_units_list = cluster_var['lesion_units_list']

    perfs_store_list = list()
    perfs_changes = list()
    cost_store_list = list()
    cost_changes = list()

    for i, lesion_units in enumerate(lesion_units_list):
        print(lesion_units)
        model = Model(m)
        hp = model.hp
        if len(rules)==0:
            rules = hp['rule_trains']
        with tf.Session() as sess:
            model.restore()
            if other:

                lesion_units_other = range((n_units))

                for les_uni in lesion_units:
                    lesion_units_other.remove(les_uni)
        
                model.lesion_units(sess, lesion_units_other)
            else:
                model.lesion_units(sess, lesion_units)

            perfs_store = list()
            cost_store = list()
            for rule in rules:

                if mode=='test':
                    n_rep = 1
                else:
                    n_rep = 16

                batch_size_test = 256
                batch_size_test_rep = int(batch_size_test / n_rep)
                clsq_tmp = list()
                perf_tmp = list()

                for i_rep in range(n_rep):

                    trial = generate_trials(rule, hp, mode,
                                            batch_size=batch_size_test_rep)

                    feed_dict = tools.gen_feed_dict(model, trial, hp)
                    y_hat_test, c_lsq = sess.run(
                        [model.y_hat, model.cost_lsq], feed_dict=feed_dict)

                    # Cost is first summed over time, and averaged across batch and units
                    # We did the averaging over time through c_mask

                    # IMPORTANT CHANGES: take overall mean
                    perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                    clsq_tmp.append(c_lsq)
                    perf_tmp.append(perf_test)

                perfs_store.append(np.mean(perf_tmp))
                cost_store.append(np.mean(clsq_tmp))

        perfs_store = np.array(perfs_store)
        cost_store = np.array(cost_store)

        perfs_store_list.append(perfs_store)
        cost_store_list.append(cost_store)

        if i > 0:
            perfs_changes.append(perfs_store - perfs_store_list[0])
            cost_changes.append(cost_store - cost_store_list[0])

    perfs_changes = np.array(perfs_changes)
    cost_changes = np.array(cost_changes)
    
    # save cluster variables
    lesion_var = {'perfs_changes':perfs_changes,
                'cost_changes':cost_changes}

    save_dir = os.path.join(m,lesion_folder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if mode=='test':
        np.savez(os.path.join(save_dir,save_f+'_test.npz'),**lesion_var)
    elif other==True:
        np.savez(os.path.join(save_dir,save_f+'_other.npz'),**lesion_var)
    else:
        np.savez(os.path.join(save_dir,save_f+'.npz'),**lesion_var)

    return perfs_changes, cost_changes

def get_filename(trial, epoch,t):
    ind_stim_loc  = str(int(180*trial.y_loc[-1,t]/np.pi))
    nonzero_stim = trial.stim_locs[t,:]<100
    stim_names = '_'.join(str(int(180*x/np.pi)) for x in trial.stim_locs[t,nonzero_stim])
    filename = epoch+'_trial'+str(t)+'_x'+stim_names+'_y'+ind_stim_loc

    return filename, ind_stim_loc

def make_D_use(m,rule,epoch,ti,axs,trial_num = [0,0]):
    trial_str = '_'.join(str(t) for t in trial_num)
    trial = gen_trials_from_model_dir(m,rule,noise_on = False)
    filename,_ = get_filename(trial,epoch,ti)
    f = os.path.join(m,'tf_fixed_pts_manytrials',rule,filename+'.npz')
    D_use = make_FP_axs(f, m, rule, epoch, axs = axs, clust = 'False')
    _, ax_rule_name, _, _ = take_names(epoch,rule)
    return D_use

def interp2d(m,D_use,rule_set,epoch_set,t_set,script_name = 'interp_tasks_small_init_mov',
    tol_q = 1e-6,interp = True,cmap_grad = plt.get_cmap('plasma'),cmap_discrete = ['w','k'],
    q_thresh = .001,n_skip_interps=2, rule_axs = 0, plot_unstable = True, color_stim = True,
    figpath = [],fig_size = 4,alpha_max = 1.1,fontsize = 20,title_y = .9,tit_name = [],
    only_rel = False, stability_thresh = 1,fp_first = True, axs = []):
    
    fig = plt.figure(figsize=(fig_size,fig_size),tight_layout=True,facecolor='white')
    plt.rcParams.update({'font.size': fontsize})
    n_interp = 20

    rule1 = rules_dict['all'][rule_set[0]]
    rule2 = rules_dict['all'][rule_set[1]]
    rule_str = [rule1,rule2]

    epoch = epoch_set[rule_axs]
    epoch_name1, rule_name1, _, _ = take_names(epoch_set[0],rule1)
    epoch_name2, rule_name2, _, _ = take_names(epoch_set[1],rule2)
    if epoch_name1==epoch_name2:
        epoch_name = epoch_name1

    if not fp_first:
        for ri in range(2):

            rule = rule_str[ri]
            epoch = epoch_set[ri]
            c_master = cmap_discrete[ri]

            trial = gen_trials_from_model_dir(m,rule,mode='test',noise_on = False)
            trial_set = range(0,np.shape(trial.x)[1],int(np.shape(trial.x)[1]/8))

            _,x = gen_X_from_model_dir(m,trial)
            T_inds = get_T_inds(trial,epoch_set[ri])
            T_inds_plot = range(np.max([T_inds[0]-1,0]),T_inds[-1])
            x_epoch = np.transpose(x[:,:,T_inds],(1,2,0))

            ax = plt.subplot(111)
            stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
            y_locs = trial.y_loc[-1,:]

            if color_stim:
                plot_N( x_epoch[trial_set,:,:],D_use.T,stim1_locs[trial_set],
                       linewidth = fig_size+1,linestyle = '-', alpha = .5, markersize = 20)
            else:
                plot_N(x_epoch[trial_set,:,:],D_use.T,y_locs[trial_set],
                       linewidth = fig_size+1,linestyle = '-', alpha = .5, markersize = 20)
                
            plot_N(x_epoch[trial_set,:,:],D_use.T,[cmap_discrete[ri],],linewidth = fig_size/2, 
                   linestyle =  '-', alpha = .9, markersize = 10)
            

    if interp:
        if script_name == 'interp_tasks_small_init_stim':
            trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
            trial2 = same_stim_trial(trial1, rule_set[1]) 
            trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
        elif script_name == 'interp_tasks_small_init_mov':
            trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
            trial2 = gen_trials_from_model_dir(m,rule2,mode='test',noise_on = False)
            trial2 = same_mov_inds_py3(trial1, trial2) 
            trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)

        save_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)))
        filename_interp = get_interp_filename(trial1,trial2,epoch_set,t_set)

        if only_rel:

            model = Model(m)
            with tf.Session() as sess:
                model.restore()
                model._sigma=0
                var_list = model.var_list
                params = [sess.run(var) for var in var_list]
                hparams = model.hp
                feed_dict = tools.gen_feed_dict(model, trial1, hparams)
                # run model
                h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)

    
        for step_i in range(n_interp):#[9, 10, 8, 11, 7, 12, 6, 13, 5, 14, 4, 15, 3, 16, 2, 17, 1, 18, 19, 0]:
            c = cmap_grad(step_i/n_interp)

            fp_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)),
                                  filename_interp+'_step_'+str(step_i)+'.npz')
            fp_struct = np.load(fp_dir)
            step_fps = fp_struct['xstar']
            fp_inds = np.where(fp_struct['qstar']<q_thresh)[0]  
            J_xstar = fp_struct['J_xstar'][fp_inds,:,:]
            sorted_fps = fp_struct['xstar'][fp_inds,:]
            eig_decomps = comp_eig_decomp(J_xstar)

            if only_rel:

                T_inds = get_T_inds(trial1,epoch_set[0])
                inputs_1 = trial1.x[T_inds,t_set[0],:]
                inputs_2 = trial2.x[T_inds,t_set[1],:]
                del_inputs = inputs_2 - inputs_1
                step_inputs = inputs_1+del_inputs*(step_i/n_interp)

                x_t = step_inputs
                h0 = h_tf[T_inds[0],t_set[0],:]

                h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
                rel_fp, _ = find_closest_fp_loc(h_t[-1,:],fp_struct['xstar'][fp_inds,:])
                fp_set = [rel_fp,]
            else:
                fp_set = range(len(fp_inds))

            for fp_ind in fp_set:

                fp = np.dot(sorted_fps[fp_ind,:],D_use)

                # stability_metric = alpha_max - np.max(eig_decomps[fp_ind]['evals'].real)
                # stability_metric = np.max((stability_metric,0))
                # stability_metric = np.min((stability_metric,1))
                stability_metric = .3
                facecolors_3d = c
                facecolors_2d = c
                edgecolors = 'w'
                al = stability_metric
                
                if np.max(eig_decomps[fp_ind]['evals'].real)>stability_thresh:
                    if plot_unstable:
                        plt.plot(fp[0],fp[1],'o',c = c,linewidth = 1,markersize = 20,markerfacecolor = 'w',
                                 alpha = stability_metric)

                else:

                    plt.plot(fp[0],fp[1],'o',c = edgecolors,linewidth = 1,markersize = 20,markerfacecolor = facecolors_2d,
                                 alpha = stability_metric)
    if fp_first:
        for ri in range(2):

            rule = rule_str[ri]
            epoch = epoch_set[ri]
            c_master = cmap_discrete[ri]

            trial = gen_trials_from_model_dir(m,rule,mode='test',noise_on = False)
            trial_set = range(0,np.shape(trial.x)[1],int(np.shape(trial.x)[1]/8))

            _,x = gen_X_from_model_dir(m,trial)
            T_inds = get_T_inds(trial,epoch_set[ri])
            T_inds_plot = range(np.max([T_inds[0]-1,0]),T_inds[-1])
            x_epoch = np.transpose(x[:,:,T_inds],(1,2,0))

            ax = plt.subplot(111)
            stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
            y_locs = trial.y_loc[-1,:]

            if color_stim:
                plot_N( x_epoch[trial_set,:,:],D_use.T,stim1_locs[trial_set],
                       linewidth = fig_size+1,linestyle = '-', alpha = .5, markersize = 20)
            else:
                plot_N(x_epoch[trial_set,:,:],D_use.T,y_locs[trial_set],
                       linewidth = fig_size+1,linestyle = '-', alpha = .5, markersize = 20)
                
            plot_N(x_epoch[trial_set,:,:],D_use.T,[cmap_discrete[ri],],linewidth = fig_size/2, 
                   linestyle =  '-', alpha = .9, markersize = 10)

    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()
    ax.set_xlim([x1-.2*abs(x1),x2+.2*abs(x2)])
    ax.set_ylim([y1-.2*abs(y1),y2+.2*abs(y2)])
    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()

    if len(tit_name)==0:
        if rule_set[0]==rule_set[1]:
            tit = 'Single Task : '+rule_name1+'\n '+r"$\bf{"+epoch_name1+"}$"+' vs. '+r"$\bf{"+epoch_name2+"}$"+' dynamics' 
            tit_name = rule_name1+'_'+epoch_name1+'_v_'+epoch_name2+'_dynamics_2d'
        elif epoch_name1==epoch_name2:       
            tit = r"$\bf{"+rule_name1+"}$"+' vs. \n'+r"$\bf{"+rule_name2+"}$"+ '\n '+epoch_name+' dynamics'
            tit_name = rule_name1+'_v_'+rule_name2+'_'+epoch_name[:-1]+'_dynamics_2d'   
        else:
            tit_name = rule_name1+'_v_'+rule_name2+'_'+epoch_name1+'_v_'+epoch_name2+'_dynamics_2d'

    else:
        tit = tit_name
        tit_name = rule_name1+'_v_'+rule_name2+'_'+epoch_name1+'_v_'+epoch_name2+'_dynamics_2d' 
    plt.title(tit,y = title_y)

    _, ax_rule_name, _, _ = take_names(epoch,rule_str[rule_axs])
    
    if axs == 'pca_h_cat':
        if epoch_name1==epoch_name2:  
            axes_label1 = epoch_name+' State PC1'
            axes_label2 = epoch_name+' State PC2'
        else:
            axes_label1 = 'PC1'
            axes_label2 = 'PC2'
    else:
        axes_label1 = r"$\bf{" + ax_rule_name + "}$"+' \n'+epoch_name+' State PC1'
        axes_label2 = r"$\bf{" + ax_rule_name + "}$"+' \n'+epoch_name+' State PC2'

    make_lil_axes(ax,[axes_label1,axes_label2],fontsize = fontsize)
    # make_lil_axes(ax,['Modified Memory State PC1','Modified Memory State PC2'],fontsize = fontsize)

    fig_fldr = 'interp_fps'
    if not os.path.exists(os.path.join(figpath,fig_fldr,script_name)):
        os.makedirs(os.path.join(figpath,fig_fldr,script_name))

    plt.savefig(os.path.join(figpath,fig_fldr,script_name,tit_name+'.pdf'),bbox_inches='tight')
    plt.show()

def interp3d(m,D_use,rule_set,epoch_set,t_set,script_name = 'interp_tasks_small_init_mov',
    tol_q = 1e-6,interp = True, cmap_grad = plt.get_cmap('plasma'),cmap_discrete = ['w','k'],
    q_thresh = .001,n_skip_interps=2, rule_axs = 0,ax_labels = [], lil_axes =True, 
    plot_unstable = True,plot_zero_plane = True,color_stim = True,fig_size = 4, fontsize = 20, 
    figpath = [], ruleset = [],alpha_fp = .7,title_y = .9, n_interp = 20,n_trials = 8, 
    stability_thresh = 1,view_ang = [],z_label = []):
    
    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=(fig_size,fig_size),tight_layout=True,facecolor='white')
    ax = fig.add_axes([0,0,1,1], projection='3d');
    al = .5

    rule1 = rules_dict['all'][rule_set[0]]
    rule2 = rules_dict['all'][rule_set[1]]
    rule_str = [rule1,rule2]

    epoch = epoch_set[rule_axs]
    epoch_name_ax, rule_name_ax, _, _ = take_names(epoch,rules_dict['all'][rule_set[rule_axs]])
    epoch_name1, rule_name1, _, _ = take_names(epoch_set[0],rule1)
    epoch_name2, rule_name2, _, _ = take_names(epoch_set[1],rule2)
    if epoch_name1==epoch_name2:
        epoch_name = epoch_name1
    
    trial1 = gen_trials_from_model_dir(m,rule_str[0],mode='test',noise_on = False)
    trial2 = gen_trials_from_model_dir(m,rule_str[1],mode='test',noise_on = False)
    trial2 = same_mov_inds_py3(trial1, trial2) 
    trial1 = gen_trials_from_model_dir(m,rule_str[0],mode='test',noise_on = False)
    save_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)))
    filename_interp = get_interp_filename(trial1,trial2,epoch_set,t_set)


    if interp:
        for step_i in range(n_interp-1,0,-n_skip_interps):
            c = cmap_grad(step_i/n_interp)

            fp_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)),
                                  filename_interp+'_step_'+str(step_i)+'.npz')
            fp_struct = np.load(fp_dir)
            step_fps = fp_struct['xstar']
            fp_inds = np.where(fp_struct['qstar']<q_thresh)[0]  
            J_xstar = fp_struct['J_xstar'][fp_inds,:,:]
            sorted_fps = fp_struct['xstar'][fp_inds,:]
            eig_decomps = comp_eig_decomp(J_xstar)

            for fp_ind in range(len(fp_inds)):

                
                fp = np.dot(sorted_fps[fp_ind,:],D_use)
                
                # stability_metric = alpha_max - np.max(eig_decomps[fp_ind]['evals'].real)
                # stability_metric = np.max((stability_metric,0))
                # stability_metric = np.min((stability_metric,1))
                stability_metric = alpha_fp
                facecolors_3d = c
                facecolors_2d = c
                edgecolors = c#'w'
                al = stability_metric
                
                if np.max(eig_decomps[fp_ind]['evals'])>stability_thresh:
                    if plot_unstable:
                        facecolors_3d = 'w'
                        ax.scatter(fp[0],fp[1],fp[2],s = 100,marker = 'o',edgecolors = edgecolors,
                           facecolors = facecolors_3d,alpha = al)
                else:   
                    s = 100
                    ax.scatter(fp[0],fp[1],fp[2],s = s,marker = 'o',edgecolors = edgecolors,
                           facecolors = facecolors_3d,alpha = al)

    for ri in range(2):

        rule = rule_str[ri]
        epoch = epoch_set[ri]
        c_master = cmap_discrete[ri]

        trial = gen_trials_from_model_dir(m,rule,noise_on = False)
        
        if n_trials==0:
            trial_set = []
        else:
            trial_set = range(0,np.shape(trial.x)[1],int(np.shape(trial.x)[1]/n_trials))

        _,x = gen_X_from_model_dir(m,trial)
        T_inds = get_T_inds(trial,epoch_set[ri])
        x_epoch = np.transpose(x[:,:,T_inds],(1,2,0))
        
        if color_stim:
            c_set = np.min(trial.stim_locs[:,[0,2]],axis=1)
            s = np.argwhere(c_set == np.min(trial.stim_locs[t_set[0],[0,2]]))[0][0]
            c = cmap_grad(np.min(trial.stim_locs[t_set[0],[0,2]])/max(np.min(trial.stim_locs[:,[0,2]],axis=1)))
        else:
            c_set = trial.y_loc[-1,:]
            s = np.argwhere(c_set==trial.y_loc[-1,t_set[0]])[0][0]
            c = cmap_grad(trial.y_loc[-1,t_set[0]]/max(trial.y_loc[-1,:]))

        if len(trial_set)>0:
            plot_N3D(ax,x_epoch[trial_set,:,:],D_use.T,c_set[trial_set],
               linewidth = 7,linestyle = '-', alpha = .3, markersize = 150, edgecolors = cmap_discrete[ri])
            plot_N3D(ax,x_epoch[[s,],:,:],D_use.T,c_set[trial_set],
               linewidth = 7,linestyle = '-', alpha = 1, markersize = 150, edgecolors = cmap_discrete[ri])
    
    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()
    ax.set_xlim([x1-.1*abs(x1),x2+.1*abs(x2)])
    ax.set_ylim([y1-.1*abs(y1),y2+.1*abs(y2)])
    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()


    if ax_labels=='pca_h':
        ax.set_xlabel(r"$\bf{"+rule_name_ax+"}$"+' \n '+epoch_name_ax+' State PC1', labelpad=-5)
        ax.set_ylabel(r"$\bf{"+rule_name_ax+"}$"+' \n '+epoch_name_ax+' State PC2', labelpad=-5)
        ax.set_zlabel(r"$\bf{"+rule_name_ax+"}$"+' \n '+epoch_name_ax+' State PC3', labelpad=-5)
        save_axes = ax_labels

    if ax_labels=='pca_fp':
        ax.set_xlabel(r"$\bf{"+rule_name_ax+"}$"+' \n '+epoch_name_ax+' FxdPt PC1', labelpad=-5)
        ax.set_ylabel(r"$\bf{"+rule_name_ax+"}$"+' \n '+epoch_name_ax+' FxdPt PC2', labelpad=-5)
        ax.set_zlabel(r"$\bf{"+rule_name_ax+"}$"+' \n '+epoch_name_ax+' FxdPt PC3', labelpad=-5)
        save_axes = ax_labels

    elif len(ax_labels)==3:
        ax.set_xlabel(ax_labels[0], labelpad=0)
        ax.set_ylabel(ax_labels[1], labelpad=0)
        ax.set_zlabel(ax_labels[2],labelpad=0)
        save_axes = 'custom'

    if plot_zero_plane:
        zlabel = 'out_null'
        save_axes = zlabel
        xx, yy = np.meshgrid(np.linspace(x1, x2, num=2), np.linspace(y1, y2, num=2))
        z = xx*0
        ax.plot_surface(xx, yy, z, alpha=0.1)
        ax.set_zlabel(r"$\bf{Output}$"+r'$\cos{\theta}$',labelpad=-10)
        # ax.text(x1, y1, 0, 'Output Null', (1,0,0))
        ax.set_zlim([-1.1,1.1])
        ax.set_zticks([-1,1])

    if len(z_label)>0:
        ax.set_zlabel(z_label,labelpad=0)
        save_axes = z_label

        
    ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
        
    if rule_str[0]==rule_str[1]:
        tit = 'Single Task : '+rule_name1+'\n '+r"$\bf{"+epoch_name1+"}$"+' vs. '+r"$\bf{"+epoch_name2+"}$"+' dynamics' 
        tit_name = ruleset+'_'+rule_name1+'_'+epoch_name1+'_v_'+epoch_name2+'_dynamics_3d'
    else:       
        tit = r"$\bf{"+rule_name1+"}$"+' vs. '+r"$\bf{"+rule_name2+"}$"+ '\n '+epoch_name+' dynamics'
        tit_name = ruleset+'_'+rule_name1+'_v_'+rule_name2+'_'+epoch_name+'_dynamics_3d'   
    plt.title(tit,y = title_y,fontsize  = fontsize)

    if len(view_ang)>0:
        ax.view_init(elev=view_ang[0], azim=view_ang[1])

    ax.dist = 12

    
    fig_fldr = os.path.join(figpath,'interp_fps',ruleset,rule_name_ax+'_'+save_axes)
    if not os.path.exists(fig_fldr):
        os.makedirs(fig_fldr)

    plt.savefig(os.path.join(figpath,fig_fldr,tit_name+'.pdf'),bbox_inches='tight')
    plt.show()

def gen_stim_trial_set(m,ri_set):
    rule1 = rules_dict['all'][ri_set[0]]
    trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    trial2 = same_stim_trial(trial1, ri_set[1]) 
    trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    return [trial1,trial2]

def gen_mov_trial_set(m,ri_set):
    if type(ri_set[0]) != str:
        rule1 = rules_dict['all'][ri_set[0]]
        rule2 = rules_dict['all'][ri_set[1]]
    else:
        rule1 = ri_set[0]
        rule2 = ri_set[1]
    trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    trial2 = gen_trials_from_model_dir(m,rule2,mode='test',noise_on = False)
    trial2 = same_mov_inds_py3(trial1, trial2) 
    trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    return [trial1,trial2]


def get_interp_filename(trial1,trial2,epoch_list,t_set):

    if epoch_list==['fix1','fix1']:
        epoch_list = ['delay1','delay1']

    n_stim_per_ring = int(np.shape(trial1.y)[2]-1)
    stim_size = int(2*n_stim_per_ring+1)

    rule1 = rules_dict['all'][np.argmax(trial1.x[0,0,stim_size:])]
    rule2 = rules_dict['all'][np.argmax(trial2.x[0,0,stim_size:])]
    ind_stim_loc1  = 180*trial1.y_loc[-1,t_set[0]]/np.pi
    ind_stim_loc2  = 180*trial2.y_loc[-1,t_set[1]]/np.pi
    filename = rule1+'_'+rule2+'_'+'_'.join(epoch_list)+'_x'+str(round(ind_stim_loc1,2))+'_x'+str(round(ind_stim_loc2,2))
    # filename = rule1+'_'+rule2+'_'+'_'.join(epoch_list)+'_x'+str(round(ind_stim_loc1,2))+'_x'+str(round(ind_stim_loc1,2))

    return filename

def gen_D_fp_qr(m,trial_set,ri_axs,epoch_list,trial_num):

    h_A = get_h_epoch(m,trial_set[ri_axs],epoch_list[ri_axs],trial_num[ri_axs])
    h_end = h_A[:,-1] - h_A[:,0]
    h_B = get_h_epoch(m,trial_set[1-ri_axs],epoch_list[1-ri_axs],trial_num[1-ri_axs])
    h_diff = h_B[:,-1] - h_A[:,-1]

    D_fp = np.concatenate((h_diff[:,np.newaxis],h_end[:,np.newaxis]),axis = 1)
    D_fp_qr,_ = LA.qr(D_fp)
    
    return D_fp_qr

def get_single_task_fp_filename(trial, epoch,t):
    ind_stim_loc  = str(int(180*trial.y_loc[-1,t]/np.pi))
    nonzero_stim = trial.stim_locs[t,:]<100
    stim_names = '_'.join(str(int(180*x/np.pi)) for x in trial.stim_locs[t,nonzero_stim])
    filename = epoch+'_trial'+str(t)+'_x'+stim_names+'_y'+ind_stim_loc

    return filename

def get_h_epoch(m,trial,epoch,t_num):
    _,x = gen_X_from_model_dir(m,trial)
    T_inds = get_T_inds(trial,epoch)
    h_epoch = x[:,t_num,T_inds]
    return h_epoch

def find_closest_fp_loc(h_end,x_star):

    closest_fp = np.argmin([LA.norm(h_end - x_star[fp,:]) for fp in range(len(x_star))])
    closest_fp_loc = x_star[closest_fp,:]
        
    return closest_fp, closest_fp_loc

def interp_h_tasks_w_context(m, ri_set,trial_set,epoch_list,D_use = [],n_trials = 8,n_interp = 20,
                        q_thresh = 1e-5,linestyle = ['-',':'],al = .3,lw = 3,tit = [],ri_axs = 0,
                             fig_width = 7,fig_height = 4,fontsize = 20,trial_num=[0,0],
                             step_file = 'interp_tasks_small_init_mov',figpath = [],ruleset = [],
                             subselect_unstable = False):
    
    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]
        hparams = model.hp

    fig = plt.figure(figsize=(fig_width,fig_height),tight_layout=True,facecolor='white')
    xs = np.linspace(-.5, .5, 1000)
    ys = np.sqrt(1 - xs**2)

    ax_fps = fig.add_axes([0,0,.7,1.3], projection='3d');
    # ax_eigenspec = fig.add_axes([.375,0,.35,1.3], projection='3d');
    ax_dst = fig.add_axes([.85,0.05,.15,1]);

    plt.rcParams.update({'font.size': fontsize})
    cmap_grad = plt.get_cmap('plasma')
    cmap_stim = plt.get_cmap('hsv')
    q_tol_name = 'tol_q_e_6.0'
    axs = 'pca_h'

    epoch_ax = epoch_list[ri_axs]
    rule_ax = rules_dict['all'][ri_set[ri_axs]]
    trial = gen_trials_from_model_dir(m,rule_ax,mode='test',noise_on = False)
    save_dir = os.path.join(m,'tf_fixed_pts_all_init',rule_ax)
    f = get_single_task_fp_filename(trial,epoch_ax,trial_num[ri_axs])
    filename = os.path.join(save_dir,f+'.npz')
    D_use = make_FP_axs(filename, m, rule_ax, epoch_ax, axs = axs, clust = 'False')

    rule1 = rules_dict['all'][ri_set[0]]
    rule2 = rules_dict['all'][ri_set[1]]
    _, rule_name1, _, _ = take_names(epoch_list[0],rule1)
    _, rule_name2, _, _ = take_names(epoch_list[1],rule2)
    axes_label1 = axs+' 1'
    axes_label2 = axs+' 2'

    D_fp_qr = gen_D_fp_qr(m,trial_set,ri_axs,epoch_list,trial_num)
    D_diff = D_fp_qr[:,0]
    D_end = D_fp_qr[:,1]

    T_inds = get_T_inds(trial_set[0],epoch_list[0])
    inputs_1 = trial_set[0].x[T_inds,trial_num[0],:]
    inputs_2 = trial_set[1].x[T_inds,trial_num[1],:]
    del_inputs = inputs_2 - inputs_1

    closest_fp_loc = np.zeros((2,len(D_fp_qr)))
    for ri in [1,0]:
        rule = rules_dict['all'][ri_set[ri]]
        trial = trial_set[ri]

        _,x = gen_X_from_model_dir(m,trial)
        h_epoch = np.transpose(x[:,:,T_inds],(1,2,0))

        h_end = x[:,trial_num[ri],T_inds[-1]]
        f = get_interp_filename(trial_set[0],trial_set[1],epoch_list,trial_num)
        save_dir = os.path.join(m,step_file,'_'.join([rules_dict['all'][ri_set[0]],rules_dict['all'][ri_set[1]]]))
        fp_struct = np.load(os.path.join(save_dir,q_tol_name,f+'_step_'+str(int(ri*(n_interp-1)))+'.npz'))
        x_star = fp_struct['xstar']
        fp_inds = np.where(fp_struct['qstar']<q_thresh)[0]
        _, closest_fp_loc[ri] = find_closest_fp_loc(h_end,x_star[fp_inds,:])

        big_stim_trials = np.where(trial.stim_strength[:,0]>.99)[0]

        for ti_ind in range(0,len(big_stim_trials),int(len(big_stim_trials)/n_trials)):
            ti = big_stim_trials[ti_ind]

            trial_label = str(round(np.min(trial.stim_locs[ti,:])/np.pi,1))+ ' pi'

            h0 = h_epoch[ti,0,:]
            x_t = trial.x[T_inds,ti,:]
            h_t = vanilla_run_with_h0(params, x_t, h0, hparams)

            jitter = np.dot(h_t,D_use)

    AB = LA.norm(closest_fp_loc[0,:] - closest_fp_loc[1,:])

    rel_fp_loc = [None]*n_interp
    diff_rel_fp = [None]*n_interp
    max_eig = [None]*n_interp
    for step_i in range(n_interp):
        c = cmap_grad(step_i/n_interp)

        step_inputs = inputs_1+del_inputs*(step_i/n_interp)
        x_t = step_inputs
        h0 = h_epoch[trial_num[ri],0,:]

        h_t = vanilla_run_with_h0(params, x_t, h0, hparams)

        f = get_interp_filename(trial_set[0],trial_set[1],epoch_list,trial_num)
        save_dir = os.path.join(m,step_file,'_'.join([rules_dict['all'][ri_set[0]],rules_dict['all'][ri_set[1]]]))
        fp_struct = np.load(os.path.join(save_dir,q_tol_name,f+'_step_'+str(step_i)+'.npz'))
        step_fps = fp_struct['xstar']
        fp_inds = np.where(fp_struct['qstar']<q_thresh)[0]
        print(fp_inds) 


        if subselect_unstable == True:
            is_unstable = np.zeros(len(fp_inds))
            for fp_ind in range(len(fp_inds)):

                fp = fp_inds[fp_ind]
                J_xstar = fp_struct['J_xstar'][[fp,],:,:]
                eig_decomps = comp_eig_decomp(J_xstar)
    
                if np.max(eig_decomps[0]['evals'].real)>.999999:
                    is_unstable[fp_ind] = 1

            fp_inds = fp_inds[is_unstable==1]

        print(fp_inds)

        fp_project_diff = np.dot(step_fps[fp_inds,:],D_diff)
        fp_project_end = np.dot(step_fps[fp_inds,:],D_end)

        step_i_frac = step_i/(n_interp-1)
        ax_fps.plot3D(step_i_frac*np.ones(len(h_t)),np.dot(h_t,D_diff),np.dot(h_t,D_end),
                   linewidth = 5, alpha = .5,linestyle = '-',c = c)
        ax_fps.plot3D(step_i_frac*np.ones(len(h_t)),np.dot(h_t,D_diff),np.dot(h_t,D_end),
                   linewidth = 2, alpha = al,linestyle = '-',c = 'k')
        ax_fps.scatter(step_i_frac,np.dot(h_t[-1,:],D_diff),np.dot(h_t[-1,:],D_end),
                       s = 40,c = 'k',marker = '^')
        ax_fps.scatter(step_i_frac,np.dot(h_t[0,:],D_diff),np.dot(h_t[0,:],D_end),
                       s = 40,c = 'k',marker = 'x')

        rel_fp, rel_fp_loc[step_i] = find_closest_fp_loc(h_t[-1,:],fp_struct['xstar'][fp_inds,:])
        if step_i>0:
            diff_rel_fp[step_i] = LA.norm(rel_fp_loc[step_i] - rel_fp_loc[step_i-1])

        for fp_ind in [rel_fp,]:#range(len(fp_inds)):

            fp = fp_inds[fp_ind]
            J_xstar = fp_struct['J_xstar'][[fp,],:,:]
            eig_decomps = comp_eig_decomp(J_xstar)

            dst_scale = 300

            stability_metric = np.max([0,1.5 - np.max(eig_decomps[0]['evals'].real)])
            if np.max(eig_decomps[0]['evals'].real)>1:
                facecolors_3d = 'w'
                edgecolors = c
            else:
                facecolors_3d = c
                edgecolors = 'w'
            
            al = stability_metric


            ax_fps.scatter(step_i_frac,fp_project_diff[fp_ind],fp_project_end[fp_ind],'o',
                          s = dst_scale, edgecolors = edgecolors, facecolors = facecolors_3d, alpha = al)

            evals, _ = LA.eig(fp_struct['J_xstar'][fp,:,:]) 
            max_eig[step_i] = np.max(evals.real)
            # ax_eigenspec.scatter(step_i_frac,evals.real[evals.real>.82],evals.imag[evals.real>.82],
            #                      s = 200, edgecolors = edgecolors, facecolors = facecolors_2d, alpha = al*.6)

    tick_fontsize = fontsize*.75
    label_fontsize = fontsize*1

    ax_fps.spines['top'].set_visible(False)
    ax_fps.spines['right'].set_visible(False)
    ax_fps.set_xticks([0,.2,.4,.6,.8,1])
    ax_fps.set_xticklabels([0,.2,.4,.6,.8,1],ha = 'center',fontdict={'fontsize':tick_fontsize})
    ax_fps.set_yticklabels([],{'fontsize':tick_fontsize})
    ax_fps.set_zticklabels([],{'fontsize':tick_fontsize})
    #     ax_fps.set_xlabel(r"Rule Input $\alpha$",fontsize = 20)
    #     ax_fps.set_ylabel(r"$\Delta{h}$ along diff "+rule_name1+ '\n vs. '+rule_name2,labelpad = -1,fontsize = 20)
    #     ax_fps.set_zlabel(r"$\Delta{h}$ along "+rule_name1+' '+r"$h_{\theta = 0}$",labelpad = -1,fontsize = 20)
    ax_fps.set_xlabel(r"Rule Input $\alpha$",fontsize = label_fontsize)
    ax_fps.xaxis.labelpad = 20
    ax_fps.set_ylabel(r"$\Delta{h^{Task2 \ End - Task1 \ End}}$",labelpad = -1,fontsize = label_fontsize)
    ax_fps.set_zlabel(r"$\Delta{h^{Task1 \ End - Task1 \ Start}}$",labelpad = -1,fontsize = label_fontsize)
    ax_fps.view_init(-136, -14)#(-142, 175)

    # twin object for two different y-axis on the sample plot
    ax_dst2=ax_dst.twinx()
    # make a plot with different y-axis using second axis object
    ax_dst.plot(np.linspace(0, 1, n_interp),diff_rel_fp,'ok', markersize = 10, alpha = .5)
    ax_dst2.plot([0,1],[1,1],'-m', alpha = .5)
    ax_dst.set_ylabel('Euc. Dst. from \n Prev. Fixed Point',color='k',fontsize=label_fontsize)
    ax_dst2.plot(np.linspace(0, 1, n_interp),max_eig,'om', markersize = 10, alpha = .5)
    ax_dst2.set_ylabel('Max Eigenvalue',color='m',fontsize=label_fontsize)
    ax_dst.spines['top'].set_visible(False)
    ax_dst.spines['right'].set_visible(False)
    ax_dst.set_xticks(range(0,n_interp+1,int(n_interp/2)))
    ax_dst.set_xticklabels([0,.5,1],fontdict={'fontsize':tick_fontsize})
    ytick_set = np.linspace(0,np.max([ax_dst.get_ylim()[1],1.2]),5)
    ax_dst.set_yticks(ytick_set)
    ytick_set = [str(round(float(ytick_set), 1)) for ytick_set in ytick_set]
    ax_dst.set_yticklabels(ytick_set,fontdict={'fontsize':tick_fontsize})
    ax_dst2.set_yticks([.9,1,1.1])
    ax_dst2.set_yticklabels([.9,1,1.1],fontdict={'fontsize':tick_fontsize,'color':'m'})
    #     ax_dst.set_yticks(fontsize=tick_fontsize)
    ax_dst.set_xlabel(r"Rule Input $\alpha$",fontsize = label_fontsize)    
    ax_dst.spines['top'].set_visible(False)
    ax_dst2.spines['top'].set_visible(False)
    ax_dst.set_xlim((0,1.1))

    if len(figpath)>0:

        tasks_str = '_'.join([rules_dict['all'][ri_set[0]],rules_dict['all'][ri_set[1]]])
        epoch_str = '_'.join(epoch_list)
        step_file='interp_3d_eig'

        fldr = os.path.join(figpath,step_file,tasks_str,epoch_str)
        if not os.path.exists(fldr):
            os.makedirs(fldr)

        if len(ruleset)>0:
            figname = ruleset+'_'+tasks_str+'_'+epoch_str+'_'+step_file
        else:
            figname = tasks_str+'_'+epoch_str+'_'+step_file
        plt.savefig(os.path.join(fldr,figname+'.pdf'),bbox_inches='tight')
        plt.savefig(os.path.join(fldr,figname+'.png'))
    return

from network import get_perf

# def init_from_other_task(m,params,hparams,ri_set,init_from,run_from,trial_offset = 0,mov = True):

#     if mov==False:
#         rule1 = rules_dict['all'][ri_set[0]]
#         trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
#         trial2 = same_stim_trial(trial1, ri_set[1]) 
#         trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
#     else:
#         trial_set = gen_mov_trial_set(m,ri_set)


#     trial_set = [trial1,trial2]

#     rule = rules_dict['all'][ri_set[run_from]]
#     trial = trial_set[run_from]
#     epoch = 'go1'
#     B = np.shape(trial.x)[1]
#     N = hparams['n_rnn']

#     _,h = gen_X_from_model_dir(m,trial_set[init_from])
#     T_inds = get_T_inds(trial,epoch)

#     y_hat = np.zeros((len(T_inds),B,3))
#     for ti in range(B):

#         h0 = h[:,(ti+trial_offset)%B,T_inds[0]]
#         x_t = trial.x[T_inds[1:],ti,:]
#         h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
#         y_hat[:,ti,:] = out_affine(params, h_t.T).T

#     y_loc = trial.y_loc[T_inds,:]
#     return get_perf(y_hat, y_loc)


def var_ex_X_task(ax, m, rule_set, epoch_set, n_components = 200, batch_size = 1000, plot_legend = False):

    rule_name_set = {}
    _, rule_name_set[0], _, _ = take_names(epoch_set[0],rule_set[0])
    _, rule_name_set[1], _, _ = take_names(epoch_set[1],rule_set[1])

    trial = gen_trials_from_model_dir(m,rule_set[0],mode='random',noise_on = False,batch_size = 200)
    X, _ = gen_X_from_model_dir_epoch(m,trial,epoch_set[0])
    pca = PCA(n_components = n_components)
    _ = pca.fit_transform(X.T)

    trial = gen_trials_from_model_dir(m,rule_set[1],mode='random',noise_on = False,batch_size = 200)
    X_ax, _ = gen_X_from_model_dir_epoch(m,trial,epoch_set[1])
    pca_ax = PCA(n_components = n_components)
    _ = pca_ax.fit_transform(X_ax.T)

    var_same = np.var(np.dot(pca.components_,X),axis = 1)
    var_ax = np.var(np.dot(pca_ax.components_,X),axis = 1)

    plt.rcParams.update({'font.size': 16})
    plt.plot(np.cumsum(var_same)/np.sum(var_same),'-ok',linewidth = 3,markersize = 10,alpha = .5,
             label = rule_name_set[0] + ' PCs')
    plt.plot(np.cumsum(var_ax)/np.sum(var_same),'-o',c = 'mediumorchid', linewidth = 3,markersize = 10,
             alpha = .5,label = rule_name_set[1] + ' PCs')
    plt.xlabel('N PCs',fontsize = 18)
    ax.xaxis.set_label_coords(.15, -0.025)
    plt.ylabel(rule_name_set[0] +'\n Variance Explained',fontsize = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([n_components,])
    ax.set_ylim([0,1.1])
    ax.set_xlim([-.05*n_components,2.5*n_components])
    if plot_legend==True:
        plt.legend(loc='upper center', bbox_to_anchor=(0.01, 1.3))

def load_fps_from_interp(m,script_name,rule_str,tol_q,filename_interp,step_i,q_thresh):

    fp_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)),
                          filename_interp+'_step_'+str(step_i)+'.npz')
    fp_struct = np.load(fp_dir)
    fp_inds = np.where(fp_struct['qstar']<q_thresh)[0] 
    J_xstar = fp_struct['J_xstar'][fp_inds,:,:]
    sorted_fps = fp_struct['xstar'][fp_inds,:]
    eig_decomps = comp_eig_decomp(J_xstar)
    
    return sorted_fps, eig_decomps

def plot_FPs(ax,sorted_fps, eig_decomps, D_use, step_i, n_interp = 20, stability_thresh = 1,ms = 20, 
    plot_unstable = True,plot_expansion = False,rf = 150):
    cmap_grad = plt.get_cmap('plasma')
    c = cmap_grad(step_i/n_interp)

    for fp_ind in range(len(sorted_fps)):

        fp = np.dot(sorted_fps[fp_ind,:],D_use[:,[0,1]])

        stability_metric = 1
        facecolors_3d = c
        facecolors_2d = c
        edgecolors = 'w'
        al = stability_metric

        if np.max(eig_decomps[fp_ind]['evals'].real)>stability_thresh:

            if plot_expansion:
                n_arg = np.argwhere(eig_decomps[fp_ind]['evals']>1)+1
                if len(n_arg)>0:
                    for arg in range(np.max(n_arg)):
                        rdots = np.dot(np.real(eig_decomps[fp_ind]['R'][:, arg]).T,D_use[:,[0,1]])
                        ldots = np.dot(np.real(eig_decomps[fp_ind]['L'][:, arg]).T,D_use[:,[0,1]])
                        overlap = np.dot(rdots,ldots.T)
                        r = np.concatenate((fp - rf*overlap*rdots, fp + rf*overlap*rdots),0)
                        plt.plot(r[0:4:2],r[1:4:2], c = 'k' ,alpha = .7,linewidth = 1.5)

            if plot_unstable:
                plt.plot(fp[0],fp[1],'o',c = c,linewidth = 10,markersize = ms,markerfacecolor = 'w',
                         alpha = stability_metric,markeredgewidth=3)
        else:
            plt.plot(fp[0],fp[1],'o',c = edgecolors,linewidth = 10,markersize = ms,
                markerfacecolor = facecolors_2d,alpha = stability_metric)

    ax.set_yticks([])
    ax.set_xticks([])
    remove_spines(ax)

def plot_bifurc_ends(m,D_use,rule_set,epoch_set,script_name,tol_q = 1e-6,t_set = [0,0],
                     q_thresh = .0000001,fig_size = 4, lims = [], fac_mult_lims = 1.5,
                     stability_thresh = 1,fp_first = True, color_stim = True, step_set = [0,19],
                     figpath = 'plot_bifurc_ends',ruleset = 'all',fontsize = 20, 
                     plot_unstable = True,plot_expansion = False,ax_labels = [],cmap_discrete = ['w','k']):
    
    cmap_grad = plt.get_cmap('plasma')

    rule1 = rules_dict['all'][rule_set[0]]
    rule2 = rules_dict['all'][rule_set[1]]
    rule_str = [rule1,rule2]

    if color_stim:
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
        trial2 = same_stim_trial(trial1, rule_set[1]) 
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    else:
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
        trial2 = gen_trials_from_model_dir(m,rule2,mode='test',noise_on = False)
        trial2 = same_mov_inds_py3(trial1, trial2) 
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)

    save_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)))

    filename_interp = get_interp_filename(trial1,trial2,epoch_set,t_set)
    trial_set = [trial1, trial2]

    trial_inds = range(0,np.shape(trial1.x)[1],int(np.shape(trial1.x)[1]/8))

    for ri in [0,1]:
        fig = plt.figure(figsize=(fig_size,fig_size),tight_layout=True,facecolor='white')
        ax = fig.add_axes([0,0,1,1])

        sorted_fps, eig_decomps = load_fps_from_interp(m,script_name,rule_str,tol_q,filename_interp,
                                                         step_set[ri],q_thresh)

        if fp_first:
            plot_FPs(ax,sorted_fps, eig_decomps, D_use, step_set[ri], 
                stability_thresh = stability_thresh, plot_unstable = plot_unstable,
                plot_expansion = plot_expansion)
        
        rule = rule_str[ri]
        epoch = epoch_set[ri]
        c_master = cmap_discrete[ri]
        print(c_master)

        trial = gen_trials_from_model_dir(m,rule,mode='test',noise_on = False)

        _,x = gen_X_from_model_dir(m,trial)
        T_inds = get_T_inds(trial,epoch_set[ri])
        T_inds_plot = range(np.max([T_inds[0]-1,0]),T_inds[-1])
        x_epoch = np.transpose(x[:,:,T_inds],(1,2,0))


        cmap_state = plt.get_cmap('hsv')
        if color_stim:
            c_set = np.min(trial.stim_locs[:,[0,2]],axis = 1)
            s = np.argwhere(c_set == np.min(trial1.stim_locs[t_set[0],[0,2]]))[0][0]
        else:
            c_set = trial.y_loc[-1,:]
            s = np.argwhere(c_set==trial1.y_loc[-1,t_set[0]])[0][0]
        c = cmap_state(c_set[s]/(2*np.pi))

        lw = 10
        plot_N(x_epoch[trial_inds,:,:], D_use.T, c_set[trial_inds], linewidth = fig_size*2, alpha = .2)
        plot_N(x_epoch[[s,],:,:], D_use.T, c, linewidth = lw, alpha = .8 ,markersize = 16)
        plot_N(x_epoch[[s,],:,:], D_use.T, c_master, linewidth = lw/3, alpha = 1 ,markersize = 16)

        if not fp_first:
            plot_FPs(ax,sorted_fps, eig_decomps, D_use, step_set[ri], 
                stability_thresh = stability_thresh, plot_unstable = plot_unstable,
                plot_expansion = plot_expansion)

        if len(lims)>0:
            ax.set_xlim([fac_mult_lims*lims[0],fac_mult_lims*lims[1]])
            ax.set_ylim([fac_mult_lims*lims[2],fac_mult_lims*lims[3]])
        else:
            [x1,x2] = ax.get_xlim()
            [y1,y2] = ax.get_ylim()
            ax.set_xlim([x1-.2*abs(x1),x2+.2*abs(x2)])
            ax.set_ylim([y1-.2*abs(y1),y2+.2*abs(y2)])

        # ax.set_ylim([-1.15,1.15])
        # ax.set_xlim([-1.15,1.15])

        epoch_name, rule_name, _, _ = take_names(epoch_set[ri],rule_str[ri])
        
        if len(ax_labels)>0:
            make_lil_axes(ax,ax_labels)

        tit = r"$\bf{"+rule_name+"}$"+'\n '+epoch_name+' dynamics'
        tit_name = rule_name+'_'+epoch_name+'_dynamics'
        plt.title(tit,fontsize = fontsize)#,y = .9
        fig_fldr = os.path.join(figpath,'interp_fps',ruleset,'bifurcation')
        if not os.path.exists(fig_fldr):
            os.makedirs(fig_fldr)

        plt.savefig(os.path.join(figpath,fig_fldr,tit_name+'.pdf'),bbox_inches='tight')
        plt.savefig(os.path.join(figpath,fig_fldr,tit_name+'.png'))
        plt.show()

def plot_bifurc(ax,x_traj,threeD,linestyle = '-',c = 'r',cd = 'k',lw = 6,alpha = 1,ms = 6):

    if threeD:
            plt.plot(x_traj[:,0],x_traj[:,1],x_traj[:,2],
                     linestyle,
                     c = c, 
                     linewidth = lw, 
                     alpha = alpha, 
                     markersize = ms)
            plt.plot(x_traj[:,0],x_traj[:,1],x_traj[:,2],
                     linestyle,
                     c = cd, 
                     linewidth = lw/2, 
                     alpha = np.min((1,alpha*2)), 
                     markersize = ms)
            
            ax.scatter(x_traj[-1,0],x_traj[-1,1],x_traj[-1,2],
                       s = 300,
                       marker = '^',
                       facecolors = cd, 
                       edgecolors = c,
                       linewidth = lw/2, 
                       alpha = alpha)
            ax.scatter(x_traj[0,0],x_traj[0,1],x_traj[0,2],
                       s = 300,
                       marker = 'x',
                       facecolors = c,
                       linewidth = lw/2, 
                       alpha = alpha)
            ax.scatter(x_traj[0,0],x_traj[0,1],x_traj[0,2],
                       s = 200,
                       marker = 'x',
                       facecolors = cd, 
                       linewidth = lw/4, 
                       alpha = alpha)
    else:
            plt.plot(x_traj[:,0],x_traj[:,1],
                     linestyle,
                     c = c, 
                     linewidth = lw, 
                     alpha = alpha, 
                     markersize = ms)
            plt.plot(x_traj[:,0],x_traj[:,1],
                     linestyle,
                     c = cd, 
                     linewidth = lw/2, 
                     alpha = alpha, 
                     markersize = ms)
            
            ax.scatter(x_traj[-1,0],x_traj[-1,1],
                       s = 300,
                       marker = '^',
                       facecolors = cd, 
                       edgecolors = c,
                       linewidth = lw/2, 
                       alpha = alpha)
            ax.scatter(x_traj[0,0],x_traj[0,1],
                       s = 300,
                       marker = 'x',
                       facecolors = c,
                       linewidth = lw/2, 
                       alpha = alpha)
            ax.scatter(x_traj[0,0],x_traj[0,1],
                       s = 200,
                       marker = 'x',
                       facecolors = cd, 
                       linewidth = lw/4, 
                       alpha = alpha)

def axis_label_bifurc(ax,ax_labels,rule_name_ax,epoch_name_ax,bifurc,threeD,fontsize):

    if bifurc:
        
        if ax_labels=='pca_h':
            ax.set_xlabel(r"Rule Input $\alpha$", fontsize = fontsize, labelpad=10) 
            ax.set_ylabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' State PC1', labelpad=-5)
            if threeD:
                ax.set_zlabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' State PC2', labelpad=-5)

        elif ax_labels=='pca_fp':
            ax.set_xlabel(r"Rule Input $\alpha$", fontsize = fontsize, labelpad=10) 
            ax.set_ylabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' FxdPt PC1', labelpad=-5)
            if threeD:
                ax.set_zlabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' FxdPt PC2', labelpad=-5)
        
        else:
            ax.set_xlabel(ax_labels[0], labelpad=10)
            ax.set_ylabel(ax_labels[1], labelpad=-5)
            if threeD:
                ax.set_zlabel(ax_labels[2],labelpad=0)
    else:


        if ax_labels=='pca_h':
            ax.set_xlabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' State PC1', labelpad=-5)
            ax.set_ylabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' State PC2', labelpad=-5)
            if threeD:
                ax.set_zlabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' State PC3', labelpad=-5)

        elif ax_labels=='pca_fp':
            ax.set_xlabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' FxdPt PC1', labelpad=-5)
            ax.set_ylabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' FxdPt PC2', labelpad=-5)
            if threeD:
                ax.set_zlabel(r"$\bf{"+rule_name_ax+"}$"+'\n '+epoch_name_ax+' FxdPt PC3', labelpad=-5)

        else:
            ax.set_xlabel(ax_labels[0], labelpad=10)
            ax.set_ylabel(ax_labels[1], labelpad=-5)
            if threeD:
                ax.set_zlabel(ax_labels[2],labelpad=0)

def title_label_bifurc(bifurc,epoch_name1,epoch_name2,rule_name1,rule_name2,ruleset,title_y,fontsize):
    
    if bifurc:
        tit_label = ' Bifurcation Diagram'
        tit_save = 'bifurcation'
    else:
        tit_label = ' Dynamics'
        tit_save = 'dynamics'

    if rule_name1==rule_name2: 
        rule_name = rule_name1
        tit = r"$\bf{"+epoch_name1+"}$"+' vs. '+r"$\bf{"+epoch_name2+"}$"+ '\n '+rule_name+tit_label
        tit_name = ruleset+'_'+epoch_name1+'_v_'+epoch_name2+'_'+rule_name+'_'+tit_save

    else:
        tit = r"$\bf{"+rule_name1+"}$"+' vs. '+r"$\bf{"+rule_name2+"}$"+ '\n '+epoch_name1+tit_label
        tit_name = ruleset+'_'+rule_name1+'_v_'+rule_name2+'_'+epoch_name1+'_'+tit_save

        
    plt.title(tit,y = title_y, fontsize=fontsize)

    return tit, tit_name, tit_save

def check_plot_zero_plane(ax,plot_zero_plane,label_null = False):
    
    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()
    ax.set_xlim([x1-.1*abs(x1),x2+.1*abs(x2)])
    ax.set_ylim([y1-.1*abs(y1),y2+.1*abs(y2)])
    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()
    
    if plot_zero_plane:
        
        xs = np.linspace(-1, 1, 1000)
        ys = np.sqrt(1 - xs**2)
        xx, yy = np.meshgrid(np.linspace(x1, x2, num=2), np.linspace(y1, y2, num=2))
        z = xx*0
        ax.plot_surface(xx, yy, z, alpha=0.1)
        ax.set_zlabel(r"$\bf{Output}$"+r'$\cos{\theta}$',labelpad=-10)
        
        if label_null:
            ax.text(x1, y1, 0, 'Output Null', (1,0,0))
            
        ax.set_zlim([-1.1,1.1])
        ax.set_zticks([-1,1])
    

def bifurc(m,D_use,rule_set,epoch_set,t_set = [0,0],script_name = 'interp_tasks_small_init_mov',
           cmap_grad = plt.get_cmap('plasma'),cmap_discrete = ['w','k'],q_thresh = .001,
           n_skip_interps=1, rule_axs = 0,ax_labels = 'pca_h', plot_unstable = True,
           color_stim = True,fig_size = 4, fontsize = 20, figpath = [],ruleset = [],
           title_y = .9, n_interp = 20,n_trials = 0, stability_thresh = 1, threeD = True,
           lims = [], view_ang = [],lw = 6,ms = 6,linestyle = '-',al_state = 1,al_fp = 1, 
           bifurc = True, plot_zero_plane = False, tol_q = 1e-6,):
    
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(fig_size,fig_size),tight_layout=True,facecolor='white')
    if threeD:
        ax = fig.add_axes([0,0,1,1], projection='3d');
    else:
        ax = fig.add_axes([0,0,1,1]);

    rule1 = rules_dict['all'][rule_set[0]]
    rule2 = rules_dict['all'][rule_set[1]]
    rule_str = [rule1,rule2]

    epoch = epoch_set[rule_axs]
    epoch_name_ax, rule_name_ax, _, _ = take_names(epoch,rules_dict['all'][rule_set[rule_axs]])
    epoch_name1, rule_name1, _, _ = take_names(epoch_set[0],rule1)
    epoch_name2, rule_name2, _, _ = take_names(epoch_set[1],rule2)

    if color_stim:
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
        trial2 = same_stim_trial(trial1, rule_set[1]) 
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
    else:
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)
        trial2 = gen_trials_from_model_dir(m,rule2,mode='test',noise_on = False)
        trial2 = same_mov_inds_py3(trial1, trial2) 
        trial1 = gen_trials_from_model_dir(m,rule1,mode='test',noise_on = False)

    save_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)))
    filename_interp = get_interp_filename(trial1,trial2,epoch_set,t_set)

    for ri in range(2):

        cd = cmap_discrete[ri]
        trial = gen_trials_from_model_dir(m,rule_str[ri],mode='test',noise_on = False)
        
        #get traj
        _,x = gen_X_from_model_dir(m,trial)
        T_inds = get_T_inds(trial,epoch_set[ri])
        x_epoch = np.transpose(x[:,:,T_inds],(1,2,0))
        
        #get state colors
        cmap_state = plt.get_cmap('hsv')
        if color_stim:
            c_set = np.min(trial.stim_locs[:,[0,2]],axis = 1)
            s = np.argwhere(c_set == np.min(trial1.stim_locs[t_set[0],[0,2]]))[0][0]
            if cmap_discrete[0]=='w': #getting fp color right for reverse input interpolation 
                c = cmap_state(c_set[s]/(2*np.pi))
            else:
                c = cmap_state(1-(c_set[s]/(2*np.pi)))
        else:
            c_set = trial.y_loc[-1,:]
            s = np.argwhere(c_set==trial1.y_loc[-1,t_set[0]])[0][0]
            if cmap_discrete[0]=='w':
                c = cmap_state(c_set[s]/(2*np.pi))
            else:
                c = cmap_state(1-(c_set[s]/(2*np.pi)))
            
        # #plot other trials   
        if n_trials==0:
            trial_set = range(0)
        else:
            trial_set = range(0,np.shape(trial.x)[1],int(np.shape(trial.x)[1]/n_trials))
            
        for trial_i in trial_set:
            ci = cmap_state(c_set[trial_i]/(2*np.pi))
            x_traj_i = np.dot(x_epoch[trial_i,:,:],D_use)
            if bifurc:
                x_traj_i[:,2] = x_traj_i[:,1]
                x_traj_i[:,1] = x_traj_i[:,0]
                x_traj_i[:,0] = ri*np.ones(len(T_inds))
            plot_bifurc(ax,x_traj_i,threeD,c = ci,cd = cmap_discrete[ri],lw = lw,alpha = .1,ms = ms)
        0
        #plot fp trial
        x_traj = np.dot(x_epoch[s,:,:],D_use)
        if bifurc:
            x_traj[:,2] = x_traj[:,1]
            x_traj[:,1] = x_traj[:,0]
            x_traj[:,0] = ri*np.ones(len(T_inds))
        plot_bifurc(ax,x_traj, threeD, c = c, cd = cmap_discrete[ri],lw = lw,ms = ms)

        # set x-axis to ri if bifurcation diaram
        if bifurc:
            x_traj[:,0] = ri*np.ones(len(T_inds))
            
    #plot fps
    for step_i in range(0,n_interp,n_skip_interps):
        c = cmap_grad(step_i/n_interp)

        fp_dir = os.path.join(m,script_name,rule_str[0]+'_'+rule_str[1],'tol_q_e_'+str(-np.log10(tol_q)),
                              filename_interp+'_step_'+str(step_i)+'.npz')
        fp_struct = np.load(fp_dir)
        fp_inds = np.where(fp_struct['qstar']<q_thresh)[0]  
        J_xstar = fp_struct['J_xstar'][fp_inds,:,:]
        sorted_fps = fp_struct['xstar'][fp_inds,:]
        eig_decomps = comp_eig_decomp(J_xstar)

        for fp_ind in range(len(fp_inds)):

            fp = np.dot(sorted_fps[fp_ind,:],D_use)
            facecolors_3d = c
            fp_s = 50

            if bifurc:
                fp[2] = fp[1]
                fp[1] = fp[0]
                fp[0] = step_i/n_interp
            
            #sometimes we don't want to distinguish marginally stable fps so threshold is variable
            if np.max(eig_decomps[fp_ind]['evals'])>stability_thresh:
                if plot_unstable:
                    #open circles for unstable
                    facecolors_3d = 'w'
                    if threeD:
                        ax.scatter(fp[0],fp[1],fp[2],s = fp_s,marker = 'o',edgecolors = c,
                           facecolors = facecolors_3d, linewidth = 1/3, alpha = al_fp)
                    else:
                        ax.scatter(fp[0],fp[1],s = fp_s+20,marker = 'o',edgecolors = c,
                           facecolors = facecolors_3d, linewidth = lw/2, alpha = al_fp)
            else:
                if threeD:
                    ax.scatter(fp[0],fp[1],fp[2],s = fp_s,marker = 'o',edgecolors = c,
                           facecolors = facecolors_3d, linewidth = lw/3, alpha = al_fp)
                else:
                    ax.scatter(fp[0],fp[1],s = fp_s+20,marker = 'o',edgecolors = c,
                           facecolors = facecolors_3d, linewidth = lw/3, alpha = al_fp)
                    
    axis_label_bifurc(ax,ax_labels,rule_name_ax,epoch_name_ax,bifurc,threeD,fontsize)
    tit, tit_name, tit_save = title_label_bifurc(bifurc,epoch_name1,epoch_name2,rule_name1,
                                                 rule_name2,ruleset,title_y,fontsize)

    check_plot_zero_plane(ax,plot_zero_plane)
    
    if not bifurc:
        ax.set_xticks([])
        
    ax.set_yticks([])
    
    if threeD:
        
        if not plot_zero_plane:
            ax.set_zticks([])
        
        if len(lims)>0:
            ax.set_xlim([lims[0],lims[1]])
            ax.set_ylim([lims[2],lims[3]]) 
            ax.set_zlim([lims[4],lims[5]]) 
        else:
            [x1,x2] = ax.get_xlim()
            [y1,y2] = ax.get_ylim()
            [z1,z2] = ax.get_ylim()
            lims = [x1,x2,y1,y2,z1,z2]
            
        if len(view_ang)>0:
            ax.view_init(elev=view_ang[0], azim=view_ang[1])
        
        ax.dist = 12
        
    else:
        lims = []
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        [x1,x2] = ax.get_xlim()
        [y1,y2] = ax.get_ylim()
        lims = [x1,x2,y1,y2]

    fig_fldr = os.path.join(figpath,tit_save,ruleset)
    if not os.path.exists(fig_fldr):
        os.makedirs(fig_fldr)

    plt.savefig(os.path.join(figpath,fig_fldr,tit_name+'.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figpath,fig_fldr,tit_name+'.png'))
    plt.show()
    
    return lims

def var_x_epoch(m, n_components=100, timing_sort=True):
    """
    Calculates the variance of projections onto principal components for different epochs.

    Args:
      m: path to tf model
      n_components: Number of principal components to use.
      timing_sort: Whether to sort epochs by their timing information.

    Returns:
      var_test_sort: Variance of projections for each epoch and component.
      tick_labels: Labels for each epoch.
      keys: designate epoch labels
    """

    # Get relevant data for epochs
    h_all_byepoch_fit, _, _, _, _, _ = make_h_all(m, mode='random', rules=[])
    h_all_byepoch_test, _, _, h_all_timing, _, _ = make_h_all(m, mode='random', rules=[])

    # Extract keys and initialize arrays for storing variance
    keys = list(h_all_byepoch_fit.keys())
    var_test = np.empty((len(keys), len(keys), n_components))
    var_test_sort = np.empty((len(keys), len(keys), n_components))

    # Iterate through epochs to calculate variance
    for key_i, key in enumerate(keys):
        h_tf = h_all_byepoch_fit[key]

        # Reshape and transpose data for PCA
        x = np.transpose(h_tf, (2, 1, 0))
        X = np.reshape(x, (x.shape[0], -1))

        # Perform PCA and fit model
        pca = PCA(n_components=n_components)
        _ = pca.fit_transform(X.T)  # Fit PCA model

        # Calculate variance for test data using fitted components
        for key_test_i, key_test in enumerate(keys):
            h_test = h_all_byepoch_test[key_test]
            x = np.transpose(h_test, (2, 1, 0))
            X_test = np.reshape(x, (x.shape[0], -1))

            # Project test data onto principal components and calculate variance
            var = np.var(np.dot(pca.components_, X_test), axis=1)
            var_test[key_i, key_test_i, :] = var / np.sum(var)

    # Sort epochs based on timing (if specified)
    if timing_sort:
        epoch_order = np.argsort(h_all_timing)  # Sort by timing
    else:
        epoch_order = range(len(h_all_timing))  # Use original order

    # Rearrange variance based on sorted epoch order
    for e1 in range(len(epoch_order)):
        for e2 in range(len(epoch_order)):
            var_test_sort[e1, e2, :] = var_test[epoch_order[e1], epoch_order[e2], :]

    # Create tick labels for epochs
    tick_labels = []
    for e in epoch_order:
        epoch_name, rule_name, _, _ = take_names(keys[e][1], keys[e][0])
        key_str = rule_name[6:] + ' ' + epoch_name
        tick_labels.append(key_str)

    return var_test_sort, tick_labels, keys

def find_fp_dst_subspaces(m, D_use, trial_set, epoch_set, rule_set, trial_num, q_thresh = 1e-6, 
    q_tol_name = 'tol_q_e_6.0',script_name = 'interp_tasks_small_init_mov',plotting = True, 
    figpath = [], tit_name = [], subselect_unstable = False):

    n_interp = 20
    
    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]
        hparams = model.hp

    T_inds = get_T_inds(trial_set[0],epoch_set[0])
    inputs_1 = trial_set[0].x[0:T_inds[-1],trial_num[0],:]
    inputs_2 = trial_set[1].x[0:T_inds[-1],trial_num[1],:]
    del_inputs = inputs_2 - inputs_1

    rel_fp_loc = [None]*n_interp
    w_in = [None]*n_interp
    w_out = [None]*n_interp

    diff_rel_fp = [None]*n_interp
    diff_rel_fp_in = [None]*n_interp
    diff_rel_fp_out = [None]*n_interp
    max_eig = [None]*n_interp

    _,x = gen_X_from_model_dir(m,trial_set[0])
    h_epoch = np.transpose(x[:,:,0:],(1,2,0))

    for step_i in range(n_interp):

        step_inputs = inputs_1+del_inputs*(step_i/n_interp)
        x_t = step_inputs
        h0 = h_epoch[trial_num[0],0,:]

        h_t = vanilla_run_with_h0(params, x_t, h0, hparams)

        f = get_interp_filename(trial_set[0],trial_set[1],epoch_set,trial_num)
        save_dir = os.path.join(m,script_name,'_'.join([rules_dict['all'][rule_set[0]],rules_dict['all'][rule_set[1]]]))
        fp_struct = np.load(os.path.join(save_dir,q_tol_name,f+'_step_'+str(step_i)+'.npz'))
        step_fps = fp_struct['xstar']
        fp_inds = np.where(fp_struct['qstar']<q_thresh)[0]

        if subselect_unstable == True:
            is_unstable = np.zeros(len(fp_inds))
            for fp_ind in range(len(fp_inds)):

                fp = fp_inds[fp_ind]
                J_xstar = fp_struct['J_xstar'][[fp,],:,:]
                eig_decomps = comp_eig_decomp(J_xstar)
    
                if np.max(eig_decomps[0]['evals'].real)>.999999:
                    is_unstable[fp_ind] = 1

            fp_inds = fp_inds[is_unstable==1]

        rel_fp, rel_fp_loc[step_i] = find_closest_fp_loc(h_t[-1,:],fp_struct['xstar'][fp_inds,:])
        w_in[step_i] = np.dot(D_use[:,:2].T,rel_fp_loc[step_i])
        w_out[step_i] = np.dot(D_use[:,2:].T,rel_fp_loc[step_i])

        if step_i>0:
            diff_rel_fp[step_i] = LA.norm(rel_fp_loc[step_i] - rel_fp_loc[step_i-1])
            diff_rel_fp_in[step_i] = LA.norm(w_in[step_i] - w_in[step_i-1])
            diff_rel_fp_out[step_i] = LA.norm(w_out[step_i] - w_out[step_i-1])

        for fp_ind in [rel_fp,]:#range(len(fp_inds)):

            fp = fp_inds[fp_ind]
            J_xstar = fp_struct['J_xstar'][[fp,],:,:]
            eig_decomps = comp_eig_decomp(J_xstar)

            evals, _ = LA.eig(fp_struct['J_xstar'][fp,:,:]) 
            max_eig[step_i] = np.max(evals.real)

    if plotting==True:

        al = .6
        lw = 2
        plt.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(4.25,7),tight_layout=True,facecolor='white') #3.75,3.5

        ax1 = plt.subplot(1,2,1)

        plt.plot(max_eig,'ok',alpha = al,lw = lw)
        plt.plot([0, n_interp],[1,1],'-k',alpha = al/2,lw = lw)

        plt.ylim([.95,1.05])
        plt.ylabel('Max Eigenvalue',fontsize = 14)
        plt.xlabel(r"Rule Input $\alpha$",fontsize = 12)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        plt.xticks([])


        ax2 = plt.subplot(1,2,2)

        plt.plot(diff_rel_fp,'o',alpha = al,lw = lw,label = 'full space')
        plt.plot(diff_rel_fp_out,'o',alpha = al,lw = lw,label = 'out vis space')
        plt.plot(diff_rel_fp_in,'o',alpha = al,lw = lw,label = 'in vis space')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 2))

        plt.ylim([0,1.5])
        plt.ylabel('Euc. dst. btwn FPs',fontsize = 14)
        plt.xlabel(r"Rule Input $\alpha$",fontsize = 12)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        plt.xticks([])

    if len(tit_name)==0:
        tit_name = f

    fig_fldr = 'fp_dst_subspaces'
    if not os.path.exists(os.path.join(figpath,fig_fldr,script_name)):
        os.makedirs(os.path.join(figpath,fig_fldr,script_name))

    plt.savefig(os.path.join(figpath,fig_fldr,script_name,tit_name+'.pdf'),bbox_inches='tight')
    plt.show()

    return diff_rel_fp, diff_rel_fp_in, diff_rel_fp_out,f

def plot_var_in(m,D_use,rule_str,epoch_set,figpath,script_name,rule_axs,axs = 'pca_h_cat'):

    """
    This function plots the variance of the data projected onto the first two principal components
    and another visualized space.

    Args:
    m: The model to use.
    rule_str: The rule string to use.
    epoch_set: The epoch set to use.
    figpath: The filepath to save the figure.
    script_name: The name of the script.
    axs: The axis to use for saving the figure ('pca_h' or 'all').
    """

    fig = plt.figure(figsize=(3.2,1.8),tight_layout=True,facecolor='white')
    ax = plt.subplot(1,1,1)
    color_set = ['orangered','dodgerblue']
    line_style = ['-',':']
    lw = 3
    n_components = 10

    for ri in [0,1]:
        trial = gen_trials_from_model_dir(m,rule_str[ri],mode='random',noise_on = False)
        X, x = gen_X_from_model_dir_epoch(m,trial,epoch_set[ri])

        pca_ax = PCA(n_components = n_components)
        _ = pca_ax.fit_transform(X.T)

        inds = np.shape(x)[2]
        proj_in = np.zeros((inds,2))
        proj_tot = np.zeros((inds,n_components))
        for t_ind in range(inds):
            proj_in[t_ind,:] = np.var(np.dot(D_use[:,:2].T,x[:,:,t_ind]),axis = 1)
            proj_tot[t_ind,:] = np.var(np.dot(pca_ax.components_,x[:,:,t_ind]),axis = 1)

        plt.plot(proj_in[:,0],':',c = color_set[ri],label = 'x',alpha = .8,linewidth = lw)
        plt.plot(proj_in[:,1],':',c = color_set[ri],label = 'y',alpha = .8,linewidth = lw)
        for pc in range(n_components):
            plt.plot(proj_tot[:,pc],'-',c = color_set[ri],label = 'sum total',alpha = .4,linewidth = lw/3)

    plt.ylabel('Variance')
    plt.xlabel('Time in Epoch')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig_fldr = 'plot_var_in'
    if not os.path.exists(os.path.join(figpath,fig_fldr,script_name)):
        os.makedirs(os.path.join(figpath,fig_fldr,script_name))

    if axs == 'pca_h':
        plt.savefig(os.path.join(figpath,fig_fldr,script_name,'_'.join(rule_str)+'_'.join(epoch_set)+'.pdf'),bbox_inches='tight')
    else:
        plt.savefig(os.path.join(figpath,fig_fldr,script_name,rule_str[rule_axs]+'_'.join(rule_str)+'_'.join(epoch_set)+'.pdf'),bbox_inches='tight')
        plt.show()

###dendrogram viz fxns
def plot_epoch_rects(ax, epoch_binary, e_set, which_axes = 'y', rect_height = 1, rect_width = 58, ll = []):
    e_color = plt.get_cmap('terrain')

    if len(ll)>0:
        xl = ll[1]
        yl = ll[0]
    else:
        yl = np.max(ax.get_ylim())+2
        xl = np.max(ax.get_xlim())+2

        
    for ei in range(len(e_set)):
        e_name = e_set[ei]
        c = e_color(ei/len(e_set))
        
        if 'y' in which_axes:
            
            for ind in np.where(epoch_binary[e_name])[0]:
                rect=mpatches.Rectangle((ind-.5,yl),rect_height,rect_width,
                                        fill=True,color=c,facecolor=c,alpha = .3,clip_on=False)
                plt.gca().add_patch(rect)
            
        if 'x' in which_axes:
            
            for ind in np.where(epoch_binary[e_name])[0]:
                rect=mpatches.Rectangle((xl,ind-.5),rect_height,rect_width,
                                        fill=True,color=c,facecolor=c,alpha = .3,clip_on=False)
                plt.gca().add_patch(rect)

def make_cluster_midpoint_labels(clust):
    d = np.concatenate(([-1], np.where(np.diff(clust))[0],[len(clust)-1]),axis = 0)
    mid = np.zeros(len(d))
    cluster_size = np.zeros(len(d))
    for xi in range(len(d)-1):
        cluster_size[xi] = d[xi+1] - d[xi]
        mid[xi] = d[xi+1]-cluster_size[xi]/2+.5
    return cluster_size, mid