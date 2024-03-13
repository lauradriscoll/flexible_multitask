"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import tools
import task
from task import generate_trials, generate_datasetTensors, datasetGeneratorFromTaskDef, defineDatasetFormat, rules_dict

def clip_grad(grad,max_norm):
    n = tf.norm(grad)

    # do_clip = tf.math.greater(n,max_norm)
    do_clip = 0

    with tf.Session() as session:
      session.run(tf.initialize_all_variables())

      if do_clip:#.eval():
          clipped_grad = (max_norm/n)*grad
      else:
          clipped_grad = grad

    return clipped_grad

def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """

    loc = np.arctan2(y[:,0], y[:,1])
    return np.mod(loc, 2*np.pi) # check this? January 22 2019


def tf_popvec(y):
    """Population vector read-out in tensorflow."""  

    loc = tf.atan2(y[:,0], y[:,1])
    return tf.mod(loc+np.pi, 2*np.pi) # check this? January 22 2019


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points

    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.1*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf


class LeakyRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 model_dir=None,
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._model_dir = model_dir
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.6#0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_tasks_total = len(rules_dict['all'])
        n_hidden = self._num_units

        if 'data' not in self._w_rec_init:
            w_in0_stim = (self.rng.randn(n_input - n_tasks_total, n_hidden) /
                     (np.sqrt(n_input) * self._w_in_start))

            w_in0_rule = (self.rng.randn(n_tasks_total, n_hidden) /
                     (np.sqrt(n_input) * self._w_in_start))

            b_in0 = []

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
        elif 'data' in self._w_rec_init:
            if os.path.exists(os.path.join(self._model_dir,'model_params.npz')):
                model_params = np.load(os.path.join(self._model_dir,'model_params.npz'))
            else:
                #first time loading from pretrained network
                model_params = np.load(os.path.join(self._w_rec_init ,'model_params.npz'))
                #save pretrain params with this network
                np.savez(os.path.join(self._model_dir,'model_params.npz'),**model_params)

            w_in0_stim = model_params['w_in'][:n_input - n_tasks_total,:]
            w_in0_rule = model_params['w_in'][n_input - n_tasks_total:n_input,:]
            w_rec0 = model_params['w_in'][-int(n_hidden):,:]
            b_in0 = model_params['b_in']


        self._initializer_input_stim = tf.constant_initializer(w_in0_stim, dtype=tf.float32)
        self._initializer_input_rule = {}

        for ri in range(n_tasks_total):
          rule = rules_dict['all'][ri]
          self._initializer_input_rule[rule] = tf.constant_initializer(w_in0_rule[ri,:], dtype=tf.float32)

        self._initializer_rnn = tf.constant_initializer(w_rec0, dtype=tf.float32)

        if len(b_in0)>0:
          self._initializer_bias = tf.constant_initializer(b_in0, dtype=tf.float32)
        else:
          self._initializer_bias = init_ops.zeros_initializer(dtype=self.dtype)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        n_tasks_total = len(rules_dict['all'])

        input_depth = inputs_shape[1].value
        self._kernel_input = {}
        self._kernel_input['stim'] = self.add_variable(
                'kernel_input_stim', shape=[input_depth-n_tasks_total, self._num_units],
                initializer=self._initializer_input_stim)

        for ri in range(n_tasks_total):
            rule = rules_dict['all'][ri]
            self._kernel_input[rule] = self.add_variable(
                'kernel_input_'+rule, shape=[1, self._num_units],
                initializer=self._initializer_input_rule[rule])

        self._kernel_rnn = self.add_variable(
                'kernel_rnn',
                shape=[self._num_units, self._num_units],
                initializer=self._initializer_rnn)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=self._initializer_bias)

        self.built = True

    def call(self, inputs, state): 
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        inputs_names = np.concatenate((['stim'], rules_dict['all']))
        inds = [self._kernel_input[x].shape[0].value for x in inputs_names]
        cumsum_inds = np.cumsum(inds)

        gate_input_ref = []
        for xi in range(len(inputs_names)):
            x = inputs_names[xi]
            if xi is 0:
                inputs_ref = inputs[:,:cumsum_inds[xi]]
                gate_input_sum = math_ops.matmul(inputs_ref,self._kernel_input[x])
            else:
                inputs_ref = inputs[:,cumsum_inds[xi-1]:cumsum_inds[xi]]
                # gate_input_sum = tf.math.add(gate_input_sum,math_ops.matmul(inputs_ref,self._kernel_input[x]))
                gate_input_sum = gate_input_sum+math_ops.matmul(inputs_ref,self._kernel_input[x])

        gate_rnn = math_ops.matmul(state, self._kernel_rnn)

        # gate_inputs = tf.math.add(gate_input_sum,gate_rnn)
        gate_inputs = gate_input_sum+gate_rnn
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        print(self._sigma)
        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        # with tf.Session() as sess:  print(noise.eval()) 
        gate_inputs = gate_inputs + noise # uncommented noise 14042020 

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output


class LeakyGRUCell(RNNCell):
  """Leaky Gated Recurrent Unit cell (cf. https://elifesciences.org/articles/21492).

  Args:
    num_units: int, The number of units in the GRU cell.
    alpha: dt/T, simulation time step over time constant
    sigma_rec: recurrent noise
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               alpha,
               sigma_rec=0,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(LeakyGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    # self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

    self._alpha = alpha
    self._sigma = np.sqrt(2 / alpha) * sigma_rec

    # TODO(gryang): allow this to use different initialization

  @property
  def state_size(self):
      return self._num_units

  @property
  def output_size(self):
      return self._num_units

  def build(self, inputs_shape):
      if inputs_shape[1].value is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                         % inputs_shape)

      input_depth = inputs_shape[1].value
      self._gate_kernel = self.add_variable(
          "gates/%s" % 'kernel',
          shape=[input_depth + self._num_units, 2 * self._num_units],
          initializer=self._kernel_initializer)
      self._gate_bias = self.add_variable(
          "gates/%s" % 'bias',
          shape=[2 * self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.constant_initializer(1.0, dtype=self.dtype)))
      self._candidate_kernel = self.add_variable(
          "candidate/%s" % 'kernel',
          shape=[input_depth + self._num_units, self._num_units],
          initializer=self._kernel_initializer)
      self._candidate_bias = self.add_variable(
          "candidate/%s" % 'bias',
          shape=[self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.zeros_initializer(dtype=self.dtype)))

      self.built = True

  def call(self, inputs, state):
      """Gated recurrent unit (GRU) with nunits cells."""

      gate_inputs = math_ops.matmul(
          array_ops.concat([inputs, state], 1), self._gate_kernel)
      gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

      value = math_ops.sigmoid(gate_inputs)
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

      r_state = r * state

      candidate = math_ops.matmul(
          array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
      candidate = nn_ops.bias_add(candidate, self._candidate_bias)
      candidate += tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)

      c = self._activation(candidate)
      # new_h = u * state + (1 - u) * c  # original GRU
      new_h = (1 - self._alpha * u) * state + (self._alpha * u) * c

      return new_h, new_h


class LeakyRNNCellSeparateInput(RNNCell):
    """The most basic RNN cell with external inputs separated.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCellSeparateInput, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
        else:
            raise ValueError

        self.w_rnn0 = w_rec0
        self._initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        self._kernel = self.add_variable(
                'kernel',
                shape=[self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """output = new_state = act(input + U * state + B)."""

        gate_inputs = math_ops.matmul(state, self._kernel)
        gate_inputs = gate_inputs + inputs  # directly add inputs
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output



class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 hp=None,
                 sigma_rec=None,
                 dt=None):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """

        # Reset tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        hp['alpha'] = 1.0*hp['dt']/hp['tau']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        if hp['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hp['in_type'])

        datasetType, datasetShape = defineDatasetFormat(hp)
        dataset = tf.data.Dataset.from_generator(lambda: datasetGeneratorFromTaskDef(
            hp, 'random'), datasetType, datasetShape)
        dataset = dataset.prefetch(4)
        self.datasetTensors = dataset.make_one_shot_iterator().get_next()

        hp['model_dir'] = model_dir
        self.model_dir = model_dir
        self.hp = hp
        self._build(hp)


    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=hp['learning_rate'])
        # Set cost
        self.set_optimizer(hp)

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver(max_to_keep=None)

    def _build_fused(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = self.datasetTensors[0] #tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        self.y = self.datasetTensors[1] #tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        self.c_mask = self.datasetTensors[2] #tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        # self.x = tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        # self.y = tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        # self.c_mask = tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        # Activation functions
        if hp['activation'] == 'power':
            f_act = lambda x: tf.square(tf.nn.relu(x))
        elif hp['activation'] == 'retanh':
            f_act = lambda x: tf.tanh(tf.nn.relu(x))
        elif hp['activation'] == 'relu+':
            f_act = lambda x: tf.nn.relu(x + tf.constant(1.))
        else:
            f_act = getattr(tf.nn, hp['activation'])

        # Recurrent activity
        if hp['rnn_type'] == 'LeakyRNN':
            n_in_rnn = self.x.get_shape().as_list()[-1]
            self.cell = LeakyRNNCell(n_rnn, n_in_rnn,
                                hp['alpha'],
                                sigma_rec=hp['sigma_rec'],
                                activation=hp['activation'],
                                w_rec_init=hp['w_rec_init'],
                                model_dir=hp['model_dir'],
                                rng=self.rng)
        elif hp['rnn_type'] == 'LeakyGRU':
            self.cell = LeakyGRUCell(
                n_rnn, hp['alpha'],
                sigma_rec=hp['sigma_rec'], activation=f_act)
        elif hp['rnn_type'] == 'LSTM':
            self.cell = tf.contrib.rnn.LSTMCell(n_rnn, activation=f_act)

        elif hp['rnn_type'] == 'GRU':
            self.cell = tf.contrib.rnn.GRUCell(n_rnn, activation=f_act)
        else:
            raise NotImplementedError("""rnn_type must be one of LeakyRNN,
                    LeakyGRU, EILeakyGRU, LSTM, GRU
                    """)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            self.cell, self.x, dtype=tf.float32, time_major=True)

        # FIXED Output
        with tf.variable_scope("output"):
            # grab outputs from saved network
            # model_params = np.load(os.path.join(hp['w_rec_init'],'model_params.npz'))
            model_params = np.load(os.path.join(self.model_dir,'model_params.npz'))
            w_out = model_params['w_out']
            b_out = model_params['b_out']
            self._initializer_w_out = tf.constant_initializer(w_out, dtype=tf.float32)
            self._initializer_b_out = tf.constant_initializer(b_out, dtype=tf.float32)

            w_out = tf.get_variable(
                'weights',
                shape = [n_rnn, n_output],
                initializer=self._initializer_w_out)

            b_out = tf.get_variable(
                'biases',
                shape = [n_output],
                initializer=self._initializer_b_out)

        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat_ shape (n_time*n_batch, n_unit)
        y_hat = tf.matmul(h_shaped, w_out) + b_out

        if hp['loss_type'] == 'lsq':
            # Least-square loss
            # y_hat = tf.sigmoid(y_hat_) #removed sigmoid Jan 24, 2019
            self.cost_lsq = tf.reduce_mean(
                tf.square((y_shaped - y_hat) * self.c_mask))
        else:
            y_hat = tf.nn.softmax(y_hat_)
            # Cross-entropy loss
            self.cost_lsq = tf.reduce_mean(
                self.c_mask * tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_shaped, logits=y_hat_))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_fused(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']


        # self._kernel_input['stim'] = self.add_variable(
        #         'kernel_input_stim', shape=[input_depth-n_tasks_total, self._num_units],
        #         initializer=self._initializer_input_stim)

        # for ri in range(n_tasks_total):
        #     rule = rules_dict['all'][ri]
        #     self._kernel_input[rule] = self.add_variable(
        #         'kernel_input_'+rule, shape=[1, self._num_units],
        #         initializer=self._initializer_input_rule[rule])

        inputs_names = np.concatenate((['stim'], rules_dict['all']))

        self.w_in = {}
        for v in self.var_list:
            if 'rnn' in v.name:
              for input_name in inputs_names:
                if input_name in v.name:
                    self.w_in[input_name] = v
                elif 'kernel_rnn' in v.name:# or 'weight' in v.name:
                    self.w_rec = v
                else:
                    self.b_rec = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        # if self.w_in.shape != (n_input, n_rnn):
        #     raise ValueError('Shape of w_in should be ' +
        #                      str((n_input, n_rnn)) + ', but found ' +
        #                      str(self.w_in.shape))

    def _build_seperate(self, hp):
        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = self.datasetTensors[0] #tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        self.y = self.datasetTensors[1] #tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        self.c_mask = self.datasetTensors[2] #tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        # self.x = tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        # self.y = tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        # self.c_mask = tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        sensory_inputs, rule_inputs = tf.split(
            self.x, [hp['rule_start'], hp['n_rule']], axis=-1)

        sensory_rnn_inputs = tf.layers.dense(sensory_inputs, n_rnn, name='sen_input')

        if 'mix_rule' in hp and hp['mix_rule'] is True:
            # rotate rule matrix
            kernel_initializer = tf.orthogonal_initializer()
            rule_inputs = tf.layers.dense(
                rule_inputs, hp['n_rule'], name='mix_rule',
                use_bias=False, trainable=False,
                kernel_initializer=kernel_initializer)

        rule_rnn_inputs = tf.layers.dense(rule_inputs, n_rnn, name='rule_input', use_bias=False)

        rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs

        # Recurrent activity
        self.cell = LeakyRNNCellSeparateInput(
            n_rnn, hp['alpha'],
            sigma_rec=hp['sigma_rec'],
            activation=hp['activation'],
            w_rec_init=hp['w_rec_init'],
            rng=self.rng)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            self.cell, rnn_inputs, dtype=tf.float32, time_major=True)

        # Output
        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat shape (n_time*n_batch, n_unit)
        y_hat = tf.layers.dense(
            h_shaped, n_output, activation=tf.nn.sigmoid, name='output')
        # Least-square loss

        self.cost_lsq = tf.reduce_mean(
            tf.square((y_shaped - y_hat) * self.c_mask))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_separate(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rec = v
                else:
                    self.b_rec = v
            elif 'sen_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_sen_in = v
                else:
                    self.b_in = v
            elif 'rule_input' in v.name:
                self.w_rule = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['rule_start'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                             str((hp['rule_start'], n_rnn)) + ', but found ' +
                             str(self.w_sen_in.shape))
        if self.w_rule.shape != (hp['n_rule'], n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((hp['n_rule'], n_rnn)) + ', but found ' +
                             str(self.w_rule.shape))

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def save_ckpt(self,step): # added Jan 9 2019
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir,'ckpts', 'model.ckpt'+'-'+str(step))
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def set_optimizer(self, hp, extra_cost=None, var_list=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = []
            for v in self.var_list:
                if 'kernel_input_'+hp['rule_trains'][0] in v.name:
                    var_list.append(v)

        print('Variables being optimized:')
        for v in [var_list[0],]: #ONLY TRAIN INPUT WEIGHTS
            print(v)

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)

        # gradient clipping
        self.clip_max_norm = 10
        clipped_gvs = [(clip_grad(grad, self.clip_max_norm), var) for grad, var in self.grads_and_vars]
        clipped_gs = [(clip_grad(grad, self.clip_max_norm)) for grad, var in self.grads_and_vars]

        self.train_step = self.opt.apply_gradients(clipped_gvs)
        self.clipped_gs = clipped_gs # trying to save gradients in log feb 8th

    def lesion_units(self, sess, units, verbose=False):
        """Lesion units given by units

        Args:
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for v in self.var_list:
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'output' in v.name:
                    # output weights
                    v_val[units, :] = 0
                elif 'rnn' in v.name:
                    # recurrent weights
                    v_val[n_input + units, :] = 0
                sess.run(v.assign(v_val))

        if verbose:
            print('Lesioned units:')
            print(units)


class FixedPoint_Model(Model):
    """For finding fixed points."""

    def __init__(self, model_dir,
                 hp=None,
                 sigma_rec=0,
                 dt=None):

        Model.__init__(self, model_dir, hp, sigma_rec, dt)

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        self.saver = tf.train.Saver(self.var_list)
