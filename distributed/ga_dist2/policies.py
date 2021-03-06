import logging
import pickle

import h5py
import numpy as np
import tensorflow as tf

import tf_util as U

logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

        logger.debug('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.debug('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.debug('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.debug('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = U.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    # === Rollouts/training ===

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

        if timestep_limit and not env_timestep_limit:
            timestep_limit = timestep_limit
        elif timestep_limit and env_timestep_limit:
            timestep_limit = min(timestep_limit, env_timestep_limit)
        else:
            timestep_limit = env_timestep_limit
        #logger.info("rolling out with timestep limit of {}".format(timestep_limit))
        rews = []
        if save_obs:
            obs = []
        ob = env.reset()
        for t in range(1,timestep_limit+1):
            ac = self.act(np.expand_dims(ob, 0), random_stream=random_stream)
            if save_obs:
                obs.append(ob)
            ob, rew, done, _ = env.step(ac)
            rews.append(rew)
            if render:
                env.render()
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, obs
        return rews, t

    def act(self, ob, random_stream=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError

    # YOU NEED THE SHAPE TO DO THE GLOROT!
    def init_from_noise_idxs(self, samples, glorot_std):  # pylint: disable=W0613
        inits = np.zeros([self.num_params])
        i = 0
        for v in self.trainable_variables:
            policy_type, layer, param_type = v.name.split("/")
            shape =v.get_shape().as_list()
            v_n_scalars = np.prod(shape)
            if layer[:5]=="dense":
                if param_type[:6] == 'kernel':
                    # glorot init
                    # logger.info("trainable variable {} has shape {} and is being glorot initilalized".format(v.name, v.get_shape()))
                    samples_reshape = np.reshape(samples[i:i+v_n_scalars], shape)
                    out = samples_reshape * glorot_std / np.sqrt(np.square(samples_reshape).sum(axis=0, keepdims=True))
                    inits[i:i+v_n_scalars] = np.reshape(out, [v_n_scalars])
                else:
                    # zero init
                    assert param_type[:4] == 'bias'
                    # logger.info(
                    #     "trainable variable {} has shape {} and is being zero initialized".format(v.name, v.get_shape()))
                    # inits is already zero here
                    # the noise idxs passed over aren't used
                    pass
            elif layer[:4] == 'conv':
                if param_type[:6] == 'kernel':

                    # logger.info(
                    #     "trainable variable {} has shape {} and is being glorot initilalized".format(v.name, v.get_shape()))
                    samples_reshape = np.reshape(samples[i:i + v_n_scalars], shape)
                    out = samples_reshape * glorot_std / np.sqrt(np.square(samples_reshape).sum(axis=(0, 1, 2), keepdims=True))
                    inits[i:i + v_n_scalars] = np.reshape(out, [v_n_scalars])
                else:
                    # zero init
                    assert param_type[:4] == 'bias'
                    # logger.info(
                    #     "trainable variable {} has shape {} and is being zero initialized".format(v.name,
                    #                                                                               v.get_shape()))
                    # inits is already zero here
                    # the noise idxs passed over aren't used
            else:
                raise NotImplementedError
            i += v_n_scalars
        assert i  == self.num_params
        return inits

def bins(x, dim, num_bins, name):
    scores = U.dense(x, dim * num_bins, name, U.normc_initializer(0.01))
    scores_nab = tf.reshape(scores, [-1, dim, num_bins])
    return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1

class MujocoPolicy(Policy):
    def _initialize(self, ob_space, ac_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.ac_bins = ac_bins
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        assert len(ob_space.shape) == len(self.ac_space.shape) == 1
        assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
            'Action bounds required'

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            ob_mean = tf.get_variable(
                'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            in_mean = tf.placeholder(tf.float32, ob_space.shape)
            in_std = tf.placeholder(tf.float32, ob_space.shape)
            self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low
        assert isinstance(self.ac_bins, str)
        ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')

        if ac_bin_mode == 'uniform':
            # Uniformly spaced bins, from ac_space.low to ac_space.high
            num_ac_bins = int(ac_bin_arg)
            aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
            ac_range_1a = (ahigh - alow)[None, :]
            a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]

        elif ac_bin_mode == 'custom':
            # Custom bins specified as a list of values from -1 to 1
            # The bins are rescaled to ac_space.low to ac_space.high
            acvals_k = np.array(list(map(float, ac_bin_arg.split(','))), dtype=np.float32)
            logger.info('Custom action values: ' + ' '.join('{:.3f}'.format(x) for x in acvals_k))
            assert acvals_k.ndim == 1 and acvals_k[0] == -1 and acvals_k[-1] == 1
            acvals_ak = (
                (ahigh - alow)[:, None] / (acvals_k[-1] - acvals_k[0]) * (acvals_k - acvals_k[0])[None, :]
                + alow[:, None]
            )

            aidx_na = bins(x, adim, len(acvals_k), 'out')  # values in [0, k-1]
            a = tf.gather_nd(
                acvals_ak,
                tf.concat(2, [
                    tf.tile(np.arange(adim)[None, :, None], [tf.shape(aidx_na)[0], 1, 1]),
                    tf.expand_dims(aidx_na, -1)
                ])  # (n,a,2)
            )  # (n,a)
        elif ac_bin_mode == 'continuous':
            a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
        else:
            raise NotImplementedError(ac_bin_mode)

        return a

    def act(self, ob, random_stream=None):
        a = self._act(ob)
        if random_stream is not None and self.ac_noise_std != 0:
            a += random_stream.randn(*a.shape) * self.ac_noise_std
        return a

    @property
    def needs_ob_stat(self):
        return True

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)

class DiscretePolicy(Policy):
    def _initialize(self, ob_space, ac_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.ac_bins = ac_bins
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        assert len(ob_space.shape) == len(self.ac_space.shape) == 1
        # assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
        #     'Action bounds required'

        self.nonlin = {'tanh': tf.tanh,
                       'relu': tf.nn.relu,
                       'lrelu': U.lrelu,
                       'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            # ob_mean = tf.get_variable(
            #     'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            # ob_std = tf.get_variable(
            #     'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            # in_mean = tf.placeholder(tf.float32, ob_space.shape)
            # in_std = tf.placeholder(tf.float32, ob_space.shape)
            # self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
            #     tf.assign(ob_mean, in_mean),
            #     tf.assign(ob_std, in_std),
            # ])

            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(o)
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, n_hidden in enumerate(self.hidden_dims):
                x = tf.layers.dense(x, n_hidden, self.nonlin, 'l{}'.format(ilayer))
        elif self.connection_type == 'dense_skip':
            x = o
            for ilayer, n_hidden in enumerate(self.hidden_dims):
                r = tf.layers.dense(x, n_hidden, self.nonlin, 'l{}'.format(ilayer))
                x = tf.concat([x, r], axis = -1)

        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        a_dim = self.ac_space.shape[0]
        assert isinstance(self.ac_bins, str)
        #ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')
        a = tf.layers.dense(x, a_dim, tf.nn.softmax)


        # if ac_bin_mode == 'uniform':
        #     # Uniformly spaced bins, from ac_space.low to ac_space.high
        #     num_ac_bins = int(ac_bin_arg)
        #     aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
        #     ac_range_1a = (ahigh - alow)[None, :]
        #     a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]
        #
        # elif ac_bin_mode == 'custom':
        #     # Custom bins specified as a list of values from -1 to 1
        #     # The bins are rescaled to ac_space.low to ac_space.high
        #     acvals_k = np.array(list(map(float, ac_bin_arg.split(','))), dtype=np.float32)
        #     logger.info('Custom action values: ' + ' '.join('{:.3f}'.format(x) for x in acvals_k))
        #     assert acvals_k.ndim == 1 and acvals_k[0] == -1 and acvals_k[-1] == 1
        #     acvals_ak = (
        #         (ahigh - alow)[:, None] / (acvals_k[-1] - acvals_k[0]) * (acvals_k - acvals_k[0])[None, :]
        #         + alow[:, None]
        #     )
        #
        #     aidx_na = bins(x, adim, len(acvals_k), 'out')  # values in [0, k-1]
        #     a = tf.gather_nd(
        #         acvals_ak,
        #         tf.concat(2, [
        #             tf.tile(np.arange(adim)[None, :, None], [tf.shape(aidx_na)[0], 1, 1]),
        #             tf.expand_dims(aidx_na, -1)
        #         ])  # (n,a,2)
        #     )  # (n,a)
        # elif ac_bin_mode == 'continuous':
        #     a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
        # else:
        #     raise NotImplementedError(ac_bin_mode)

        return a

    def act(self, ob, random_stream=None):
        p =  self._act(ob)[0]
        choices = self.ac_space.shape[0]
        a = np.random.choice(choices, p=p)
        return a

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)

class TimeConvDiscreteOutPolicy(Policy):
    def _initialize(self, ob_space, ac_space, kernel_size,
                    n_channels, nonlin_type, n_blocks, layers_per_block):
        self.n_blocks = n_blocks
        self.layers_per_block = layers_per_block
        self.ac_space = ac_space
        self.ob_space = ob_space
        self.kernel_size = kernel_size
        self.n_channels = n_channels

        assert len(ob_space.shape) == 1
        assert len(self.ac_space.shape) == 0
        # assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
        #     'Action bounds required'

        self.nonlin = {'tanh': tf.tanh,
                       'relu': tf.nn.relu,
                       'lrelu': U.lrelu,
                       'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Policy network
            #logger.info("Observation space has shape {}".format(ob_space.shape))
            o = tf.placeholder(tf.float32, [None, self.ob_space.shape[0]])
            self.clear_ops, self.fill_ops, self.push_ops, ps  = self._make_net(o)

            self._act = U.function(inputs=[o], outputs=ps, updates=self.push_ops)

        U.get_session().run(self.clear_ops)
        U.get_session().run(self.fill_ops)

        queue_sizes = U.get_session().run([q.size() for q in self.qs])
        for i, qs in enumerate(queue_sizes):
            assert qs == 2**(i%self.layers_per_block)

        return scope

    def _make_net(self, x):

        h = x
        self.qs = []
        fill_ops = []
        clear_ops = []
        push_ops = []
        state_size = self.ac_space.n+2
        for b in range(self.n_blocks):
            for i in range(self.layers_per_block):
                rate = 2 ** i
                name = 'b{}-l{}'.format(b, i)

                q = tf.FIFOQueue(capacity=rate,
                                 dtypes=tf.float32,
                                 shapes=(1, state_size,))
                self.qs.append(q)

                clear = q.dequeue_many(q.size())
                fill = q.enqueue_many(tf.zeros((rate, 1, state_size)))

                push = q.enqueue([h])
                state_ = q.dequeue()

                clear_ops.append(clear)
                fill_ops.append(fill)
                push_ops.append(push)

                r = tf.layers.dense(tf.concat([state_, h], axis = 1), self.n_channels, activation=tf.nn.relu)
                h = tf.concat([r,h], axis = 1)
                state_size += self.n_channels

        ps = tf.layers.dense(h, self.ac_space.n,  activation = tf.nn.softmax)



        # # Initialize queues.
        # U.get_session().run(self.init_ops)

        return  clear_ops, fill_ops, push_ops, ps

    def act(self, ob, random_stream=None):
        if random_stream:
            return random_stream.choice(range(self.ac_space.n), self._act(ob)[0])
        else:
            p = self._act(ob)[0]
            return np.random.choice(self.ac_space.n, p=p)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):

        U.get_session().run(self.clear_ops)
        U.get_session().run(self.fill_ops)


        return super().rollout(env=env, render=render,
                               timestep_limit=timestep_limit,
                               save_obs=save_obs,
                               random_stream=random_stream)

class MetaMazePolicy(Policy):
    def _initialize(self, ob_space, ac_space, kernel_size,
                    n_channels, nonlin_type, n_blocks, layers_per_block):
        self.n_blocks = n_blocks
        self.layers_per_block = layers_per_block
        self.ac_space = ac_space
        self.ob_space = ob_space
        self.kernel_size = kernel_size
        self.n_channels = n_channels

        assert len(ob_space.shape) == 1
        assert len(self.ac_space.shape) == 0
        # assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
        #     'Action bounds required'

        self.nonlin = {'tanh': tf.tanh,
                       'relu': tf.nn.relu,
                       'lrelu': U.lrelu,
                       'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Policy network
            #logger.info("Observation space has shape {}".format(ob_space.shape))
            o = tf.placeholder(tf.float32, [None, self.ob_space.shape[0]])
            self.clear_ops, self.fill_ops, self.push_ops, ps  = self._make_net(o)

            self._act = U.function(inputs=[o], outputs=ps, updates=self.push_ops)

        U.get_session().run(self.clear_ops)
        U.get_session().run(self.fill_ops)

        queue_sizes = U.get_session().run([q.size() for q in self.qs])
        for i, qs in enumerate(queue_sizes):
            assert qs == 2**(i%self.layers_per_block)

        return scope

    def _make_net(self, x):

        h = x
        self.qs = []
        fill_ops = []
        clear_ops = []
        push_ops = []
        state_size = self.ac_space.n+2
        for b in range(self.n_blocks):
            for i in range(self.layers_per_block):
                rate = 2 ** i
                name = 'b{}-l{}'.format(b, i)

                q = tf.FIFOQueue(capacity=rate,
                                 dtypes=tf.float32,
                                 shapes=(1, state_size,))
                self.qs.append(q)

                clear = q.dequeue_many(q.size())
                fill = q.enqueue_many(tf.zeros((rate, 1, state_size)))

                push = q.enqueue([h])
                state_ = q.dequeue()

                clear_ops.append(clear)
                fill_ops.append(fill)
                push_ops.append(push)

                r = tf.layers.dense(tf.concat([state_, h], axis = 1), self.n_channels, activation=tf.nn.relu)
                h = tf.concat([r,h], axis = 1)
                state_size += self.n_channels

        ps = tf.layers.dense(h, self.ac_space.n,  activation = tf.nn.softmax)



        # # Initialize queues.
        # U.get_session().run(self.init_ops)

        return  clear_ops, fill_ops, push_ops, ps

    def act(self, ob, random_stream=None):
        if random_stream:
            return random_stream.choice(range(self.ac_space.n), self._act(ob)[0])
        else:
            p = self._act(ob)[0]
            return np.random.choice(self.ac_space.n, p=p)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):

        U.get_session().run(self.clear_ops)
        U.get_session().run(self.fill_ops)


        return super().rollout(env=env, render=render,
                               timestep_limit=timestep_limit,
                               save_obs=save_obs,
                               random_stream=random_stream)

class AtariPolicy(Policy):
    def _initialize(self, ob_space, ac_space, kernel_sizes, strides, n_channels, hidden_dims, nonlin_type):
        self.ac_space = ac_space
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.n_channels = n_channels

        assert len(kernel_sizes)==len(strides)==len(n_channels)

        assert len(ob_space.shape) == 3
        assert len(self.ac_space.shape) == 0
        # assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
        #     'Action bounds required'

        self.nonlin = {'tanh': tf.tanh,
                       'relu': tf.nn.relu,
                       'lrelu': U.lrelu,
                       'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            # ob_mean = tf.get_variable(
            #     'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            # ob_std = tf.get_variable(
            #     'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            # in_mean = tf.placeholder(tf.float32, ob_space.shape)
            # in_std = tf.placeholder(tf.float32, ob_space.shape)
            # self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
            #     tf.assign(ob_mean, in_mean),
            #     tf.assign(ob_std, in_std),
            # ])

            # Policy network
            #logger.info("Observation space has shape {}".format(ob_space.shape))
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(o)
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        x = o

        for iconv, kernel_size, stride, channel_dim in zip(range(len(self.kernel_sizes)), self.kernel_sizes, self.strides, self.n_channels):
            x = tf.layers.conv2d(x, channel_dim, kernel_size, stride, padding="SAME", name="conv{}".format(iconv))

        y = U.flattenallbut0(x)

        for ihidden, n_hidden in enumerate(self.hidden_dims):
            y = tf.layers.dense(y, n_hidden, self.nonlin, 'h{}'.format(ihidden))

        # Map to action
        a_dim = self.ac_space.n
        logits = tf.layers.dense(y, a_dim)
        a=tf.argmax(logits,1)

        return a

    def act(self, ob, random_stream=None):
        return self._act(ob)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)
