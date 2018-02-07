import numpy as np
import tensorflow as tf
import random
import os
import pickle

from six.moves import xrange
import pcl_rl.controller as controller
import pcl_rl.model as model
import pcl_rl.policy as policy
import pcl_rl.baseline as baseline
import pcl_rl.objective as objective
import pcl_rl.full_episode_objective as full_episode_objective
import pcl_rl.trust_region as trust_region
import pcl_rl.optimizers as optmizers
import pcl_rl.replay_buffer as replay_buffer
import pcl_rl.expert_paths as expert_paths
import pcl_rl.gym_wrapper as gym_wrapper
import pcl_rl.env_spec as env_spec

import rllab.misc.logger as rllab_logger

logging = tf.logging
gfile = tf.gfile

class Trainer(object):
  def __init__(self, train_args):
    # import ipdb; ipdb.set_trace()
    self.batch_size = train_args['batch_size']
    self.replay_batch_size = None
    if self.replay_batch_size is None:
      self.replay_batch_size = self.batch_size
    self.num_samples = 1

    self.env_str = train_args['env_str']
    self.env = gym_wrapper.GymWrapper(self.env_str,
                                      distinct=self.batch_size // self.num_samples,
                                      count=self.num_samples)
    self.eval_env = gym_wrapper.GymWrapper(
        self.env_str,
        distinct=self.batch_size // self.num_samples,
        count=self.num_samples)
    self.env_spec = env_spec.EnvSpec(self.env.get_one())

    self.max_step = train_args['max_step'] # 10?
    self.cutoff_agent = 1000
    self.num_steps = train_args['num_steps']
    self.validation_frequency = 1000

    self.target_network_lag = 0.99
    self.sample_from = 'online'
    assert self.sample_from in ['online', 'target']

    self.critic_weight = 1.0
    self.objective = 'pcl'
    self.trust_region_p = False
    self.value_opt = None
    assert not self.trust_region_p or self.objective in ['pcl', 'trpo']
    assert self.objective != 'trpo' or self.trust_region_p
    assert self.value_opt is None or self.value_opt == 'None' or \
        self.critic_weight == 0.0
    self.max_divergence = train_args['max_divergence']

    self.learning_rate = 0.0001
    self.clip_norm = 40.
    self.clip_adv = 1.
    self.tau = 0.0
    self.tau_decay = None
    self.tau_start = 0.1
    self.eps_lambda = 0.0
    self.update_eps_lambda = True
    self.gamma = 0.995
    self.rollout = train_args['rollout']
    self.use_target_values = True
    self.fixed_std = True
    self.input_prev_actions = False
    self.recurrent = False
    assert not self.trust_region_p or not self.recurrent
    self.input_time_step = False
    assert not self.input_time_step or (self.cutoff_agent <= self.max_step)

    self.use_online_batch = False
    self.batch_by_steps = True
    self.unify_episodes = False
    if self.unify_episodes:
      assert self.batch_size == 1

    self.replay_buffer_size = 5000
    self.replay_buffer_alpha = 0.001
    self.replay_buffer_freq = 1
    assert self.replay_buffer_freq in [-1, 0, 1]
    self.eviction = 'fifo'
    self.prioritize_by = 'step'
    assert self.prioritize_by in ['rewards', 'step']
    self.num_expert_paths = 0

    self.internal_dim = 128 # changed from 256
    self.value_hidden_layers = 2
    self.tf_seed = train_args['seed']

    self.save_trajectories_dir = None
    self.save_trajectories_file = (
        os.path.join(
            self.save_trajectories_dir, self.env_str.replace('-', '_'))
        if self.save_trajectories_dir else None)
    self.load_trajectories_file = None

    self.hparams = dict((attr, getattr(self, attr))
                        for attr in dir(self)
                        if not attr.startswith('__') and
                        not callable(getattr(self, attr)))

  def hparams_string(self):
    return '\n'.join('%s: %s' % item for item in sorted(self.hparams.items()))

  def get_objective(self):
    tau = self.tau
    if self.tau_decay is not None:
      assert self.tau_start >= self.tau
      tau = tf.maximum(
          tf.train.exponential_decay(
              self.tau_start, self.global_step, 100, self.tau_decay),
          self.tau)

    if self.objective in ['pcl', 'a3c', 'trpo', 'upcl']:
      cls = (objective.PCL if self.objective in ['pcl', 'upcl'] else
             objective.TRPO if self.objective == 'trpo' else
             objective.ActorCritic)
      policy_weight = 1.0

      return cls(self.learning_rate,
                 clip_norm=self.clip_norm,
                 policy_weight=policy_weight,
                 critic_weight=self.critic_weight,
                 tau=tau, gamma=self.gamma, rollout=self.rollout,
                 eps_lambda=self.eps_lambda, clip_adv=self.clip_adv,
                 use_target_values=self.use_target_values)
    elif self.objective in ['reinforce', 'urex']:
      cls = (full_episode_objective.Reinforce
             if self.objective == 'reinforce' else
             full_episode_objective.UREX)
      return cls(self.learning_rate,
                 clip_norm=self.clip_norm,
                 num_samples=self.num_samples,
                 tau=tau, bonus_weight=1.0)  # TODO: bonus weight?
    else:
      assert False, 'Unknown objective %s' % self.objective

  def get_policy(self):
    if self.recurrent:
      cls = policy.Policy
    else:
      cls = policy.MLPPolicy
    return cls(self.env_spec, self.internal_dim,
               fixed_std=self.fixed_std,
               recurrent=self.recurrent,
               input_prev_actions=self.input_prev_actions)

  def get_baseline(self):
    cls = (baseline.UnifiedBaseline if self.objective == 'upcl' else
           baseline.Baseline)
    return cls(self.env_spec, self.internal_dim,
               input_prev_actions=self.input_prev_actions,
               input_time_step=self.input_time_step,
               input_policy_state=self.recurrent,  # may want to change this
               n_hidden_layers=self.value_hidden_layers,
               hidden_dim=self.internal_dim,
               tau=self.tau)

  def get_trust_region_p_opt(self):
    if self.trust_region_p:
      return trust_region.TrustRegionOptimization(
          max_divergence=self.max_divergence)
    else:
      return None

  def get_value_opt(self):
    if self.value_opt == 'grad':
      return optimizers.GradOptimization(
          learning_rate=self.learning_rate, max_iter=5, mix_frac=0.05)
    elif self.value_opt == 'lbfgs':
      return optimizers.LbfgsOptimization(max_iter=25, mix_frac=0.1)
    elif self.value_opt == 'best_fit':
      return optimizers.BestFitOptimization(mix_frac=1.0)
    else:
      return None

  def get_model(self):
    cls = model.Model
    return cls(self.env_spec, self.global_step,
               target_network_lag=self.target_network_lag,
               sample_from=self.sample_from,
               get_policy=self.get_policy,
               get_baseline=self.get_baseline,
               get_objective=self.get_objective,
               get_trust_region_p_opt=self.get_trust_region_p_opt,
               get_value_opt=self.get_value_opt)

  def get_replay_buffer(self):
    if self.replay_buffer_freq <= 0:
      return None
    else:
      assert self.objective in ['pcl', 'upcl'], 'Can\'t use replay buffer with %s' % (
          self.objective)
    cls = replay_buffer.PrioritizedReplayBuffer
    return cls(self.replay_buffer_size,
               alpha=self.replay_buffer_alpha,
               eviction_strategy=self.eviction)

  def get_buffer_seeds(self):
    return expert_paths.sample_expert_paths(
        self.num_expert_paths, self.env_str, self.env_spec,
        load_trajectories_file=self.load_trajectories_file)

  def get_controller(self, env):
    """Get controller."""
    cls = controller.Controller
    return cls(env, self.env_spec, self.internal_dim,
               use_online_batch=self.use_online_batch,
               batch_by_steps=self.batch_by_steps,
               unify_episodes=self.unify_episodes,
               replay_batch_size=self.replay_batch_size,
               max_step=self.max_step,
               max_divergence=self.max_divergence,
               cutoff_agent=self.cutoff_agent,
               save_trajectories_file=self.save_trajectories_file,
               use_trust_region=self.trust_region_p,
               use_value_opt=self.value_opt not in [None, 'None'],
               update_eps_lambda=self.update_eps_lambda,
               prioritize_by=self.prioritize_by,
               get_model=self.get_model,
               get_replay_buffer=self.get_replay_buffer,
               get_buffer_seeds=self.get_buffer_seeds)

  def do_before_step(self, step):
    pass

  def run(self):
    """Run training."""
    is_chief = True
    sv = None

    def init_fn(sess, saver):
      ckpt = None
      if '' and sv is None:
        load_dir = ''
        ckpt = tf.train.get_checkpoint_state(load_dir)
      if ckpt and ckpt.model_checkpoint_path:
        logging.info('restoring from %s', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
      elif '': 
        logging.info('restoring from %s', '') 
        saver.restore(sess, '')

    tf.set_random_seed(self.tf_seed)
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self.controller = self.get_controller(self.env)
    self.model = self.controller.model
    self.controller.setup()
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      self.eval_controller = self.get_controller(self.eval_env)
      self.eval_controller.setup(train=False)

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    init_fn(sess, saver)

    self.sv = sv
    self.sess = sess

    logging.info('hparams:\n%s', self.hparams_string())

    model_step = sess.run(self.model.global_step)
    if model_step >= self.num_steps:
      logging.info('training has reached final step')
      return

    losses = []
    rewards = []
    all_ep_rewards = []
    for step in xrange(1 + self.num_steps):

      if sv is not None and sv.ShouldStop():
        logging.info('stopping supervisor')
        break

      self.do_before_step(step)

      (loss, summary,
       total_rewards, episode_rewards) = self.controller.train(sess)
      _, greedy_episode_rewards = self.eval_controller.eval(sess)
      self.controller.greedy_episode_rewards = greedy_episode_rewards
      losses.append(loss)
      rewards.append(total_rewards)
      all_ep_rewards.extend(episode_rewards)

      if (random.random() < 0.1 and summary and episode_rewards and
          is_chief and sv and sv._summary_writer):
        sv.summary_computed(sess, summary)

      last_greedy=0
      if len(greedy_episode_rewards) > 0:
        last_greedy = greedy_episode_rewards = greedy_episode_rewards[0]
      model_step = sess.run(self.model.global_step)
      if is_chief and step % self.validation_frequency == 0:
        # import ipdb; ipdb.set_trace()
        logging.info('at training step %d, model step %d: '
                     'avg loss %f, avg reward %f, '
                     'episode rewards: %f, greedy rewards: %f',
                     step, model_step,
                     np.mean(losses), np.mean(rewards),
                     np.mean(all_ep_rewards),
                     np.mean(greedy_episode_rewards),
		     ) 
        rllab_logger.record_tabular('Iteration', step)
        rllab_logger.record_tabular('AvgTrainReturn', np.mean(all_ep_rewards))
        rllab_logger.record_tabular('AvgEvalReturn', np.mean(greedy_episode_rewards))
        rllab_logger.record_tabular('EnvSteps', model_step)
        rllab_logger.dump_tabular(with_prefix=False)
        print(step)
        losses = []
        rewards = []
        all_ep_rewards = []

      if model_step >= self.num_steps:
        logging.info('training has reached final step')
        break

    if is_chief and sv is not None:
      logging.info('saving final model to %s', sv.save_path)
      sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

def main():
  logging.set_verbosity(logging.INFO)
  args = dict(batch_size=1,
              env_str="HalfCheetah-v1",
	      num_steps=1000000,
	      max_step=10,
              max_divergence=0.001,
	      rollout=1,
	      seed=1,
	)
  Trainer(args).run()

def run(args):
  Trainer(args).run()
    
if __name__ == '__main__':
  main()
