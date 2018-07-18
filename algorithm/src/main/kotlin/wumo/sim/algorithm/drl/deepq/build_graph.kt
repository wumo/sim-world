package wumo.sim.algorithm.drl.deepq

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Scope
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.tensorflow.training.Optimizer
import wumo.sim.util.a
import wumo.sim.util.dim
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.scalarDimension
import wumo.sim.util.tuple4
import kotlin.run

/**
 * Creates the train function
 *
 * @param make_obs_ph: str -> tf.placeholder or TfInput, a function that takes a name and creates a placeholder of input with that name
 * @param q_func: (tf.Variable, int, str, bool) -> tf.Variable, the model that takes the following inputs:
 * @param observation_in: object, the output of observation placeholder
 * @param num_actions: int, number of actions
 * @param num_actions: int, number of actions
 * @param optimizer: tf.train.Optimizer, optimizer to use for the Q-learning objective.
 * @param grad_norm_clipping: float or None, clip gradient norms to this value. If None no clipping is performed.
 * @param gamma: float, discount rate.
 * @param double_q: bool, if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a good idea to keep it enabled.
 * @param name: str or VariableScope, optional scope for variable_scope.
 * @param param_noise: bool, whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
 * @param param_noise_filter_func: tf.Variable -> bool, function that decides whether or not a variable should be perturbed.
 * Only applicable if param_noise is True. If set to None, default_param_noise_filter is used by default.
 
 * @return act: (tf.Variable, bool, float) -> tf.Variable, function to select an action given observation. See the top of the file for details.
 * train: (object, np.array, np.array, object, np.array, np.array) -> np.array, optimize the error in Bellman's equation.
 * See the top of the file for details. update_target: () -> () copy the parameters from optimized Q function to the target Q function.
 * See the top of the file for details. debug: {str: function}
 * a bunch of functions to print debug data like q_values.
 */
fun build_train(make_obs_ph: (String) -> TfInput,
                q_func: Q_func,
                num_actions: Int,
                optimizer: Optimizer,
                grad_norm_clipping: Tensor? = null,
                gamma: Float = 1f,
                double_q: Boolean = true,
                name: String = "deepq",
                param_noise: Boolean = false,
                param_noise_filter_func: ((Variable) -> Boolean)? = null)
    : tuple4<Any, Function, Function, Map<String, Function>> {
  tf.subscope(name) {
    val act_f = if (param_noise)
      build_act_with_param_noise(make_obs_ph, q_func, num_actions, param_noise_filter_func = param_noise_filter_func, scope = this)
    else
      build_act(make_obs_ph, q_func, num_actions, scope = this)
    
    //set up placeholders
    val obs_t_input = make_obs_ph("obs_t")
    val act_t_ph = tf.placeholder(dim(-1), DT_INT32, name = "action")
    val rew_t_ph = tf.placeholder(dim(-1), DT_FLOAT, name = "reward")
    val obs_tp1_input = make_obs_ph("obs_tp1")
    val done_mask_ph = tf.placeholder(dim(-1), DT_FLOAT, name = "done")
    val importance_weights_ph = tf.placeholder(dim(-1), DT_FLOAT, name = "weight")
    
    //q network evaluation
    val q_t = q_func(obs_t_input.get(), num_actions, "q_func", true)//TODO reuse parameters from act
    val q_func_vars = tf.global_variables//filter
    
    //target q network evaluation
    val q_tp1 = q_func(obs_tp1_input.get(), num_actions, "target_q_func", false)
    val target_q_func_vars = tf.global_variables
    
    //q scores for actions which we know were selected in the given state.
    val q_t_selected = tf.sum(q_t * tf.oneHot(act_t_ph, tf.const(num_actions)), tf.const(1))
    
    //compute estimate of best possible value starting from state at t + 1
    val q_tp1_best = if (double_q) {
      val q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, "q_func", true)
      val q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
      tf.sum(q_tp1 * tf.oneHot(q_tp1_best_using_online_net, tf.const(num_actions)), tf.const(1))
    } else {
      tf.max(q_tp1, tf.const(1))
    }
    
    val q_tp1_best_masked = (tf.const(1f) - done_mask_ph) * q_tp1_best
    
    //compute RHS of bellman equation
    val q_t_selected_target = rew_t_ph + tf.const(gamma) * q_tp1_best_masked
    
    //compute the error (potentially clipped)
    val td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
    val errors = huber_loss(td_error)
    val weighted_error = tf.mean(importance_weights_ph * errors)
    
    //compute optimization op (potentially with gradient clipping)
    val optimize_expr = if (grad_norm_clipping != null) {
      val gradients = optimizer.compute_gradients(weighted_error, q_func_vars)
      for ((i, pair) in gradients.withIndex()) {
        val (grad, v) = pair
        gradients[i]._1 = tf.clip_by_norm(grad, grad_norm_clipping)
      }
      optimizer.apply_gradients(gradients)
    } else
      optimizer.minimize(weighted_error, q_func_vars)
    
    //update_target_fn will be called periodically to copy Q network to target Q network
    val update_target_expr = run {
      val update_target_expr = mutableListOf<Operation>()
      for ((v, v_target) in q_func_vars.apply { sortBy { it.name } }.zip(
          target_q_func_vars.apply { sortBy { it.name } }))
        update_target_expr += v_target.assign(v).op
      tf.group(update_target_expr)
    }
    
    val train = function(
        inputs = a(
            obs_t_input.get(),
            act_t_ph,
            rew_t_ph,
            obs_tp1_input.get(),
            done_mask_ph,
            importance_weights_ph),
        outputs = td_error,
        updates = a(optimize_expr))
    val update_target = function(updates = a(update_target_expr))
    val q_values = function(a(obs_t_input.get()), q_t)
    return tuple4(act_f, train, update_target, mapOf("q_values" to q_values))
  }
}

/**
Creates the act function with support for parameter space noise exploration (https://arxiv.org/abs/1706.01905):
 * @param make_obs_ph: str -> tf.placeholder or TfInput
a function that take a name and creates a placeholder of input with that name
 * @param q_func: (tf.Variable, int, str, bool) -> tf.Variable
the model that takes the following inputs:
 * @param observation_in: object
the output of observation placeholder
 * @param num_actions: int
number of actions
 * @param name: str
 * @param param_noise_filter_func: tf.Variable -> bool
function that decides whether or not a variable should be perturbed. Only applicable
if param_noise is True. If set to None, default_param_noise_filter is used by default.
 
 * @return act: (tf.Variable, bool, float, bool, float, bool) -> tf.Variable
function to select and action given observation.
 */
fun build_act_with_param_noise(make_obs_ph: (String) -> TfInput,
                               q_func: Q_func,
                               num_actions: Int,
                               param_noise_filter_func: ((Variable) -> Boolean)?,
                               scope: Scope): Function {
  val param_noise_filter_func = param_noise_filter_func ?: { v: Variable -> true }
  fun scope_vars(original_scope: String): List<Variable> {
    TODO("not implemented")
  }
  with(scope) {
    val observations_ph = make_obs_ph("observation")
    val stochastic_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "stochastic")
    val update_eps_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_eps")
    val update_param_noise_threshold_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_param_noise_threshold")
    val update_param_noise_scale_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "update_param_noise_scale")
    val reset_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "reset")
    
    val eps = tf.variable(scalarDimension, tf.constant_initializer(0), name = "eps")
    val param_noise_scale = tf.variable(scalarDimension, initializer = tf.constant_initializer(0.01), trainable = false, name = "param_noise_scale")
    val param_noise_threshold = tf.variable(scalarDimension, initializer = tf.constant_initializer(0.05), trainable = false, name = "param_noise_threshold")
    
    //unmodified Q
    val q_values = q_func(observations_ph.get(), num_actions, "q_func", false)
    
    //Perturbable Q used for the actual rollout.
    val q_values_perturbed = q_func(observations_ph.get(), num_actions, "perturbed_q_func", false)
    //We have to wrap this code into a function due to the way tf.cond() works. See
    //https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for
    //a more detailed discussion.
    fun perturb_vars(original_scope: String, perturbed_scope: String): Tensor {
      val all_vars = scope_vars(original_scope)
      val all_perturbed_vars = scope_vars(perturbed_scope)
      val perturb_ops = mutableListOf<Tensor>()
      for ((v, perturbed_var) in all_vars.zip(all_perturbed_vars)) {
        val op = if (param_noise_filter_func(perturbed_var))
          tf.assign(perturbed_var, v + tf.random_normal(tf.shape(v), mean = tf.const(0f), stddev = param_noise_scale))
        else
          tf.assign(perturbed_var, v)
        perturb_ops += op
      }
      val t = tf.group(perturb_ops)
      return Tensor(t, 0)
    }
    
    // Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
    // of the network and measures the effect of that perturbation in action space. If the perturbation
    // is too big, reduce scale of perturbation, otherwise increase.
    val q_values_adaptive = q_func(observations_ph.get(), num_actions, "adaptive_q_func", false)
    val perturb_for_adaption = perturb_vars(original_scope = "q_func", perturbed_scope = "adaptive_q_func")
    val kl = tf.sum(tf.softmax(q_values) * (tf.log(tf.softmax(q_values)) - tf.log(tf.softmax(q_values_adaptive))), axis = tf.const(-1))
    val mean_kl = tf.mean(kl)
    fun update_scale() =
        control_dependencies(listOf(perturb_for_adaption.op)) {
          tf.cond(tf.less(mean_kl, param_noise_threshold),
                  { param_noise_scale.assign(param_noise_scale * tf.const(1.01f)) },
                  { param_noise_scale.assign(param_noise_scale / tf.const(1.01f)) })
        }
    
    //Functionality to update the threshold for parameter space noise.
    val update_param_noise_threshold_expr = param_noise_threshold.assign(
        tf.cond(tf.greaterEqual(update_param_noise_threshold_ph, tf.const(0f)),
                { update_param_noise_threshold_ph }, { param_noise_threshold }))
    
    //Put everything together.
    val deterministic_actions = tf.argmax(q_values_perturbed, axis = 1)
    val batch_size = tf.shape(observations_ph.get())[0]
    val random_actions = tf.random_uniform(tf.stack(a(batch_size)), dtype = DT_INT64, min = 0, max = num_actions)
    val chose_random = tf.less(tf.random_uniform(tf.stack(a(batch_size)), dtype = DT_FLOAT, min = 0, max = 1), eps)
    val stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
    
    val output_actions = tf.cond(stochastic_ph, { stochastic_actions }, { deterministic_actions })
    val update_eps_expr = eps.assign(tf.cond(tf.greaterEqual(update_eps_ph, tf.const(0)), { update_eps_ph }, { eps }))
    val updates = a(
        update_eps_expr,
        tf.cond(reset_ph, { perturb_vars(original_scope = "q_func", perturbed_scope = "perturbed_q_func") }, { Tensor(tf.group(listOf()), 0) }),
        tf.cond(update_param_noise_scale_ph, { update_scale() }, { tf.variable(0f, trainable = false) }),
        update_param_noise_threshold_expr)
    val inputs = a(observations_ph.get(), stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph, update_param_noise_scale_ph)
    val outputs = output_actions
    val givens = a(update_eps_ph to -1.0, stochastic_ph to true, reset_ph to false, update_param_noise_threshold_ph to false, update_param_noise_scale_ph to false)
    val _act = function(inputs = inputs,
                        outputs = outputs,
                        givens = givens,
                        updates = updates)

//    fun act(ob, reset, update_param_noise_threshold, update_param_noise_scale, stochastic = True, update_eps = -1):
//        return _act(ob, stochastic, update_eps, reset, update_param_noise_threshold, update_param_noise_scale)
//    return act
    TODO()
  }
}

/**
 * Creates the act function: Function to chose an action given an observation
 * @param make_obs_ph: str -> tf.placeholder or TfInput,a function that take a name and creates a placeholder of input with that name
 * @param q_func: (tf.Variable, int, str, bool) -> tf.Variable, the model that takes the following inputs:
 * @param num_actions: int,number of actions
 * @param name: str
 * @param num_actions: int, number of actions.
 * @return act: (tf.Variable, bool, float) -> tf.Variable
 *function to select and action given observation.
 *       See the top of the file for details.
 */
fun build_act(make_obs_ph: (String) -> TfInput, q_func: Q_func, num_actions: Int, scope: Scope): Function {
  with(scope) {
    val observations_ph = make_obs_ph("observation")
    val stochastic_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "stochastic")
    val update_eps_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_eps")
    
    val eps = tf.variable(scalarDimension, tf.constant_initializer(0), name = "eps")
    
    val q_values = q_func(observations_ph.get(), num_actions, "q_func", false)
    val deterministic_actions = tf.argmax(q_values, 1)
    
    val batch_size = tf.shape(observations_ph.get())[0]
    val random_actions = tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = num_actions, dtype = DT_INT32)
    val chose_random = tf.less(tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = 1, dtype = DT_FLOAT), eps)
    val stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
    
    val output_actions = tf.cond(stochastic_ph, { stochastic_actions }, { deterministic_actions })
    val update_eps_expr = eps.assign(tf.cond(tf.greater(update_eps_ph, tf.const(0f)), { update_eps_ph }, { eps }))
    val _act = function(inputs = a(observations_ph, stochastic_ph, update_eps_ph),
                        outputs = output_actions,
                        givens = a(update_eps_ph to -1.0f, stochastic_ph to true),
                        updates = a(update_eps_expr))
    return _act
  }
}
