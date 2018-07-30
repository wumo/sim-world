package wumo.sim.algorithm.drl.deepq

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.training.Optimizer
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray

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
    : tuple4<ActFunction, Function, Function, Map<String, Any>> {
  
  val (act_f, act_vars, act_graph_def) = if (param_noise)
    build_act_with_param_noise(make_obs_ph, q_func, num_actions,
                               _param_noise_filter_func = param_noise_filter_func,
                               name = name)
  else
    build_act(make_obs_ph, q_func, num_actions, name)
  
  tf.variable_scope(name) {
    //set up placeholders
    val obs_t_input = make_obs_ph("obs_t")
    val act_t_ph = tf.placeholder(dim(-1), DT_INT32, name = "action")
    val rew_t_ph = tf.placeholder(dim(-1), DT_FLOAT, name = "reward")
    val obs_tp1_input = make_obs_ph("obs_tp1")
    val done_mask_ph = tf.placeholder(dim(-1), DT_FLOAT, name = "done")
    val importance_weights_ph = tf.placeholder(dim(-1), DT_FLOAT, name = "weight")
    
    //q network evaluation
    val q_t = q_func(obs_t_input.get(), num_actions, "q_func", true)
    val q_func_vars = tf.ctxVs.variable_subscopes["q_func"]!!.all_variables()
    
    //target q network evaluation
    val q_tp1 = q_func(obs_tp1_input.get(), num_actions, "target_q_func", false)
    val target_q_func_vars = tf.ctxVs.variable_subscopes["target_q_func"]!!.all_variables()
    
    //q scores for actions which we know were selected in the given state.
    val q_t_selected = tf.sum(q_t * tf.oneHot(act_t_ph, tf.const(num_actions)), tf.const(1))
    
    //compute estimate of best possible value starting from state at t + 1
    val q_tp1_best = if (double_q) {
      val q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, "q_func", true)
      val q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
      tf.sum(q_tp1 * tf.oneHot(q_tp1_best_using_online_net, tf.const(num_actions)), tf.const(1))
    } else
      tf.max(q_tp1, tf.const(1))
    
    val q_tp1_best_masked = (tf.const(1f) - done_mask_ph) * q_tp1_best
    
    //compute RHS of bellman equation
    val q_t_selected_target = rew_t_ph + tf.const(gamma) * q_tp1_best_masked
    
    //compute the error (potentially clipped)
    val td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
    val errors = huber_loss(td_error)
    val weighted_error = tf.mean(importance_weights_ph * errors, name = "weighted_error")
    
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
      for ((v, v_target) in q_func_vars.sortedBy { it.name }
          .zip(target_q_func_vars.sortedBy { it.name }))
        update_target_expr += v_target.assign(v).op!!
      tf.group(update_target_expr)
    }
    
    tf.printGraph()
    
    val train = function(
        inputs = a(
            obs_t_input,
            act_t_ph,
            rew_t_ph,
            obs_tp1_input,
            done_mask_ph,
            importance_weights_ph),
        outputs = td_error,
        updates = a(optimize_expr))
    val update_target = function(updates = a(update_target_expr))
    val q_values = function(a(obs_t_input), q_t)
    return tuple4(act_f, train, update_target, mapOf("q_values" to q_values, "act_vars" to act_vars, "act_graph_def" to act_graph_def))
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
 * @param _param_noise_filter_func: tf.Variable -> bool
function that decides whether or not a variable should be perturbed. Only applicable
if param_noise is True. If set to None, default_param_noise_filter is used by default.
 
 * @return act: (tf.Variable, bool, float, bool, float, bool) -> tf.Variable
function to select and action given observation.
 */
fun build_act_with_param_noise(make_obs_ph: (String) -> TfInput,
                               q_func: Q_func,
                               num_actions: Int,
                               _param_noise_filter_func: ((Variable) -> Boolean)?,
                               name: String): tuple3<ActFunction, List<Variable>, ByteArray> {
  val param_noise_filter_func = _param_noise_filter_func ?: {
    when {
    //We never perturb non -trainable vars .
      it !in tf.trainables -> false
    //We perturb fully-connected layers.
      "fully_connected" in it.name -> true
    /*
    The remaining layers are likely conv or layer norm layers, which we do not wish to
    perturb (in the former case because they only extract features, in the latter case because
    we use them for normalization purposes). If you change your network, you will likely want
    to re-consider which layers to perturb and which to keep untouched.
     */
      else -> false
    }
  }
  
  tf.variable_scope(name) {
    val observations_ph = make_obs_ph("observation")
    val stochastic_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "stochastic")
    val update_eps_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_eps")
    val update_param_noise_threshold_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_param_noise_threshold")
    val update_param_noise_scale_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "update_param_noise_scale")
    val reset_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "reset")
    
    val eps = tf.get_variable(scalarDimension, tf.constant_initializer(0), name = "eps")
    val param_noise_scale = tf.get_variable(scalarDimension, initializer = tf.constant_initializer(0.01f), trainable = false, name = "param_noise_scale")
    val param_noise_threshold = tf.get_variable(scalarDimension, initializer = tf.constant_initializer(0.05f), trainable = false, name = "param_noise_threshold")
    
    //unmodified Q
    val q_values = q_func(observations_ph.get(), num_actions, "q_func", false)
    
    //Perturbable Q used for the actual rollout.
    val q_values_perturbed = q_func(observations_ph.get(), num_actions, "perturbed_q_func", false)
    //We have to wrap this code into a function due to the way tf.cond() works. See
    //https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for
    //a more detailed discussion.
    fun perturb_vars(original_scope: String, perturbed_scope: String): Operation {
      val all_vars = tf.ctxVs.variable_subscopes[original_scope]!!.all_variables()
      val all_perturbed_vars = tf.ctxVs.variable_subscopes[perturbed_scope]!!.all_variables()
      assert(all_vars.size == all_perturbed_vars.size)
      val perturb_ops = mutableListOf<Tensor>()
      for ((v, perturbed_var) in all_vars.zip(all_perturbed_vars)) {
        val op = if (param_noise_filter_func(perturbed_var))
        //Perturb this variable.
          tf.assign(perturbed_var, v + tf.random_normal(tf.shape(v), mean = tf.const(0f), stddev = param_noise_scale))
        else
        //Do not perturb, just assign.
          tf.assign(perturbed_var, v)
        perturb_ops += op
      }
      val t = tf.group(perturb_ops)
      return t
    }
    
    // Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
    // of the network and measures the effect of that perturbation in action space. If the perturbation
    // is too big, reduce scale of perturbation, otherwise increase.
    val q_values_adaptive = q_func(observations_ph.get(), num_actions, "adaptive_q_func", false)
    val perturb_for_adaption = perturb_vars(original_scope = "q_func", perturbed_scope = "adaptive_q_func")
    val kl = tf.sum(tf.softmax(q_values) * (tf.log(tf.softmax(q_values)) - tf.log(tf.softmax(q_values_adaptive))), axis = tf.const(-1))
    val mean_kl = tf.mean(kl)
    fun update_scale() =
        tf.control_dependencies(perturb_for_adaption) {
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
    val random_actions = tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = num_actions, dtype = DT_INT32)
    val chose_random = tf.less(tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = 1, dtype = DT_FLOAT), eps)
    val stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
    
    val output_actions = tf.cond(stochastic_ph, { stochastic_actions }, { deterministic_actions })
    val update_eps_expr = eps.assign(tf.cond(tf.greaterEqual(update_eps_ph, tf.const(0f)), { update_eps_ph }, { eps }))
    val updates = a(
        update_eps_expr,
        tf.cond(reset_ph, { Tensor(perturb_vars(original_scope = "q_func", perturbed_scope = "perturbed_q_func"), 0) },
                { Tensor(tf.group(listOf()), 0) }),
        tf.cond(update_param_noise_scale_ph, { update_scale() }, { tf.variable(0f, trainable = false) }),
        update_param_noise_threshold_expr)
    val inputs = a(observations_ph, stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph, update_param_noise_scale_ph)
    val outputs = output_actions
    val givens = a(update_eps_ph to -1.0, stochastic_ph to true, reset_ph to false, update_param_noise_threshold_ph to 0f, update_param_noise_scale_ph to false)
    val _act = function(inputs = inputs,
                        outputs = outputs,
                        givens = givens,
                        updates = updates)
    val q_func_vars = tf.ctxVs.variable_subscopes["q_func"]!!.all_variables()
    val act_graph_def =
        defaut(TF()) {
          val (_, _, act_graph_def) = build_act(make_obs_ph, q_func, num_actions, name)
          act_graph_def
        }
    return tuple3(ActWithParamNoise(_act), listOf(eps) + q_func_vars, act_graph_def)
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
fun build_act(make_obs_ph: (String) -> TfInput, q_func: Q_func, num_actions: Int, name: String): tuple3<ActFunction, List<Variable>, ByteArray> {
  tf.variable_scope(name) {
    val observations_ph = make_obs_ph("observation")
    val stochastic_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "stochastic")
    val update_eps_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_eps")
    
    val eps = tf.get_variable(scalarDimension, tf.constant_initializer(0), name = "eps")
    
    val q_values = q_func(observations_ph.get(), num_actions, "q_func", false)
    val deterministic_actions = tf.argmax(q_values, 1, name = "deterministic_actions")
    
    val batch_size = tf.shape(observations_ph.get())[0]
    val random_actions = tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = num_actions, dtype = DT_INT32, name = "random_actions")
    val chose_random = tf.less(tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = 1, dtype = DT_FLOAT), eps)
    val stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions, name = "stochastic_action")
    
    val output_actions = tf.cond(stochastic_ph, { stochastic_actions }, { deterministic_actions }, name = "output_actions")
    val update_eps_expr = eps.assign(tf.cond(tf.greaterEqual(update_eps_ph, tf.const(0f)), { update_eps_ph }, { eps }))
    val q_func_vars = tf.ctxVs.variable_subscopes["q_func"]!!.all_variables()
    val _act = function(inputs = a(observations_ph, stochastic_ph, update_eps_ph),
                        outputs = output_actions,
                        givens = a(update_eps_ph to -1.0f, stochastic_ph to true),
                        updates = a(update_eps_expr))
    val gradphDef = tf.g.toGraphDef()
    return tuple3(ActFunction(_act), listOf(eps) + q_func_vars, gradphDef)
  }
}

open class ActFunction(val act: Function) {
  operator fun invoke(ob: NDArray<*>, stochastic: Boolean = true, update_eps: Float = -1f) =
      act(ob, stochastic, update_eps)
}

class ActWithParamNoise(act: Function) : ActFunction(act) {
  operator fun invoke(ob: NDArray<*>,
                      reset: Boolean,
                      update_param_noise_threshold: Float,
                      update_param_noise_scale: Boolean,
                      stochastic: Boolean = true, update_eps: Float = -1f) =
      act(ob, stochastic, update_eps, reset, update_param_noise_threshold, update_param_noise_scale)
}