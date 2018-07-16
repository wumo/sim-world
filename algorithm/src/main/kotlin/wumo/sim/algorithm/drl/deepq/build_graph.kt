package wumo.sim.algorithm.drl.deepq

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.tensorflow.training.Optimizer
import wumo.sim.util.a
import wumo.sim.util.scalarDimension

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
fun build_train(make_obs_ph: (String) -> Tensor, q_func: Q_func, num_actions: Int, optimizer: Optimizer,
                grad_norm_clipping: Any? = null, gamma: Double = 1.0,
                double_q: Boolean = true, name: String = "deepq",
                param_noise: Boolean = false, param_noise_filter_func: Any? = null) {
  val act_f = if (param_noise)
    build_act(make_obs_ph, q_func, num_actions, name)
  else
    build_act(make_obs_ph, q_func, num_actions, name)
}


/**
 * Creates the act function:
 * @param make_obs_ph: str -> tf.placeholder or TfInput,a function that take a name and creates a placeholder of input with that name
 * @param q_func: (tf.Variable, int, str, bool) -> tf.Variable, the model that takes the following inputs:
 * @param num_actions: int,number of actions
 * @param name: str
 * @param num_actions: int, number of actions.
 * @return act: (tf.Variable, bool, float) -> tf.Variable
 *function to select and action given observation.
 *       See the top of the file for details.
 */
fun build_act(make_obs_ph: (String) -> Tensor, q_func: Q_func, num_actions: Int, name: String = "deeqp"): Any {
  tf.subscope(name) {
    val observations_ph = make_obs_ph("observation")
    val stochastic_ph = tf.placeholder(scalarDimension, DT_BOOL, name = "stochastic")
    val update_eps_ph = tf.placeholder(scalarDimension, DT_FLOAT, name = "update_eps")
    
    val eps = tf.variable(scalarDimension, tf.constant_initializer(0), name = "eps")
    
    val q_values = q_func(observations_ph, num_actions, "q_func")
    val deterministic_actions = tf.argmax(q_values, 1)
    
    val batch_size = tf.shape(observations_ph)[0]
    val random_actions = tf.random_uniform(tf.stack(a(batch_size)), min = 0f, max = num_actions, dtype = DT_INT64)
    val chose_random = tf.less(tf.random_uniform(tf.stack(a(batch_size)), min = 0, max = 1, dtype = DT_FLOAT), eps)
    val stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
    
    val output_actions = tf.cond(stochastic_ph, { stochastic_actions }, { deterministic_actions })
    val update_eps_expr = eps.assign(tf.cond(tf.greater(update_eps_ph, tf.const(0)), { update_eps_ph }, { eps }))
    val _act = tf.function(inputs = a(observations_ph, stochastic_ph, update_eps_ph),
                           outputs = output_actions,
                           givens = mapOf(update_eps_ph to -1.0, stochastic_ph to true),
                           updates = listOf(update_eps_expr))

//    fun act(ob: Tensor, stochastic: Boolean = true, update_eps: Double = -1.0) =
//        _act(ob, stochastic, update_eps)
    TODO()
  }
}


private fun TF.function(inputs: Array<Tensor>, outputs: Tensor, givens: Any, updates: List<Operation>): Function {
  TODO()
}

interface Function {
  operator fun invoke(vararg inputs: Tensor): Unit
}
