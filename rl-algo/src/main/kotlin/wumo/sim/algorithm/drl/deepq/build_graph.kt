package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.drl.common.Function
import wumo.sim.algorithm.drl.common.Q_func
import wumo.sim.algorithm.drl.common.function
import wumo.sim.algorithm.drl.common.huber_loss
import wumo.sim.tensorflow.core.Graph.Graph.Keys
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.get
import wumo.sim.tensorflow.ops.basic.minus
import wumo.sim.tensorflow.ops.basic.plus
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.ops.training.Optimizer
import wumo.sim.tensorflow.ops.variables.*
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.tf.variableScope
import wumo.sim.tensorflow.types.BOOL
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.tensorflow.types.INT64
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray

fun build_train(
    makeObsPh: (String) -> TfInput,
    qFunc: Q_func,
    numActions: Int,
    optimizer: Optimizer,
    gradNormClipping: Output? = null,
    gamma: Float = 1f,
    doubleQ: Boolean = true,
    scope: String = "deepq",
    reuse: Reuse = ReuseOrCreateNew,
    paramNoise: Boolean = false,
    paramNoiseFilterFunc: ((Variable) -> Boolean)? = null)
    : t4<ActFunction, Function, Function, Map<String, Any>> {
  val (act_f, act_vars) = if (paramNoise)
    buildActWithParamNoise()
  else
    buildAct(makeObsPh, qFunc, numActions, scope, reuse)
  
  return variableScope(scope, reuse = reuse) {
    //set up placeholders
    val obs_t_input = makeObsPh("obs_t")
    val act_t_ph = tf.placeholder(Shape(-1), INT32, name = "action")
    val rew_t_ph = tf.placeholder(Shape(-1), FLOAT, name = "reward")
    val obs_tp1_input = makeObsPh("obs_tp1")
    val done_mask_ph = tf.placeholder(Shape(-1), FLOAT, name = "done")
    val importance_weights_ph = tf.placeholder(Shape(-1), FLOAT, name = "weight")
    
    //q network evaluation
    val q_t = qFunc(obs_t_input.get(), numActions, "q_func", ReuseExistingOnly)
    val q_func_vars = tf.currentGraph.getCollection(
        Keys.GLOBAL_VARIABLES,
        "^${tf.currentVariableScope.name}/q_func")
    
    //target q network evaluation
    val q_tp1 = qFunc(obs_tp1_input.get(), numActions, "target_q_func", ReuseOrCreateNew)
    val target_q_func_vars = tf.currentGraph.getCollection(
        Keys.GLOBAL_VARIABLES,
        "^${tf.currentVariableScope.name}/target_q_func")
    
    //q scores for actions which we know were selected in the given state.
    val q_t_selected = tf.sum(q_t * tf.oneHot({ act_t_ph }, { tf.const(numActions, it) }),
                              tf.const(1))
    
    //compute estimate of best possible value starting from state at t + 1
    val q_tp1_best = if (doubleQ) {
      val q_tp1_using_online_net = qFunc(obs_tp1_input.get(), numActions,
                                         "q_func", ReuseExistingOnly)
      val q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
      tf.sum(q_tp1 * tf.oneHot(q_tp1_best_using_online_net, tf.const(numActions)),
             tf.const(1))
    } else
      tf.max(q_tp1, tf.const(1))
    val q_tp1_best_masked = (tf.const(1f) - done_mask_ph) * q_tp1_best
    
    //compute RHS of bellman equation
    val q_t_selected_target = rew_t_ph + tf.const(gamma) * q_tp1_best_masked
    
    //compute the error (potentially clipped)
    val td_error = q_t_selected - tf.stopGradient(q_t_selected_target)
    val errors = huber_loss(td_error)
    val weighted_error = tf.mean(importance_weights_ph * errors, name = "weighted_error")
    
    //compute optimization op (potentially with gradient clipping)
    val optimize_expr = if (gradNormClipping != null) {
      var gradients = optimizer.computeGradients(weighted_error, variables = q_func_vars)
      gradients = gradients.map { pair ->
        val (grad, v) = pair
        if (grad != null)
          tf.clipByNorm(grad.toOutput(), gradNormClipping) to v
        else pair
      }
      optimizer.applyGradients(gradients)
    } else
      optimizer.minimize(weighted_error, variables = q_func_vars)
    
    //update_target_fn will be called periodically to copy Q network to target Q network
    val update_target_expr = run {
      val update_target_expr = mutableListOf<Op>()
      for ((v, v_target) in q_func_vars.sortedBy { it.name }
          .zip(target_q_func_vars.sortedBy { it.name }))
        update_target_expr += v_target.assign(v.toOutput()).op
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
    t4(act_f, train, update_target, mapOf("q_values" to q_values, "act_vars" to act_vars))
  }
}

fun buildAct(makeObsPh: (String) -> TfInput,
             qFunc: Q_func,
             numActions: Int,
             scope: String = "deepq",
             reuse: Reuse = CreateNewOnly): t2<ActFunction, Set<Variable>> =
    variableScope(scope, reuse) {
      val observations_ph = makeObsPh("observation")
      val stochastic_ph = tf.placeholder(scalarDimension, BOOL, name = "stochastic")
      val update_eps_ph = tf.placeholder(scalarDimension, FLOAT, name = "update_eps")
      val eps = tf.variable(scalarDimension, initializer = tf.constantInitializer(0), name = "eps")
  
      val q_values = qFunc(observations_ph.get(), numActions, "q_func", ReuseOrCreateNew)
      val deterministic_actions = tf.argmax(q_values, 1, name = "deterministic_actions")
      
      val batch_size = tf.shape(observations_ph.get())[0]
      val random_actions = tf.randomUniform(tf.stack(listOf(batch_size)),
                                            min = 0, max = numActions,
                                            dtype = INT64, name = "random_actions")
      val chose_random = tf.less(tf.randomUniform(tf.stack(listOf(batch_size)),
                                                  min = 0, max = 1, dtype = FLOAT), eps.toOutput())
      val stochastic_actions = tf.where(chose_random, random_actions,
                                        deterministic_actions, name = "stochastic_action")
      val output_actions = tf.cond(stochastic_ph,
                                   { stochastic_actions },
                                   { deterministic_actions }, name = "output_actions")
      val update_eps_expr = eps.assign(tf.cond(tf.greaterEqual({ update_eps_ph }, { tf.const(0f, it) }),
                                               { update_eps_ph }, { eps.toOutput() }))
//    val q_func_vars = tf.ctxVs.variable_subscopes["q_func"]!!.all_variables()
      val _act = function(inputs = a(observations_ph, stochastic_ph, update_eps_ph),
                          outputs = output_actions,
                          givens = a(update_eps_ph to -1.0f, stochastic_ph to true),
                          updates = a(update_eps_expr))
      t2(ActFunction(_act), tf.currentGraph.trainableVariables)
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

fun buildActWithParamNoise(): t2<ActFunction, Set<Variable>> {
  TODO("not implemented")
}
