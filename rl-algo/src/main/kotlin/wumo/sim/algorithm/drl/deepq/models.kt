package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.drl.common.Q_func
import wumo.sim.algorithm.drl.common.get_nerwork_builder
import wumo.sim.tensorflow.contrib.layers
import wumo.sim.tensorflow.ops.minus
import wumo.sim.tensorflow.ops.plus
import wumo.sim.tensorflow.tf

fun build_q_func(network: String,
                 network_kwargs: Map<String, Any>): Q_func {
  
  val hiddens = network_kwargs.getOrElse("hiddens") { listOf(256) } as List<Int>
  val dueling = network_kwargs.getOrDefault("dueling", true) as Boolean
  val layer_norm = network_kwargs.getOrDefault("layer_norm", false) as Boolean
  
  val network = get_nerwork_builder(network)(network_kwargs)
  return { input_placeholder, num_actions, scope, reuse ->
    tf.variableScope(scope, reuse = reuse) {
      var latent = network(input_placeholder)
      latent = layers.flatten(latent)
      val action_scores = tf.variableScope("action_value") {
        var action_out = latent
        
        for (hidden in hiddens) {
          action_out = layers.fully_connected(action_out,
                                              num_outputs = hidden,
                                              activation_fn = null)
          if (layer_norm)
            action_out = layers.layer_norm(action_out, center = true, scale = true)
        }
        layers.fully_connected(action_out, num_outputs = num_actions, activation_fn = null)
      }
      if (dueling) {
        val state_score = tf.variableScope("state_value") {
          var state_out = latent
          for (hidden in hiddens) {
            state_out = layers.fully_connected(state_out, num_outputs = hidden, activation_fn = null)
            if (layer_norm)
              state_out = layers.layer_norm(state_out, center = true, scale = true)
          }
          layers.fully_connected(state_out, num_outputs = 1, activation_fn = null)
        }
        val action_scores_mean = tf.mean(action_scores, tf.const(1))
        val action_scores_centered = action_scores - tf.expandDims(action_scores_mean, tf.const(1))
        state_score + action_scores_centered
      } else
        action_scores
    }
  }
}