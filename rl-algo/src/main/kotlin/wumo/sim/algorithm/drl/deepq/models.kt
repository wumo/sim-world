package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.drl.common.Q_func
import wumo.sim.tensorflow.contrib.layers
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.minus
import wumo.sim.tensorflow.ops.basic.plus
import wumo.sim.tensorflow.ops.variables.Reuse
import wumo.sim.tensorflow.tf

fun build_q_func(network: TensorFunction,
                 hiddens: List<Int> = listOf(256),
                 dueling: Boolean = true,
                 layer_norm: Boolean = false): Q_func =
    fun(input_placeholder: Output,
        num_actions: Int,
        scope: String,
        reuse: Reuse): Output =
        tf.variableScope(scope, reuse = reuse) {
          var latent = network(input_placeholder)!!
          latent = layers.flatten(latent)
          val action_scores = tf.variableScope("action_value") {
            var action_out = latent
            
            for (hidden in hiddens) {
              action_out = layers.fullyConnected(action_out,
                                                 num_outputs = hidden,
                                                 activation_fn = null)
              if (layer_norm)
                action_out = layers.layerNorm(action_out,
                                              center = true,
                                              scale = true)
              action_out = tf.relu(action_out)
            }
            layers.fullyConnected(action_out,
                                  num_outputs = num_actions,
                                  activation_fn = null)
          }
          if (dueling) {
            val state_score = tf.variableScope("state_value") {
              var state_out = latent
              for (hidden in hiddens) {
                state_out = layers.fullyConnected(state_out,
                                                  num_outputs = hidden,
                                                  activation_fn = null)
                if (layer_norm)
                  state_out = layers.layerNorm(state_out,
                                               center = true,
                                               scale = true)
                state_out = tf.relu(state_out)
              }
              layers.fullyConnected(state_out,
                                    num_outputs = 1,
                                    activation_fn = null)
            }
            val action_scores_mean = tf.mean(action_scores, tf.const(1))
            val action_scores_centered = action_scores - tf.expandDims(action_scores_mean,
                                                                       tf.const(1))
            state_score + action_scores_centered
          } else
            action_scores
        }