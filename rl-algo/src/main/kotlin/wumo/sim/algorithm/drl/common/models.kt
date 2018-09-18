package wumo.sim.algorithm.drl.common

import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.div
import wumo.sim.tensorflow.ops.variables.Reuse
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.t3
import kotlin.math.sqrt
import wumo.sim.tensorflow.layers.core.layers as core_layers

typealias Q_func = (Output, Int, String, Reuse) -> Output

val identity: TensorFunction = {
  it
}

fun mlp(num_layers: Int = 2,
        num_hidden: Int = 64,
        activation: TensorFunction = { tf.tanh(it) }): TensorFunction =
    { X ->
      var h = core_layers.flatten(X)
      for (i in 0 until num_layers)
        h = activation(fc(h, "mlp_fc$i",
                          num_hidden = num_hidden,
                          init_scale = sqrt(2f)))!!
      h
    }

fun convOnly(convs: List<t3<Int, Int, Int>> =
                 listOf(t3(32, 8, 4),
                        t3(64, 4, 2),
                        t3(64, 3, 1))): TensorFunction =
    { X ->
      var out = tf.cast(X, FLOAT) / 255f
      tf.variableScope("convnet") {
        for ((num_outputs, kernel_size, stride) in convs) {
        
        }
      }
      out
    }