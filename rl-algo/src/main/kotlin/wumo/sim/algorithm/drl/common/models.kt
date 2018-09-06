package wumo.sim.algorithm.drl.common

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.Reuse
import wumo.sim.tensorflow.tf
import kotlin.math.sqrt
import wumo.sim.tensorflow.layers.layers as core_layers

typealias Q_func = (Output, Int, String, Reuse) -> Output
typealias TensorFunction = (Output) -> Output
typealias NetworkBuilder = (Map<String, Any>) -> TensorFunction

fun mlp(network_kwargs: Map<String, Any>): TensorFunction {
  val num_layers = network_kwargs.getOrDefault("num_layers", 2) as Int
  val num_hidden = network_kwargs.getOrDefault("num_hidden", 64) as Int
  val activation = network_kwargs.getOrElse("num_layers") {
    { x: Output ->
      tf.tanh(x)
    }
  } as TensorFunction
  
  return { X ->
    var h = core_layers.flatten(X)
    for (i in 0 until num_layers)
      h = activation(fc(h, "mlp_fc$i",
                        num_hidden = num_hidden,
                        init_scale = sqrt(2f)))
    h
  }
}

fun get_nerwork_builder(name: String): NetworkBuilder =
    when (name) {
      "mlp" -> ::mlp
      else -> error("Unknown network type: $name")
    }