package wumo.sim.algorithm.util.c_api.core

import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.c_api.Operation
import wumo.sim.algorithm.util.c_api.TF_C

fun TF_C.variable(shape: Dimension, initial_value: Any,
                  name: String = "variable",
                  trainable: Boolean = true) =
    scope(name) { variable(shape, const(shape, initial_value), useContextName, trainable) }

fun TF_C.variable(shape: Dimension, initializer: Operation,
                  name: String = "variable",
                  trainable: Boolean = true): Operation {
  val input = initializer[0]
  scope(name) {
    return g.opBuilder("VariableV2", contextPath)
        .setAttr("dtype", input.type)
        .setAttr("shape", shape)
        .build().apply {
          init_ops += assign(this[0], input)
          if (trainable) trainables += this
        }
  }
}