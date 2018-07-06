package wumo.sim.algorithm.util.cpp_api.contrib

import org.bytedeco.javacpp.tensorflow.*
import org.lwjgl.system.linux.X11.True
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.cpp_api.layers.Dense
import wumo.sim.algorithm.util.cpp_api.ops.cast
import wumo.sim.algorithm.util.cpp_api.ops.const
import wumo.sim.algorithm.util.cpp_api.ops.oneHot

@Suppress("NAME_SHADOWING")
fun TF_CPP.one_hot_encoding(labels: Output,
                            num_class: Int,
                            on_value: Float = 1f, off_value: Float = 0f,
                            name: String = "OneHotEncoding", scope: Scope = root): Output {
  scope.NewSubScope(name).let { s ->
    val labels = if (labels.type() == DT_INT32) cast(labels, DT_INT64, scope = s) else labels
    return oneHot(labels, const(num_class, name = "depth", scope = s),
                  const(on_value, name = "on_value", scope = s),
                  const(off_value, name = "off_value", scope = s), scope = s)
  }
}

fun TF_CPP.fully_connected(inputs: Output,
                           num_outputs: Int,
                           activation_fn: Any? = null,
                           normalizer_fn: Any? = null,
                           normalizer_params: Any? = null,
                           weights_initializer: Any? = null,
                           weights_regularizer: Any? = null,
                           biases_initializer: Any? = null,
                           biases_regularizer: Any? = null,
                           reuse: Any? = null,
                           variables_collections: Any? = null,
                           outputs_collections: Any? = null,
                           trainable: Boolean = true,
                           scope: Scope? = root) {
  val layer = Dense(units = num_outputs,
                    use_bias = normalizer_fn == null && biases_initializer != null,
                    kernel_initializer = weights_initializer,
                    bias_initializer = biases_initializer,
                    kernel_regularizer = weights_regularizer,
                    bias_regularizer = biases_regularizer,
                    activity_regularizer = null,
                    trainable = trainable,
                    name = "fully_connected",
                    dtype = inputs.type(),
                    scope = scope)
//  val outputs=layer(inputs)
}

