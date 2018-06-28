package wumo.sim.algorithm.util.cpp_api.ops


import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.cpp_api.core.const

fun TF_CPP.assign(a: Output, b: Output, name: String = "") =
    Assign(if (name.isEmpty()) scope else scope.WithOpName(name),
        Input(a), Input(b)).asOutput()

fun TF_CPP.variable(shape: Dimension, initial_value: Any,
                    name: String = "",
                    trainable: Boolean = true): Output {
  return variable(shape, const(shape, initial_value), name, trainable)
}

fun TF_CPP.variable(shape: Dimension, initializer: Output,
                    name: String = "",
                    trainable: Boolean = true): Output {
  val tensorShape = TensorShape()
  TF_CHECK_OK(TensorShapeUtils.MakeShape(shape.asLongArray(), tensorShape))
  val dtype = initializer.type() % (DT_FLOAT_REF - 1)
  val subscope = if (name.isEmpty()) scope else scope.WithOpName(name)
  return Variable(subscope, tensorShape.asPartialTensorShape(), dtype).apply {
    val assign = Assign(subscope, this.asInput(), Input(initializer))
    init_ops += assign.asOutput()
  }.asOutput()
}
