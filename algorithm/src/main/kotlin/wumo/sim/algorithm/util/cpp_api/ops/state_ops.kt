package wumo.sim.algorithm.util.cpp_api.ops


import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.assign(a: Output, b: Output, name: String = "", scope: Scope = root) =
    Assign(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.variable(initial_value: Any,
                    name: String = "", trainable: Boolean = true,
                    scope: Scope = root): Output {
  return variable(Dimension(), initial_value, name, trainable, scope)
}

fun TF_CPP.variable(shape: Dimension, initial_value: Any,
                    name: String = "", trainable: Boolean = true,
                    scope: Scope = root): Output {
  return variable(shape, { const(shape, initial_value, scope = it) },
      name, trainable, scope)
}

fun TF_CPP.variable(shape: Dimension, initializer: Output,
                    name: String = "", trainable: Boolean = true,
                    scope: Scope = root): Output {
  return variable(shape, { initializer }, name, trainable, scope)
}

inline fun TF_CPP.variable(shape: Dimension, initializer: (Scope) -> Output,
                           name: String = "", trainable: Boolean = true,
                           scope: Scope = root): Output {
  scope.NewSubScope(name).let { s ->
    val tensorShape = TensorShape()
    TF_CHECK_OK(TensorShapeUtils.MakeShape(shape.asLongArray(), tensorShape))
    val init = initializer(s)
    val dtype = init.type() % (DT_FLOAT_REF - 1)
    return Variable(s, tensorShape.asPartialTensorShape(), dtype).apply {
      val assign = assign(this.asOutput(), init, scope = s)
      init_ops += assign
    }.asOutput()
  }
}
