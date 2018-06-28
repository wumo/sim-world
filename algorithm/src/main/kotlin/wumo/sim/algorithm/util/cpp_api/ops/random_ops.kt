package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.dim

fun TF_CPP.random_uniform(shape: Dimension, dtype: Int, name: String = "") =
    random_uniform(shape, dtype, scope.WithOpName(name))

private fun TF_CPP.random_uniform(shape: Dimension, dtype: Int, s: Scope) =
    RandomUniform(s,
        Input(const(dim(shape.rank()), shape.asLongArray())),
        dtype).asOutput()

fun TF_CPP.random_uniform(shape: Dimension, min: Float, max: Float,
                          name: String = ""): Output {
  val s = scope.WithOpName(name)
  val rand = random_uniform(shape, DT_FLOAT, s)
  val min = const(min, "min")
  val max = const(max, "max")
  return add(mul(rand, sub(max, min)), min)
}