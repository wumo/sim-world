package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.dim

fun TF_CPP.random_uniform(shape: Dimension, dtype: Int, name: String = "", scope: Scope = root) =
    scope.NewSubScope(name).let { s ->
      RandomUniform(s.WithOpName(name),
          Input(const(dim(shape.rank()), shape.asLongArray(), scope = s)),
          dtype).asOutput()
    }

fun TF_CPP.random_uniform(shape: Dimension, min: Float, max: Float,
                          name: String = "", scope: Scope = root): Output {
  scope.NewSubScope(name).let { s ->
    val rand = random_uniform(shape, DT_FLOAT, scope = s)
    val min = const(min, "min", s)
    val max = const(max, "max", s)
    return add(mul(rand, sub(max, min, scope = s), scope = s), min, name, s)
  }
}