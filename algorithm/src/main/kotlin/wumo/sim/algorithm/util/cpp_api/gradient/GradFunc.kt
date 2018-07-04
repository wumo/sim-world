package wumo.sim.algorithm.util.cpp_api.gradient

import org.bytedeco.javacpp.tensorflow

class GradFunc {
  operator fun invoke(scope_: tensorflow.Scope, op: tensorflow.Operation, grad_inputs: MutableList<tensorflow.Output>, grad_outputs: MutableList<tensorflow.Output>) {
    TODO("not implemented")
  }
}