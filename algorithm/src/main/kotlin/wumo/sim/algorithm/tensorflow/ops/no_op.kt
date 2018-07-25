package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.naryOp


fun TF.biasAddGrad(out_backprop: Tensor, data_format: String = "NHWC", name: String = "BiasAddGrad") =
    naryOp("BiasAddGrad", out_backprop, name = name) {
      attr("data_format", data_format)
    }

fun TF.noOpDep(dep: Iterable<Operation>, name: String = "NoOp"): Operation {
  return g.nodeBuilder("NoOp", ctxNs.getUniqueFullName(name))
      .setDevice(ctxNs.device)
      .apply {
        for (op in dep)
          addControlInput(op)
      }.build()
}