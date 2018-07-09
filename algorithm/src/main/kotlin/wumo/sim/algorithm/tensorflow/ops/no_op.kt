package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF

fun TF.noOpDep(dep: Iterable<Operation>, name: String = "NoOp"): Operation {
  return g.nodeBuilder("NoOp", ctx.getUniqueFullName(name))
      .setDevice(ctx.device)
      .apply {
        for (op in dep)
          addControlInput(op)
      }.build()
}