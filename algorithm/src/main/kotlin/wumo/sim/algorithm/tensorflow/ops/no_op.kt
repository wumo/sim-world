package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.*

fun TF.biasAddGrad(out_backprop: Tensor, data_format: String = "NHWC", name: String = "BiasAddGrad") =
    naryOp("BiasAddGrad", out_backprop.value(), name = name) {
      attr("data_format", data_format)
    }

fun TF.noOpDep(dep: Iterable<Operation>, name: String = "NoOp") =
    buildOp("NoOp", name = name) {
      setDevice(device)
      for (op in dep)
        addControlInput(op)
    }