package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.naryOp

fun TF.dynamicStitch(indices: Array<Tensor>, data: Array<Tensor>, name: String = "DynamicStitch") =
    naryOp("DynamicStitch", name = name) {
      addInput(indices.map { it.value() })
      addInput(data.map { it.value() })
    }