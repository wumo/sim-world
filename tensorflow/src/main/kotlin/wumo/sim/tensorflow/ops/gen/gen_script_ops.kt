/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.Shape
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray

interface gen_script_ops {
  fun _eagerPyFunc(input: Output, token: String, tout: Array<Long>, name: String = "EagerPyFunc") = run {
    buildOpTensors("EagerPyFunc", name) {
      addInput(input, false)
      attr("token", token)
      attr("Tout", tout)
    }
  }
}