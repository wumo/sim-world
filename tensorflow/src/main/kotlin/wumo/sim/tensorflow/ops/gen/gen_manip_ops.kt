/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.buildOpTensor

fun TF.roll(input: Output, shift: Output, axis: Output, name: String = "Roll") = run {
  buildOpTensor("Roll", name) {
    addInput(input, false)
    addInput(shift, false)
    addInput(axis, false)
  }
}