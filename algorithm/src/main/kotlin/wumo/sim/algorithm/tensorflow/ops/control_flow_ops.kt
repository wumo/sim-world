package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF

fun TF.group(inputs: List<Operation>, name: String = "group_deps"): Operation {
  val ops_on_device = mutableMapOf<String, MutableList<Operation>>()
  for (input in inputs) {
    val dev = input.device
    ops_on_device.compute(dev) { _, list ->
      val list = list ?: mutableListOf()
      list += input
      list
    }
  }
  if (ops_on_device.size == 1) {
    val (dev, deps) = ops_on_device.entries.first()
    ctx.with_device(dev) {
      return noOpDep(deps, name)
    }
  }
  val all_deps = mutableListOf<Operation>()
  subscope(name) {
    for ((dev, deps) in ops_on_device) {
      ctx.with_device(dev) {
        all_deps += noOpDep(deps)
      }
    }
  }
  return noOpDep(all_deps, name)
}