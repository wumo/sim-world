package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.algorithm.tensorflow.ops.Op

class WhileContext(override val gradState: GradientLoopState? = null) : ControlFlowContext() {
  override fun addOp(op: Op) {
    TODO("not implemented")
  }
}