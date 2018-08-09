package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.Op

class WhileContext(override val gradState: GradientLoopState? = null) : ControlFlowContext() {
  override fun addOp(op: Op) {
    TODO("not implemented")
  }
}