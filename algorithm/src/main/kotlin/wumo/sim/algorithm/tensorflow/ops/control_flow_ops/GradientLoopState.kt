package wumo.sim.algorithm.tensorflow.ops.control_flow_ops

class GradientLoopState(val forwardContext: WhileContext, val outerGradState: GradientLoopState) {
}