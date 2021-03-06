import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable

fun register_state_grad() {
  /**Gradients for operators defined in state_ops.py.*/
  registerNonDifferentiable("Assign")
  registerNonDifferentiable("AssignAdd")
  registerNonDifferentiable("AssignSub")
  registerNonDifferentiable("ScatterAdd")
  registerNonDifferentiable("ScatterSub")
  registerNonDifferentiable("ScatterMul")
  registerNonDifferentiable("ScatterDiv")
  registerNonDifferentiable("ScatterNdUpdate")
  registerNonDifferentiable("ScatterNdAdd")
  registerNonDifferentiable("ScatterNdSub")
  registerNonDifferentiable("ScatterNdMul")
  registerNonDifferentiable("ScatterNdDiv")
}