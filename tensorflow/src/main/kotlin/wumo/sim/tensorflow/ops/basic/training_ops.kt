package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_training_ops

object training_ops {
  interface API {
    fun applyAdaMax(_var: Output, m: Output, v: Output, beta1Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyAdaMax"): Output {
      return gen_training_ops.applyAdaMax(_var, m, v, beta1Power, lr, beta1, beta2, epsilon, grad, useLocking, name)
    }
    
    fun applyAdadelta(_var: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyAdadelta"): Output {
      return gen_training_ops.applyAdadelta(_var, accum, accumUpdate, lr, rho, epsilon, grad, useLocking, name)
    }
    
    fun applyAdagrad(_var: Output, accum: Output, lr: Output, grad: Output, useLocking: Boolean = false, updateSlots: Boolean = true, name: String = "ApplyAdagrad"): Output {
      return gen_training_ops.applyAdagrad(_var, accum, lr, grad, useLocking, updateSlots, name)
    }
    
    fun applyAdagradDA(_var: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking: Boolean = false, name: String = "ApplyAdagradDA"): Output {
      return gen_training_ops.applyAdagradDA(_var, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep, useLocking, name)
    }
    
    fun applyAdam(_var: Output, m: Output, v: Output, beta1Power: Output, beta2Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, useNesterov: Boolean = false, name: String = "ApplyAdam"): Output {
      return gen_training_ops.applyAdam(_var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, useLocking, useNesterov, name)
    }
    
    fun applyAddSign(_var: Output, m: Output, lr: Output, alpha: Output, signDecay: Output, beta: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyAddSign"): Output {
      return gen_training_ops.applyAddSign(_var, m, lr, alpha, signDecay, beta, grad, useLocking, name)
    }
    
    fun applyCenteredRMSProp(_var: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyCenteredRMSProp"): Output {
      return gen_training_ops.applyCenteredRMSProp(_var, mg, ms, mom, lr, rho, momentum, epsilon, grad, useLocking, name)
    }
    
    fun applyFtrl(_var: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking: Boolean = false, name: String = "ApplyFtrl"): Output {
      return gen_training_ops.applyFtrl(_var, accum, linear, grad, lr, l1, l2, lrPower, useLocking, name)
    }
    
    fun applyFtrlV2(_var: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, useLocking: Boolean = false, name: String = "ApplyFtrlV2"): Output {
      return gen_training_ops.applyFtrlV2(_var, accum, linear, grad, lr, l1, l2, l2Shrinkage, lrPower, useLocking, name)
    }
    
    fun applyGradientDescent(_var: Output, alpha: Output, delta: Output, useLocking: Boolean = false, name: String = "ApplyGradientDescent"): Output {
      return gen_training_ops.applyGradientDescent(_var, alpha, delta, useLocking, name)
    }
    
    fun applyMomentum(_var: Output, accum: Output, lr: Output, grad: Output, momentum: Output, useLocking: Boolean = false, useNesterov: Boolean = false, name: String = "ApplyMomentum"): Output {
      return gen_training_ops.applyMomentum(_var, accum, lr, grad, momentum, useLocking, useNesterov, name)
    }
    
    fun applyPowerSign(_var: Output, m: Output, lr: Output, logbase: Output, signDecay: Output, beta: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyPowerSign"): Output {
      return gen_training_ops.applyPowerSign(_var, m, lr, logbase, signDecay, beta, grad, useLocking, name)
    }
    
    fun applyProximalAdagrad(_var: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyProximalAdagrad"): Output {
      return gen_training_ops.applyProximalAdagrad(_var, accum, lr, l1, l2, grad, useLocking, name)
    }
    
    fun applyProximalGradientDescent(_var: Output, alpha: Output, l1: Output, l2: Output, delta: Output, useLocking: Boolean = false, name: String = "ApplyProximalGradientDescent"): Output {
      return gen_training_ops.applyProximalGradientDescent(_var, alpha, l1, l2, delta, useLocking, name)
    }
    
    fun applyRMSProp(_var: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ApplyRMSProp"): Output {
      return gen_training_ops.applyRMSProp(_var, ms, mom, lr, rho, momentum, epsilon, grad, useLocking, name)
    }
    
    fun resourceApplyAdaMax(_var: Output, m: Output, v: Output, beta1Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyAdaMax"): Op {
      return gen_training_ops.resourceApplyAdaMax(_var, m, v, beta1Power, lr, beta1, beta2, epsilon, grad, useLocking, name)
    }
    
    fun resourceApplyAdadelta(_var: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyAdadelta"): Op {
      return gen_training_ops.resourceApplyAdadelta(_var, accum, accumUpdate, lr, rho, epsilon, grad, useLocking, name)
    }
    
    fun resourceApplyAdagrad(_var: Output, accum: Output, lr: Output, grad: Output, useLocking: Boolean = false, updateSlots: Boolean = true, name: String = "ResourceApplyAdagrad"): Op {
      return gen_training_ops.resourceApplyAdagrad(_var, accum, lr, grad, useLocking, updateSlots, name)
    }
    
    fun resourceApplyAdagradDA(_var: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking: Boolean = false, name: String = "ResourceApplyAdagradDA"): Op {
      return gen_training_ops.resourceApplyAdagradDA(_var, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep, useLocking, name)
    }
    
    fun resourceApplyAdam(_var: Output, m: Output, v: Output, beta1Power: Output, beta2Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, useNesterov: Boolean = false, name: String = "ResourceApplyAdam"): Op {
      return gen_training_ops.resourceApplyAdam(_var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, useLocking, useNesterov, name)
    }
    
    fun resourceApplyAddSign(_var: Output, m: Output, lr: Output, alpha: Output, signDecay: Output, beta: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyAddSign"): Op {
      return gen_training_ops.resourceApplyAddSign(_var, m, lr, alpha, signDecay, beta, grad, useLocking, name)
    }
    
    fun resourceApplyCenteredRMSProp(_var: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyCenteredRMSProp"): Op {
      return gen_training_ops.resourceApplyCenteredRMSProp(_var, mg, ms, mom, lr, rho, momentum, epsilon, grad, useLocking, name)
    }
    
    fun resourceApplyFtrl(_var: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking: Boolean = false, name: String = "ResourceApplyFtrl"): Op {
      return gen_training_ops.resourceApplyFtrl(_var, accum, linear, grad, lr, l1, l2, lrPower, useLocking, name)
    }
    
    fun resourceApplyFtrlV2(_var: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, useLocking: Boolean = false, name: String = "ResourceApplyFtrlV2"): Op {
      return gen_training_ops.resourceApplyFtrlV2(_var, accum, linear, grad, lr, l1, l2, l2Shrinkage, lrPower, useLocking, name)
    }
    
    fun resourceApplyGradientDescent(_var: Output, alpha: Output, delta: Output, useLocking: Boolean = false, name: String = "ResourceApplyGradientDescent"): Op {
      return gen_training_ops.resourceApplyGradientDescent(_var, alpha, delta, useLocking, name)
    }
    
    fun resourceApplyMomentum(_var: Output, accum: Output, lr: Output, grad: Output, momentum: Output, useLocking: Boolean = false, useNesterov: Boolean = false, name: String = "ResourceApplyMomentum"): Op {
      return gen_training_ops.resourceApplyMomentum(_var, accum, lr, grad, momentum, useLocking, useNesterov, name)
    }
    
    fun resourceApplyPowerSign(_var: Output, m: Output, lr: Output, logbase: Output, signDecay: Output, beta: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyPowerSign"): Op {
      return gen_training_ops.resourceApplyPowerSign(_var, m, lr, logbase, signDecay, beta, grad, useLocking, name)
    }
    
    fun resourceApplyProximalAdagrad(_var: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyProximalAdagrad"): Op {
      return gen_training_ops.resourceApplyProximalAdagrad(_var, accum, lr, l1, l2, grad, useLocking, name)
    }
    
    fun resourceApplyProximalGradientDescent(_var: Output, alpha: Output, l1: Output, l2: Output, delta: Output, useLocking: Boolean = false, name: String = "ResourceApplyProximalGradientDescent"): Op {
      return gen_training_ops.resourceApplyProximalGradientDescent(_var, alpha, l1, l2, delta, useLocking, name)
    }
    
    fun resourceApplyRMSProp(_var: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Boolean = false, name: String = "ResourceApplyRMSProp"): Op {
      return gen_training_ops.resourceApplyRMSProp(_var, ms, mom, lr, rho, momentum, epsilon, grad, useLocking, name)
    }
    
    fun resourceSparseApplyAdadelta(_var: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyAdadelta"): Op {
      return gen_training_ops.resourceSparseApplyAdadelta(_var, accum, accumUpdate, lr, rho, epsilon, grad, indices, useLocking, name)
    }
    
    fun resourceSparseApplyAdagrad(_var: Output, accum: Output, lr: Output, grad: Output, indices: Output, useLocking: Boolean = false, updateSlots: Boolean = true, name: String = "ResourceSparseApplyAdagrad"): Op {
      return gen_training_ops.resourceSparseApplyAdagrad(_var, accum, lr, grad, indices, useLocking, updateSlots, name)
    }
    
    fun resourceSparseApplyAdagradDA(_var: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyAdagradDA"): Op {
      return gen_training_ops.resourceSparseApplyAdagradDA(_var, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep, useLocking, name)
    }
    
    fun resourceSparseApplyCenteredRMSProp(_var: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyCenteredRMSProp"): Op {
      return gen_training_ops.resourceSparseApplyCenteredRMSProp(_var, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices, useLocking, name)
    }
    
    fun resourceSparseApplyFtrl(_var: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyFtrl"): Op {
      return gen_training_ops.resourceSparseApplyFtrl(_var, accum, linear, grad, indices, lr, l1, l2, lrPower, useLocking, name)
    }
    
    fun resourceSparseApplyFtrlV2(_var: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyFtrlV2"): Op {
      return gen_training_ops.resourceSparseApplyFtrlV2(_var, accum, linear, grad, indices, lr, l1, l2, l2Shrinkage, lrPower, useLocking, name)
    }
    
    fun resourceSparseApplyMomentum(_var: Output, accum: Output, lr: Output, grad: Output, indices: Output, momentum: Output, useLocking: Boolean = false, useNesterov: Boolean = false, name: String = "ResourceSparseApplyMomentum"): Op {
      return gen_training_ops.resourceSparseApplyMomentum(_var, accum, lr, grad, indices, momentum, useLocking, useNesterov, name)
    }
    
    fun resourceSparseApplyProximalAdagrad(_var: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyProximalAdagrad"): Op {
      return gen_training_ops.resourceSparseApplyProximalAdagrad(_var, accum, lr, l1, l2, grad, indices, useLocking, name)
    }
    
    fun resourceSparseApplyProximalGradientDescent(_var: Output, alpha: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyProximalGradientDescent"): Op {
      return gen_training_ops.resourceSparseApplyProximalGradientDescent(_var, alpha, l1, l2, grad, indices, useLocking, name)
    }
    
    fun resourceSparseApplyRMSProp(_var: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "ResourceSparseApplyRMSProp"): Op {
      return gen_training_ops.resourceSparseApplyRMSProp(_var, ms, mom, lr, rho, momentum, epsilon, grad, indices, useLocking, name)
    }
    
    fun sparseApplyAdadelta(_var: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "SparseApplyAdadelta"): Output {
      return gen_training_ops.sparseApplyAdadelta(_var, accum, accumUpdate, lr, rho, epsilon, grad, indices, useLocking, name)
    }
    
    fun sparseApplyAdagrad(_var: Output, accum: Output, lr: Output, grad: Output, indices: Output, useLocking: Boolean = false, updateSlots: Boolean = true, name: String = "SparseApplyAdagrad"): Output {
      return gen_training_ops.sparseApplyAdagrad(_var, accum, lr, grad, indices, useLocking, updateSlots, name)
    }
    
    fun sparseApplyAdagradDA(_var: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking: Boolean = false, name: String = "SparseApplyAdagradDA"): Output {
      return gen_training_ops.sparseApplyAdagradDA(_var, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep, useLocking, name)
    }
    
    fun sparseApplyCenteredRMSProp(_var: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "SparseApplyCenteredRMSProp"): Output {
      return gen_training_ops.sparseApplyCenteredRMSProp(_var, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices, useLocking, name)
    }
    
    fun sparseApplyFtrl(_var: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking: Boolean = false, name: String = "SparseApplyFtrl"): Output {
      return gen_training_ops.sparseApplyFtrl(_var, accum, linear, grad, indices, lr, l1, l2, lrPower, useLocking, name)
    }
    
    fun sparseApplyFtrlV2(_var: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, useLocking: Boolean = false, name: String = "SparseApplyFtrlV2"): Output {
      return gen_training_ops.sparseApplyFtrlV2(_var, accum, linear, grad, indices, lr, l1, l2, l2Shrinkage, lrPower, useLocking, name)
    }
    
    fun sparseApplyMomentum(_var: Output, accum: Output, lr: Output, grad: Output, indices: Output, momentum: Output, useLocking: Boolean = false, useNesterov: Boolean = false, name: String = "SparseApplyMomentum"): Output {
      return gen_training_ops.sparseApplyMomentum(_var, accum, lr, grad, indices, momentum, useLocking, useNesterov, name)
    }
    
    fun sparseApplyProximalAdagrad(_var: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "SparseApplyProximalAdagrad"): Output {
      return gen_training_ops.sparseApplyProximalAdagrad(_var, accum, lr, l1, l2, grad, indices, useLocking, name)
    }
    
    fun sparseApplyProximalGradientDescent(_var: Output, alpha: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "SparseApplyProximalGradientDescent"): Output {
      return gen_training_ops.sparseApplyProximalGradientDescent(_var, alpha, l1, l2, grad, indices, useLocking, name)
    }
    
    fun sparseApplyRMSProp(_var: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking: Boolean = false, name: String = "SparseApplyRMSProp"): Output {
      return gen_training_ops.sparseApplyRMSProp(_var, ms, mom, lr, rho, momentum, epsilon, grad, indices, useLocking, name)
    }
  }
}