/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.buildOp
import wumo.sim.algorithm.tensorflow.buildOpTensor
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension

object gen_resource_variable_ops {
  fun assignAddVariableOp(resource: Tensor, value: Tensor, name: String = "AssignAddVariableOp") = run {
    tf.buildOp("AssignAddVariableOp", name) {
      addInput(resource, false)
      addInput(value, false)
    }
  }
  
  fun assignSubVariableOp(resource: Tensor, value: Tensor, name: String = "AssignSubVariableOp") = run {
    tf.buildOp("AssignSubVariableOp", name) {
      addInput(resource, false)
      addInput(value, false)
    }
  }
  
  fun assignVariableOp(resource: Tensor, value: Tensor, name: String = "AssignVariableOp") = run {
    tf.buildOp("AssignVariableOp", name) {
      addInput(resource, false)
      addInput(value, false)
    }
  }
  
  fun consumeMutexLock(mutex_lock: Tensor, name: String = "ConsumeMutexLock") = run {
    tf.buildOp("ConsumeMutexLock", name) {
      addInput(mutex_lock, false)
    }
  }
  
  fun destroyResourceOp(resource: Tensor, ignore_lookup_error: Boolean = true, name: String = "DestroyResourceOp") = run {
    tf.buildOp("DestroyResourceOp", name) {
      addInput(resource, false)
      attr("ignore_lookup_error", ignore_lookup_error)
    }
  }
  
  fun mutexLock(mutex: Tensor, name: String = "MutexLock") = run {
    tf.buildOpTensor("MutexLock", name) {
      addInput(mutex, false)
    }
  }
  
  fun mutexV2(container: String = "", shared_name: String = "", name: String = "MutexV2") = run {
    tf.buildOpTensor("MutexV2", name) {
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun readVariableOp(resource: Tensor, dtype: Int, name: String = "ReadVariableOp") = run {
    tf.buildOpTensor("ReadVariableOp", name) {
      addInput(resource, false)
      attrType("dtype", dtype)
    }
  }
  
  fun resourceGather(resource: Tensor, indices: Tensor, validate_indices: Boolean = true, dtype: Int, name: String = "ResourceGather") = run {
    tf.buildOpTensor("ResourceGather", name) {
      addInput(resource, false)
      addInput(indices, false)
      attr("validate_indices", validate_indices)
      attrType("dtype", dtype)
    }
  }
  
  fun resourceScatterAdd(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterAdd") = run {
    tf.buildOp("ResourceScatterAdd", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun resourceScatterDiv(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterDiv") = run {
    tf.buildOp("ResourceScatterDiv", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun resourceScatterMax(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterMax") = run {
    tf.buildOp("ResourceScatterMax", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun resourceScatterMin(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterMin") = run {
    tf.buildOp("ResourceScatterMin", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun resourceScatterMul(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterMul") = run {
    tf.buildOp("ResourceScatterMul", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun resourceScatterSub(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterSub") = run {
    tf.buildOp("ResourceScatterSub", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun resourceScatterUpdate(resource: Tensor, indices: Tensor, updates: Tensor, name: String = "ResourceScatterUpdate") = run {
    tf.buildOp("ResourceScatterUpdate", name) {
      addInput(resource, false)
      addInput(indices, false)
      addInput(updates, false)
    }
  }
  
  fun varHandleOp(container: String = "", shared_name: String = "", dtype: Int, shape: Dimension, name: String = "VarHandleOp") = run {
    tf.buildOpTensor("VarHandleOp", name) {
      attr("container", container)
      attr("shared_name", shared_name)
      attrType("dtype", dtype)
      attr("shape", shape)
    }
  }
  
  fun varIsInitializedOp(resource: Tensor, name: String = "VarIsInitializedOp") = run {
    tf.buildOpTensor("VarIsInitializedOp", name) {
      addInput(resource, false)
    }
  }
  
  fun variableShape(input: Tensor, out_type: Int = 3, name: String = "VariableShape") = run {
    tf.buildOpTensor("VariableShape", name) {
      addInput(input, false)
      attrType("out_type", out_type)
    }
  }
}