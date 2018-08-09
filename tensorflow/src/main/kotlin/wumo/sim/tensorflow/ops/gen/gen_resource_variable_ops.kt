/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.DT_INT32
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.util.Shape

fun TF.assignAddVariableOp(resource: Output, value: Output, name: String = "AssignAddVariableOp") = run {
  buildOp("AssignAddVariableOp", name) {
    addInput(resource, false)
    addInput(value, false)
  }
}

fun TF.assignSubVariableOp(resource: Output, value: Output, name: String = "AssignSubVariableOp") = run {
  buildOp("AssignSubVariableOp", name) {
    addInput(resource, false)
    addInput(value, false)
  }
}

fun TF.assignVariableOp(resource: Output, value: Output, name: String = "AssignVariableOp") = run {
  buildOp("AssignVariableOp", name) {
    addInput(resource, false)
    addInput(value, false)
  }
}

fun TF.consumeMutexLock(mutex_lock: Output, name: String = "ConsumeMutexLock") = run {
  buildOp("ConsumeMutexLock", name) {
    addInput(mutex_lock, false)
  }
}

fun TF.destroyResourceOp(resource: Output, ignore_lookup_error: Boolean = true, name: String = "DestroyResourceOp") = run {
  buildOp("DestroyResourceOp", name) {
    addInput(resource, false)
    attr("ignore_lookup_error", ignore_lookup_error)
  }
}

fun TF.mutexLock(mutex: Output, name: String = "MutexLock") = run {
  buildOpTensor("MutexLock", name) {
    addInput(mutex, false)
  }
}

fun TF.mutexV2(container: String = "", shared_name: String = "", name: String = "MutexV2") = run {
  buildOpTensor("MutexV2", name) {
    attr("container", container)
    attr("shared_name", shared_name)
  }
}

fun TF.readVariableOp(resource: Output, dtype: Int, name: String = "ReadVariableOp") = run {
  buildOpTensor("ReadVariableOp", name) {
    addInput(resource, false)
    attrType("dtype", dtype)
  }
}

fun TF.resourceGather(resource: Output, indices: Output, dtype: Int, validate_indices: Boolean = true, name: String = "ResourceGather") = run {
  buildOpTensor("ResourceGather", name) {
    addInput(resource, false)
    addInput(indices, false)
    attrType("dtype", dtype)
    attr("validate_indices", validate_indices)
  }
}

fun TF.resourceScatterAdd(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterAdd") = run {
  buildOp("ResourceScatterAdd", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.resourceScatterDiv(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterDiv") = run {
  buildOp("ResourceScatterDiv", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.resourceScatterMax(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterMax") = run {
  buildOp("ResourceScatterMax", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.resourceScatterMin(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterMin") = run {
  buildOp("ResourceScatterMin", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.resourceScatterMul(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterMul") = run {
  buildOp("ResourceScatterMul", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.resourceScatterSub(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterSub") = run {
  buildOp("ResourceScatterSub", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.resourceScatterUpdate(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterUpdate") = run {
  buildOp("ResourceScatterUpdate", name) {
    addInput(resource, false)
    addInput(indices, false)
    addInput(updates, false)
  }
}

fun TF.varHandleOp(dtype: Int, shape: Shape, container: String = "", shared_name: String = "", name: String = "VarHandleOp") = run {
  buildOpTensor("VarHandleOp", name) {
    attrType("dtype", dtype)
    attr("shape", shape)
    attr("container", container)
    attr("shared_name", shared_name)
  }
}

fun TF.varIsInitializedOp(resource: Output, name: String = "VarIsInitializedOp") = run {
  buildOpTensor("VarIsInitializedOp", name) {
    addInput(resource, false)
  }
}

fun TF.variableShape(input: Output, out_type: Int = DT_INT32, name: String = "VariableShape") = run {
  buildOpTensor("VariableShape", name) {
    addInput(input, false)
    attrType("out_type", out_type)
  }
}
