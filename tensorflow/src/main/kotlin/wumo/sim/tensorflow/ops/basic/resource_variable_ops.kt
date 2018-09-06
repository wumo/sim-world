package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_resource_variable_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.Shape

object resource_variable_ops {
  interface API {
    fun assignAddVariableOp(resource: Output, value: Output, name: String = "AssignAddVariableOp"): Op {
      return gen_resource_variable_ops.assignAddVariableOp(resource, value, name)
    }
    
    fun assignSubVariableOp(resource: Output, value: Output, name: String = "AssignSubVariableOp"): Op {
      return gen_resource_variable_ops.assignSubVariableOp(resource, value, name)
    }
    
    fun assignVariableOp(resource: Output, value: Output, name: String = "AssignVariableOp"): Op {
      return gen_resource_variable_ops.assignVariableOp(resource, value, name)
    }
    
    fun consumeMutexLock(mutexLock: Output, name: String = "ConsumeMutexLock"): Op {
      return gen_resource_variable_ops.consumeMutexLock(mutexLock, name)
    }
    
    fun destroyResourceOp(resource: Output, ignoreLookupError: Boolean = true, name: String = "DestroyResourceOp"): Op {
      return gen_resource_variable_ops.destroyResourceOp(resource, ignoreLookupError, name)
    }
    
    fun mutexLock(mutex: Output, name: String = "MutexLock"): Output {
      return gen_resource_variable_ops.mutexLock(mutex, name)
    }
    
    fun mutexV2(container: String = "", sharedName: String = "", name: String = "MutexV2"): Output {
      return gen_resource_variable_ops.mutexV2(container, sharedName, name)
    }
    
    fun readVariableOp(resource: Output, dtype: DataType<*>, name: String = "ReadVariableOp"): Output {
      return gen_resource_variable_ops.readVariableOp(resource, dtype, name)
    }
    
    fun resourceGather(resource: Output, indices: Output, dtype: DataType<*>, validateIndices: Boolean = true, name: String = "ResourceGather"): Output {
      return gen_resource_variable_ops.resourceGather(resource, indices, dtype, validateIndices, name)
    }
    
    fun resourceScatterAdd(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterAdd"): Op {
      return gen_resource_variable_ops.resourceScatterAdd(resource, indices, updates, name)
    }
    
    fun resourceScatterDiv(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterDiv"): Op {
      return gen_resource_variable_ops.resourceScatterDiv(resource, indices, updates, name)
    }
    
    fun resourceScatterMax(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterMax"): Op {
      return gen_resource_variable_ops.resourceScatterMax(resource, indices, updates, name)
    }
    
    fun resourceScatterMin(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterMin"): Op {
      return gen_resource_variable_ops.resourceScatterMin(resource, indices, updates, name)
    }
    
    fun resourceScatterMul(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterMul"): Op {
      return gen_resource_variable_ops.resourceScatterMul(resource, indices, updates, name)
    }
    
    fun resourceScatterSub(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterSub"): Op {
      return gen_resource_variable_ops.resourceScatterSub(resource, indices, updates, name)
    }
    
    fun resourceScatterUpdate(resource: Output, indices: Output, updates: Output, name: String = "ResourceScatterUpdate"): Op {
      return gen_resource_variable_ops.resourceScatterUpdate(resource, indices, updates, name)
    }
    
    fun varHandleOp(dtype: DataType<*>, shape: Shape, container: String = "", sharedName: String = "", name: String = "VarHandleOp"): Output {
      return gen_resource_variable_ops.varHandleOp(dtype, shape, container, sharedName, name)
    }
    
    fun varIsInitializedOp(resource: Output, name: String = "VarIsInitializedOp"): Output {
      return gen_resource_variable_ops.varIsInitializedOp(resource, name)
    }
    
    fun variableShape(input: Output, outType: DataType<*> = INT32, name: String = "VariableShape"): Output {
      return gen_resource_variable_ops.variableShape(input, outType, name)
    }
  }
}