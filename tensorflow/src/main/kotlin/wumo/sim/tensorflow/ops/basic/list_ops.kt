package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_list_ops
import wumo.sim.tensorflow.types.DataType

object list_ops {
  interface API {
    fun emptyTensorList(elementShape: Output, elementDtype: DataType<*>, name: String = "EmptyTensorList"): Output {
      return gen_list_ops.emptyTensorList(elementShape, elementDtype, name)
    }
    
    fun tensorListConcatLists(inputA: Output, inputB: Output, elementDtype: DataType<*>, name: String = "TensorListConcatLists"): Output {
      return gen_list_ops.tensorListConcatLists(inputA, inputB, elementDtype, name)
    }
    
    fun tensorListElementShape(inputHandle: Output, shapeType: DataType<*>, name: String = "TensorListElementShape"): Output {
      return gen_list_ops.tensorListElementShape(inputHandle, shapeType, name)
    }
    
    fun tensorListFromTensor(tensor: Output, elementShape: Output, name: String = "TensorListFromTensor"): Output {
      return gen_list_ops.tensorListFromTensor(tensor, elementShape, name)
    }
    
    fun tensorListGetItem(inputHandle: Output, index: Output, elementDtype: DataType<*>, name: String = "TensorListGetItem"): Output {
      return gen_list_ops.tensorListGetItem(inputHandle, index, elementDtype, name)
    }
    
    fun tensorListLength(inputHandle: Output, name: String = "TensorListLength"): Output {
      return gen_list_ops.tensorListLength(inputHandle, name)
    }
    
    fun tensorListPopBack(inputHandle: Output, elementDtype: DataType<*>, name: String = "TensorListPopBack"): List<Output> {
      return gen_list_ops.tensorListPopBack(inputHandle, elementDtype, name)
    }
    
    fun tensorListPushBack(inputHandle: Output, tensor: Output, name: String = "TensorListPushBack"): Output {
      return gen_list_ops.tensorListPushBack(inputHandle, tensor, name)
    }
    
    fun tensorListPushBackBatch(inputHandles: Output, tensor: Output, name: String = "TensorListPushBackBatch"): Output {
      return gen_list_ops.tensorListPushBackBatch(inputHandles, tensor, name)
    }
    
    fun tensorListReserve(elementShape: Output, numElements: Output, elementDtype: DataType<*>, name: String = "TensorListReserve"): Output {
      return gen_list_ops.tensorListReserve(elementShape, numElements, elementDtype, name)
    }
    
    fun tensorListSetItem(inputHandle: Output, index: Output, item: Output, name: String = "TensorListSetItem"): Output {
      return gen_list_ops.tensorListSetItem(inputHandle, index, item, name)
    }
    
    fun tensorListStack(inputHandle: Output, elementDtype: DataType<*>, numElements: Long = -1L, name: String = "TensorListStack"): Output {
      return gen_list_ops.tensorListStack(inputHandle, elementDtype, numElements, name)
    }
  }
}