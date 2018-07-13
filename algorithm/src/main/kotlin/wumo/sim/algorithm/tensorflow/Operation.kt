package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.tensorflow.*

class Operation(val graph: Graph, val c_op: TF_Operation) {
  /**
   * Update the input to this operation at the given index.
   * NOTE: This is for TF internal use only. Please don't use it.
   *
   * @param i the index of the input to update.
   * @param tensor the Tensor to be used as the input at the given index.
   * @param update_dtype If `False`, the type for this input is not updated.
   */
  internal fun update_input(i: Int, tensor: Tensor, update_dtype: Boolean = true) {
    TODO("not implemented")
  }
  
  val name = TF_OperationName(c_op).string
  val device = TF_OperationDevice(c_op).string
  val opType = TF_OperationOpType(c_op).string
  val inputs: List<Tensor>
    get() {
      val numInputs = TF_OperationNumInputs(c_op)
      
      return List(numInputs) {
        Tensor(Operation(graph, c_op), it, isInput = true)
      }
    }
  val outputs: List<Tensor>
    get() {
      val numOutputs = TF_OperationNumOutputs(c_op)
      
      return List(numOutputs) {
        Tensor(Operation(graph, c_op), it)
      }
    }
  val control_inputs: List<Operation>
    get() {
      val numControlOps = TF_OperationNumControlInputs(c_op)
      val control_ops = PointerPointer<TF_Operation>(numControlOps.toLong())
      TF_OperationGetControlInputs(c_op, control_ops, numControlOps)
      return List(numControlOps) {
        Operation(graph, control_ops.get(TF_Operation::class.java, it.toLong()))
      }
    }
  val input_types: List<Int>
    get() {
      val numOutputs = TF_OperationNumInputs(c_op)
      
      return List(numOutputs) {
        TF_OperationInputType(TF_Input().oper(c_op).index(it))
      }
    }
  val output_types: List<Int>
    get() {
      val numOutputs = TF_OperationNumOutputs(c_op)
      
      return List(numOutputs) {
        TF_OperationOutputType(TF_Output().oper(c_op).index(it))
      }
    }
  val attr: Map<String, Any>
    get() {
      TODO()
    }
}