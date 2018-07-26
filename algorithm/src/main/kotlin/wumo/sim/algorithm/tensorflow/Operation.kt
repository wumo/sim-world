package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.ops.gradients.iterate
import wumo.sim.algorithm.tensorflow.ops.gradients.toTensor

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
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (javaClass != other?.javaClass) return false
    
    other as Operation
    if (c_op != other.c_op) return false
    return true
  }
  
  override fun hashCode(): Int {
    val result = c_op.hashCode()
    return result
  }
  
  val name: String by lazy { TF_OperationName(c_op).string }
  val device: String by lazy { TF_OperationDevice(c_op).string }
  val opType: String by lazy { TF_OperationOpType(c_op).string }
  val node: Node by lazy { c_op.node() }
  val attrs: AttrSlice by lazy { node.attrs() }
  
  val inputs: List<Tensor> by lazy {
    val node = c_op.node()
    val numInputs = node.num_inputs()
    val inputs = MutableList(numInputs) { Tensor(null, it) }
    for (e in node.in_edges().iterate()) {
      if (e.IsControlEdge()) continue
      inputs[e.dst_input()] = toTensor(e.src(), e.src_output())
    }
    inputs
  }
  
  val outputs: List<Tensor> by lazy {
    val numOutputs = TF_OperationNumOutputs(c_op)
    
    List(numOutputs) {
      Tensor(Operation(graph, c_op), it)
    }
  }
  
  val control_inputs: List<Operation> by lazy {
    val numControlOps = TF_OperationNumControlInputs(c_op)
    val control_ops = PointerPointer<TF_Operation>(numControlOps.toLong())
    TF_OperationGetControlInputs(c_op, control_ops, numControlOps)
    List(numControlOps) {
      Operation(graph, control_ops.get(TF_Operation::class.java, it.toLong()))
    }
  }
  
  val input_types: List<Int> by lazy {
    val numOutputs = TF_OperationNumInputs(c_op)
    
    List(numOutputs) {
      TF_OperationInputType(TF_Input().oper(c_op).index(it))
    }
  }
  
  val output_types: List<Int> by lazy {
    val numOutputs = TF_OperationNumOutputs(c_op)
    
    List(numOutputs) {
      TF_OperationOutputType(TF_Output().oper(c_op).index(it))
    }
  }
  
  fun attrString(name: String): String {
    val value = attrs.Find(name)
    return value.s().string
  }
  
  fun attrBool(name: String): Boolean {
    val value = attrs.Find(name)
    return value.b()
  }
  
  fun attrBoolList(name: String): BooleanArray {
    val value = attrs.Find(name)
    val list_value = value.list()
    return BooleanArray(list_value.i_size()) {
      list_value.b(it)
    }
  }
  
  fun attrLong(name: String): Long {
    val value = attrs.Find(name)
    return value.i()
  }
  
  fun attrLongList(name: String): LongArray {
    val value = attrs.Find(name)
    val list_value = value.list()
    return LongArray(list_value.i_size()) {
      list_value.i(it)
    }
  }
  
  fun attrFloat(name: String): Float {
    val value = attrs.Find(name)
    return value.f()
  }
  
  fun attrFloatList(name: String): FloatArray {
    val value = attrs.Find(name)
    val list_value = value.list()
    return FloatArray(list_value.i_size()) {
      list_value.f(it)
    }
  }
  
  val attr: Map<String, Any>
    get() {
      TODO()
    }
  
  override fun toString(): String {
    return """"Operation("$name", op=$opType, dev=$device)"""
  }
}