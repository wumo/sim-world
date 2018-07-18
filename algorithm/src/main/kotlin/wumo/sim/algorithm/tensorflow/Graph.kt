package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.helper.tensorflow
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Graph.newGraph
import org.bytedeco.javacpp.tensorflow.*

/**
 * A TensorFlow computation, represented as a dataflow graph.
 *
 * A `Graph` contains a set of [Operation] objects,
 * which represent units of computation; and
 * [Tensor] objects, which represent
 * the units of data that flow between operations.
 */
class Graph(val tf: TF) {
  val c_graph = newGraph()!!
  
  /**Set of tensors that are dangerous to feed!*/
  private val unfeedable_tensors = mutableSetOf<Tensor>()
  /**Set of operations that are dangerous to fetch!*/
  private val unfetchable_ops = mutableSetOf<Operation>()
  
  fun nodeBuilder(opType: String, name: String) = OperationBuilder(this, opType, name)
  fun operation(name: String): Operation {
    val op = TF_GraphOperationByName(c_graph, name)
    return Operation(this, op)
  }
  
  fun toGraphDef(): ByteArray {
    val buf = TF_NewBuffer()
    val status = TF_NewStatus()
    TF_GraphToGraphDef(c_graph, buf, status)
    throwExceptionIfNotOk(status)
    val len = buf.length()
    val bytes = ByteArray(len.toInt())
    val d = buf.data()
    d.capacity<Pointer>(len)
    val data = d.asByteBuffer()
    data.get(bytes)
    TF_DeleteStatus(status)
    TF_DeleteBuffer(buf)
    return bytes
  }
  
  fun create_op(new_op_type: String,
                new_op_inputs: MutableList<Tensor>,
                output_types: List<Int>,
                name: String,
                attrs: Map<String, Any>): Operation {
    TODO("not implemented")
  }
  
  fun prevent_fetching(op: Operation) {
    unfetchable_ops += op
  }
}