package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.util.Dimension

interface TensorLike

/**
 * Represents one of the outputs of an `Operation`.
 *
 * `Tensor` is a symbolic handle to one of the outputs of an
 * `Operation`. It does not hold the values of that operation's output,
 * but instead provides a means of computing those values in a
 * TensorFlow [Session].
 *
 * This class has two primary purposes:
 * 1. A `Tensor` can be passed as an input to another `Operation`.
 * This builds a dataflow connection between operations, which
 * enables TensorFlow to execute an entire `Graph` that represents a
 * large, multi-step computation.
 *
 * 2. After the graph has been launched in a session, the value of the
 * `Tensor` can be computed by passing it to@{tf.Session.run}.
 * `t.eval()` is a shortcut for calling`tf.get_default_session().run(t)`.
 */
open class Tensor(val op: Operation?, val value_index: Int) : TensorLike {
  val dtype: Int
    get() = if (op != null) {
      op.output_types[value_index]
    } else DT_INVALID
  
  val shape: Dimension by lazy {
    val c_graph = op!!.graph.c_graph
    val output = asTF_Output()
    val status = newStatus()
    val numDims = TF_GraphGetTensorNumDims(c_graph, output, status)
    throwExceptionIfNotOk(status)
    val dims = LongArray(numDims)
    TF_GraphGetTensorShape(c_graph, output, dims, numDims, status)
    throwExceptionIfNotOk(status)
    Dimension(dims)
  }
  
  val tf: TF by lazy { op!!.graph.tf }
  
  fun asTF_Output() = TF_Output().oper(op!!.c_op).index(value_index)
  val name: String by lazy { op!!.name }
  
  inline fun node() = op!!.c_op.node()
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is Tensor) return false
    
    if (op != other.op) return false
    if (value_index != other.value_index) return false
    if (dtype != other.dtype) return false
    
    return true
  }
  
  override fun hashCode(): Int {
    if (op == null) return 0
    var result = op.hashCode()
    result = 31 * result + value_index
    return result
  }
  
  /**
   * 主要用于[Variable]输出的tensor并不是自己，而是[Variable.snapshot]
   */
  open fun value() = this
  
  open fun asRef(): Tensor = throw UnsupportedOperationException("This tensor is not mutable!")
  
  override fun toString() =
      when (op) {
        null -> "Tensor(null)"
        else -> """Tensor("$name:$value_index", shape=$shape, dtype=${dtype.name()}, op=${op.opType})"""
      }
}