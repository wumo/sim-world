package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.util.Dimension

interface TensorLike

/**
 * Represents one of the outputs of an `Op`.
 *
 * `Tensor` is a symbolic handle to one of the outputs of an
 * `Op`. It does not hold the values of that findOp's output,
 * but instead provides a means of computing those values in a
 * TensorFlow [Session].
 *
 * This class has two primary purposes:
 * 1. A `Tensor` can be passed as an input to another `Op`.
 * This builds a dataflow connection between operations, which
 * enables TensorFlow to execute an entire `Graph` that represents a
 * large, multi-step computation.
 *
 * 2. After the graph has been launched in a session, the value of the
 * `Tensor` can be computed by passing it to@{tf.Session.run}.
 * `t.eval()` is a shortcut for calling`tf.get_default_session().run(t)`.
 */
open class Tensor(val op: Op?, val value_index: Int) : TensorLike {
  val dtype: Int
    get() = if (op != null) {
      op.output_types[value_index]
    } else DT_INVALID
  
  val shape: Dimension
    get() {
      val c_graph = op!!.graph.c_graph
      val output = asTF_Output()
      val status = newStatus()
      val numDims = TF_GraphGetTensorNumDims(c_graph, output, status)
      throwExceptionIfNotOk(status)
      val dims = LongArray(numDims)
      TF_GraphGetTensorShape(c_graph, output, dims, numDims, status)
      throwExceptionIfNotOk(status)
      return Dimension(dims)
    }
  
  /**
   * Updates the shape of this tensor.
  
  This method can be called multiple times, and will merge the given
  `shape` with the current shape of this tensor. It can be used to
  provide additional information about the shape of this tensor that
  cannot be inferred from the graph alone. For example, this can be used
  to provide additional information about the shapes of images:
  
  ```python
  _, image_data = tf.TFRecordReader(...).read(...)
  image = tf.image.decode_png(image_data, channels=3)
  
  # The height and width dimensions of `image` are data dependent, and
  # cannot be computed without executing the op.
  print(image.shape)
  ==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])
  
  # We know that each image in this dataset is 28 x 28 pixels.
  image.set_shape([28, 28, 3])
  print(image.shape)
  ==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
  ```
   
   * @param shape: A `TensorShape` representing the shape of this tensor, a
  `TensorShapeProto`, a list, a tuple, or None.
  
  Raises:
  ValueError: If `shape` is not compatible with the current shape of
  this tensor.
   */
  fun set_shape(shape: Dimension) {
    assert(this.shape.isCompatibleWith(shape))
    op!!
    val dims = shape.asLongArray()
    val status = newStatus()
    TF_GraphSetTensorShape(op.graph.c_graph, asTF_Output(), dims, dims.size, status)
    throwExceptionIfNotOk(status)
  }
  
  val tf: TF by lazy { op!!.graph.tf }
  
  fun asTF_Output() = TF_Output().oper(op!!.c_op).index(value_index)
  val name: String by lazy { "${op!!.name}:$value_index" }
//  val name: String by lazy { op!!.name }
  
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
  
  open fun asRef(): Tensor = this
  
  override fun toString() =
      when (op) {
        null -> "Tensor(null)"
        else -> """Tensor("$name", shape=$shape, dtype=${dtype.name()}, op=$op)"""
      }
}