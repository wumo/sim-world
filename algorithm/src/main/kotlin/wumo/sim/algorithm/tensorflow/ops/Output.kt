package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Graph
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.name
import wumo.sim.algorithm.tensorflow.throwExceptionIfNotOk
import wumo.sim.util.Shape

interface OutputConvertible {
  fun toTensor(): Output
}

sealed class OutputLike : OutputConvertible {
  abstract val graph: Graph
  abstract val name: String
  abstract val dtype: Int
  abstract val device: String
  abstract val op: Op?
  abstract val consumers: Array<Input>
}

class IndexedSlices : OutputLike() {
  override val name: String
    get() = TODO("not implemented")
  override val dtype: Int
    get() = TODO("not implemented")
  override val device: String
    get() = TODO("not implemented")
  override val op: Op?
    get() = TODO("not implemented")
  override val consumers: Array<Input>
    get() = TODO("not implemented")
  override val graph: Graph
    get() = TODO("not implemented")
  
  override fun toTensor(): Output {
    TODO("not implemented")
  }
}

class SparseOutput : OutputLike() {
  override val name: String
    get() = TODO("not implemented")
  override val dtype: Int
    get() = TODO("not implemented")
  override val device: String
    get() = TODO("not implemented")
  override val op: Op?
    get() = TODO("not implemented")
  override val consumers: Array<Input>
    get() = TODO("not implemented")
  override val graph: Graph
    get() = TODO("not implemented")
  
  override fun toTensor(): Output {
    TODO("not implemented")
  }
}

/**
 * Represents one of the outputs of an [Op].
 *
 * A [Output] is a symbolic handle to one of the outputs of an
 * [Op]. It does not hold the values of that findOp's output,
 * but instead provides a means of computing those values in a
 * TensorFlow [Session].
 *
 * This class has two primary purposes:
 *
 *  1. A [Output] can be passed as an input to another [Op].
 *  This builds a dataflow connection between operations, which
 *  enables TensorFlow to execute an entire [Graph] that represents a
 *  large, multi-step computation.
 *
 *  2. After the graph has been launched in a [Session], the value of the
 *  `Output` can be computed by passing it to@{tf.Session.run}.
 *  `t.eval()` is a shortcut for calling`tf.get_default_session().run(t)`.
 *
 * In the following example, `c`, `d`, and `e` are symbolic [Output] objects,
 * whereas `result` is a [NDArray][wumo.sim.util.ndarray.NDArray] that stores a concrete value:
 *
 * ```
 * // Build a dataflow graph.
 * val c = tf.const([[1.0, 2.0], [3.0, 4.0]])
 * val d = tf.const([[1.0, 1.0], [0.0, 1.0]])
 * val e = tf.matmul(c, d)
 *
 * //Construct a `Session` to execute the graph.
 * val sess = tf.Session()
 *
 * //Execute the graph and store the value that `e` represents in `result`.
 * val result = sess.eval(e)
 * ```
 *
 * @param[op] [Op] that computes this tensor.
 * @param[value_index] Index of the operation's endpoint that produces this tensor.
 */
class Output(override val op: Op?, val value_index: Int) : OutputLike() {
  override val graph = op!!.graph
  override val device = op!!.device
  override val consumers: Array<Input>
    get() = TODO("not implemented")
  
  override fun toTensor() = this
  
  override val dtype: Int
    get() = if (op != null) {
      op.output_types[value_index]
    } else DT_INVALID
  
  val shape: Shape
    get() {
      val c_graph = op!!.graph.c_graph
      val output = asTF_Output()
      val status = newStatus()
      val numDims = TF_GraphGetTensorNumDims(c_graph, output, status)
      if (numDims < 0) return Shape(unknow_rank = true)
      throwExceptionIfNotOk(status)
      val dims = LongArray(numDims)
      TF_GraphGetTensorShape(c_graph, output, dims, numDims, status)
      throwExceptionIfNotOk(status)
      return Shape(dims)
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
  ==> TensorShape([Shape(None), Shape(None), Shape(3)])
  
  # We know that each image in this dataset is 28 x 28 pixels.
  image.set_shape([28, 28, 3])
  print(image.shape)
  ==> TensorShape([Shape(28), Shape(28), Shape(3)])
  ```
   
   * @param shape: A `TensorShape` representing the shape of this tensor, a
  `TensorShapeProto`, a list, a tuple, or None.
  
  Raises:
  ValueError: If `shape` is not compatible with the current shape of
  this tensor.
   */
  fun set_shape(shape: Shape) {
    assert(this.shape.isCompatibleWith(shape))
    op!!
    val dims = shape.asLongArray()
    val status = newStatus()
    TF_GraphSetTensorShape(op.graph.c_graph, asTF_Output(), dims, dims.size, status)
    throwExceptionIfNotOk(status)
  }
  
  val tf: TF by lazy { op!!.graph.tf }
  
  fun asTF_Output() = TF_Output().oper(op!!.c_op).index(value_index)
  override val name: String by lazy { "${op!!.name}:$value_index" }
//  val name: String by lazy { op!!.name }
  
  inline fun node() = op!!.c_op.node()
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is Output) return false
    
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
  
  open fun asRef(): Output = this
  
  override fun toString() =
      when (op) {
        null -> "Output(null)"
        else -> """Output("$name", shape=$shape, dtype=${dtype.name()}, op=$op)"""
      }
}