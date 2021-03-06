package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.core.check
import wumo.sim.tensorflow.ops.basic.get
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape
import wumo.sim.util.warn

interface OutputConvertible {
  fun toOutput(): Output
}

sealed class OutputLike : OutputConvertible, HasName {
  abstract val graph: Graph
  abstract val dataType: DataType<*>
  abstract val device: String
  abstract val op: Op
  abstract val consumers: List<Op>
}

val LARGE_SPARSE_NUM_ELEMENTS = 100000000

/** Sparse representation of a set of tensor slices at given indices.
 *
 * This class if a simple wrapper for a pair (or a set of three) of [[Output]] objects:
 *   - `indices`: A one-dimensional integer [[Output]] with shape `[D0]`.
 *   - `values`: An [[Output]] of any data type, with shape `[D0, D1, ..., Dn]`.
 *   - `denseShape`: Optionally, an integer [[Output]] with shape `[LARGE0, D1, ..., Dn]`.
 *
 * An [IndexedSlices] is typically used to represent a subset of a larger [[Output]], `dense`, of shape
 * `[LARGE0, D1, ..., Dn]`, where `LARGE0 >> D0`. The values in `indices` are the indices in the first dimension of
 * the slices that have been extracted from the larger tensor.
 *
 * The dense [[Output]], `dense`, represented by [IndexedSlices], `slices`, has:
 * {{{
 *   dense(slices.indices(i), ::, ::, ...) = slices.values(i, ::, ::, ...)
 * }}}
 *
 * The [IndexedSlices] class is used primarily in the definition of gradients for operations that have
 * sparse gradients, such as `gather`.
 *
 * Note that this is different than [[SparseOutput]] which uses multi-dimensional indices and scalar values.
 *
 * @param  indices    Indices along the first dimension of the corresponding dense [[Output]].
 * @param  values     Values corresponding to the provided indices.
 * @param  denseShape Shape of the corresponding dense [[Output]].
 *
 */
class IndexedSlices(val indices: Output, val values: Output, val denseShape: Output? = null) : OutputLike() {
  
  override val graph: Graph = ops.getGraphFromInputs(setOf(values.op, indices.op) +
                                                         if (denseShape == null) emptySet() else setOf(denseShape.op))
  override val name: String = "${values.name}[${indices.name}]" +
      if (denseShape != null) "(shape = ${denseShape.name})" else ""
  override val dataType: DataType<*> = values.dataType
  override val device: String = values.device
  override val op: Op = values.op
  override val consumers: List<Op> = values.consumers
  
  override fun toOutput(): Output {
    if (denseShape == null)
      throw IllegalStateException("Conversion of 'OutputIndexedSlices', '$this', " +
                                      "which has no dense shape information available, is not possible.")
    val denseShapeValue = constantValue<Any>(denseShape)
    if (denseShapeValue != null
        && denseShapeValue.numElements >= LARGE_SPARSE_NUM_ELEMENTS) {
      ops.logger.warn {
        "Converting sparse 'OutputIndexedSlices' to a dense 'Output' " +
            "with ${denseShapeValue.numElements} " +
            "elements. This may consume a large amount of memory."
      }
    } else
      ops.logger.warn {
        "Converting sparse 'OutputIndexedSlices' to a dense 'Output' of unknown shape. " +
            "This may consume a large amount of memory."
      }
    return tf.unsortedSegmentSum(values, indices, denseShape[0], "${values.op.name}/ToOutput")
  }
}

class SparseOutput(val indices: Output, val values: Output, val denseShape: Output? = null) : OutputLike() {
  override val name: String
    get() = TODO("not implemented")
  override val dataType: DataType<*>
    get() = TODO("not implemented")
  override val device: String
    get() = TODO("not implemented")
  override val op: Op
    get() = TODO("not implemented")
  override val consumers: List<Op>
    get() = TODO("not implemented")
  override val graph: Graph
    get() = TODO("not implemented")
  
  override fun toOutput(): Output {
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
 * @param[valueIndex] Index of the operation's endpoint that produces this tensor.
 */
class Output(override val op: Op, val valueIndex: Int) : OutputLike() {
  
  override val graph = op.graph
  override val device = op.device
  override val consumers: List<Op>
    get() {
      val tf_output = asTF_Output()
      val numConsumers = TF_OperationOutputNumConsumers(tf_output)
      val consumers = TF_Input(numConsumers.toLong())
      TF_OperationOutputConsumers(tf_output, consumers, numConsumers)
      return List(numConsumers) {
        graph.cache(consumers.position(it.toLong()).oper())
      }
    }
  
  override fun toOutput() = this
  
  override val dataType: DataType<*>
    get() = op.output_types[valueIndex]
  
  val shape: Shape
    get() {
      val c_graph = op.graph.c_graph
      val output = asTF_Output()
      val status = newStatus()
      val numDims = TF_GraphGetTensorNumDims(c_graph, output, status)
      if (numDims < 0) return Shape()
      status.check()
      val dims = LongArray(numDims)
      TF_GraphGetTensorShape(c_graph, output, dims, numDims, status)
      status.check()
      return Shape(dims)
    }
  
  val rank: Int
    get() = shape.rank
  
  val size: Int
    get() = shape.numElements()
  
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
  image.setShape([28, 28, 3])
  print(image.shape)
  ==> TensorShape([Shape(28), Shape(28), Shape(3)])
  ```
   
   * @param shape: A `TensorShape` representing the shape of this tensor, a
  `TensorShapeProto`, a list, a tuple, or None.
  
  Raises:
  ValueError: If `shape` is not compatible with the current shape of
  this tensor.
   */
  fun setShape(shape: Shape) {
    assert(this.shape.isCompatibleWith(shape))
    val dims = shape.asLongArray()
    val status = newStatus()
    TF_GraphSetTensorShape(op.graph.c_graph, asTF_Output(), dims, dims!!.size, status)
    status.check()
  }
  
  //  val tf: TF by lazy { TODO("op!!.graph.tf") }
  fun asTF_Output() = TF_Output().oper(op.c_op).index(valueIndex)
  
  override val name: String by lazy { "${op.name}:$valueIndex" }
//  val name: String by lazy { op!!.name }
  
  inline fun node() = op.c_op.node()
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is Output) return false
    
    if (op != other.op) return false
    if (valueIndex != other.valueIndex) return false
    if (dataType != other.dataType) return false
    
    return true
  }
  
  override fun hashCode(): Int {
    var result = op.hashCode()
    result = 31 * result + valueIndex
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
        else -> """Output("$name", shape=$shape, dataType=$dataType, op=$op)"""
      }
}