package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow.*

/**
 * Interface implemented by operands of a TensorFlow operation.
 *
 *
 * Example usage:
 *
 * <pre>`// The "decodeJpeg" operation can be used as an operand to the "cast" operation
 * Operand<UInt8> decodeJpeg = ops.image().decodeJpeg(...);
 * ops.math().cast(decodeJpeg, DataType.FLOAT);
 *
 * // The output "y" of the "unique" operation can be used as an operand to the "cast" operation
 * Output<Integer> y = ops.array().unique(...).y();
 * ops.math().cast(y, Float.class);
 *
 * // The "split" operation can be used as operand list to the "concat" operation
 * Iterable<? extends Operand<Float>> split = ops.array().split(...);
 * ops.array().concat(0, split);
`</pre> *
 */
interface Operand {
  
  /**
   * Returns the symbolic handle of a tensor.
   *
   *
   * Inputs to TensorFlow operations are outputs of another TensorFlow operation. This method is
   * used to obtain a symbolic handle that represents the computation of the input.
   *
   * @see OperationBuilder.addInput
   */
  fun asOutput(): Output
}

class Output(val op: Operation, val idx: Int) : Operand {
  val type = op.outputType(idx)
  
  override fun asOutput() = this
}