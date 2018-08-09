package wumo.sim.tensorflow.training

import org.bytedeco.javacpp.tensorflow.DT_INVALID
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Shape

fun create_slot_var(primary: Output,
                    initializer: Initializer,
                    name: String,
                    validate_shape: Boolean,
                    shape: Shape,
                    dtype: Int): Variable {
  //TODO partition
  return tf.variable(shape, dtype, initializer, name, trainable = false, validate_shape = validate_shape)
}

fun create_slot(primary: Output, v: Output, name: String, colocate_with_primary: Boolean): Variable {
  TODO("not implemented")
}


/**
 * Creates a slot initialized using an [Initializer].
 *
 * The type of the slot is determined by the given value.
 *
 * @param primary The primary `Variable` or `Output`.
 * @param initializer An `Initializer`.  The initial value of the slot.
 * @param shape Shape of the initial value of the slot.
 * @param dtype Type of the value of the slot.
 * @param name Name to use for the slot variable.
 * @param colocate_with_primary If True the slot is located on the same device as `primary`.
 * @return  A `Variable` object.
 */
fun create_slot_with_initializer(primary: Output,
                                 initializer: Initializer,
                                 shape: Shape, dtype: Int,
                                 name: String,
                                 colocate_with_primary: Boolean): Variable {
  val validate_shape = shape.is_fully_defined
  return if (colocate_with_primary)
    tf.colocate_with(primary) {
      create_slot_var(primary, initializer, name, validate_shape, shape, dtype)
    }
  else
    create_slot_var(primary, initializer, name, validate_shape, shape, dtype)
}

/**Create a slot initialized to 0 with same shape as the primary object
 * @param primary he primary `Variable` or [Output].
 * @param name Name to use for the slot variable.
 * @param dtype Type of the slot variable.  Defaults to the type of [primary].
 * @param colocate_with_primary f True the slot is located on the same device as [primary].
 * @return A [Variable] object.
 */
fun create_zeros_slot(primary: Output, name: String, dtype: Int = DT_INVALID, colocate_with_primary: Boolean = true): Variable {
  val dtype = if (dtype == DT_INVALID) primary.dtype else dtype
  val slot_shape = primary.shape
  return if (slot_shape.is_fully_defined) {
    val initializer = tf.zeros_initializer(dtype)
    create_slot_with_initializer(primary,
                                 initializer,
                                 slot_shape, dtype,
                                 name,
                                 colocate_with_primary = colocate_with_primary)
  } else {
    val slot_shape_t = if (primary is Variable) tf.shape(primary.initialized_value())
    else tf.shape(primary)
    val v = tf.zeros(slot_shape_t, dtype = dtype)
    create_slot(primary, v, name, colocate_with_primary = colocate_with_primary)
  }
}



