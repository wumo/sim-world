package wumo.sim.tensorflow.ops.training

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.*
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape

object slot_creator {
  /** Creates a slot initialized with zeros with the same shape as the primary variable.
   *
   * @param  primary             Primary variable.
   * @param  dataType            Data type of the slot variable.
   * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
   *                             variable.
   * @return Created slot variable.
   */
  internal fun zeros(
      primary: Variable,
      name: String,
      dataType: DataType<*>? = null,
      colocateWithPrimary: Boolean = true
  ): Variable {
    val inferredDataType = dataType ?: primary.dataType
    // TODO: [VARIABLES] What if the shape is not fully defined?
    return if (primary.shape.isFullyDefined) {
      create(primary, ZerosInitializer(), name, inferredDataType, primary.shape, colocateWithPrimary)
    } else {
      val initialValue = tf.zerosLike(primary.initializedValue, dataType)
      create(primary, DynamicInitializer(initialValue), name, inferredDataType, null, colocateWithPrimary)
    }
  }
  
  /** Creates a slot initialized with zeros with the same shape as the primary value.
   *
   * @param  primary             Primary value.
   * @param  dataType            Data type of the slot variable.
   * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
   *                             variable.
   * @return Created slot variable.
   */
  internal fun zerosForOutput(
      primary: Output,
      name: String,
      dataType: DataType<*>? = null,
      colocateWithPrimary: Boolean = true
  ): Variable {
    val inferredDataType = dataType ?: primary.dataType
    return if (primary.shape.isFullyDefined) {
      createForOutput(primary, ZerosInitializer(), name, inferredDataType, primary.shape, colocateWithPrimary)
    } else {
      val initialValue = tf.zerosLike(primary, dataType)
      createForOutput(
          primary, DynamicInitializer(initialValue), name, inferredDataType, null, colocateWithPrimary)
    }
  }
  
  /** Creates a new slow variable.
   *
   * @param  primary             Primary variable.
   * @param  initializer         Initializer for the slot variable.
   * @param  name                Name of the slot variable.
   * @param  dataType            Data type of the slot variable.
   * @param  shape               Shape of the slot variable. If `null`, then an attempt will be made to infer its value
   *                             from the provided initializer.
   * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
   *                             variable.
   * @return Created slot variable.
   * @see "tensorflow.python.training.slot_creator.create_slot_with_initializer"
   */
  internal fun create(
      primary: Variable,
      initializer: Initializer,
      name: String,
      dataType: DataType<*>? = null,
      shape: Shape? = null,
      colocateWithPrimary: Boolean = true
  ): Variable {
    // Scope the slot name in the namespace of the primary variable. Set "primary.op.name + '/' + name" as the default
    // name, so that the native name of the slot variable user can be shared when reuse is 'true'. Meanwhile, when reuse
    // is 'false' and the same name has been previously used, the native name will be made unique by appending an integer
    // to it.
    val inferredDataType = dataType ?: (initializer.dataType ?: FLOAT)
    val inferredShape = when {
      shape != null -> shape
      primary.shape.isFullyDefined -> primary.shape
      else -> initializer.shape
    }
    
    return VariableScope.scope("${primary.op.name}/$name", isDefaultName = true) {
      if (colocateWithPrimary)
        tf.colocateWith(mutableSetOf(primary.op)) {
          createSlotVariable(primary, initializer, "", inferredDataType, inferredShape)
        }
      else
        createSlotVariable(primary, initializer, "", inferredDataType, inferredShape)
    }
  }
  
  /** Creates a new slow variable.
   *
   * @param  primary             Primary value.
   * @param  initializer         Initializer for the slot variable.
   * @param  name                Name of the slot variable.
   * @param  dataType            Data type of the slot variable.
   * @param  shape               Shape of the slot variable. If `null`, then an attempt will be made to infer its value
   *                             from the provided initializer.
   * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
   *                             variable.
   * @return Created slot variable.
   */
  internal fun createForOutput(
      primary: Output,
      initializer: Initializer,
      name: String,
      dataType: DataType<*>? = null,
      shape: Shape? = null,
      colocateWithPrimary: Boolean = true
  ): Variable {
    // Scope the slot name in the namespace of the primary value. Set "primary.op.name + '/' + name" as the default
    // name, so that the native name of the slot variable user can be shared when reuse is 'true'. Meanwhile, when reuse
    // is 'false' and the same name has been previously used, the native name will be made unique by appending an integer
    // to it.
    val inferredDataType = if (dataType == null) initializer.dataType ?: FLOAT else dataType
    val inferredShape = when {
      shape != null -> shape
      primary.shape.isFullyDefined -> primary.shape
      else -> initializer.shape
    }
    
    return VariableScope.scope("${primary.op!!.name}/$name", isDefaultName = true) {
      if (colocateWithPrimary)
        tf.colocateWith(mutableSetOf(primary.op)) {
          createSlotVariable(primary, initializer, "", inferredDataType, inferredShape)
        }
      else
        createSlotVariable(primary, initializer, "", inferredDataType, inferredShape)
    }
  }
  
  /** Helper function for creating slot variables. */
  private fun createSlotVariable(
      primary: Variable,
      initializer: Initializer,
      scope: String,
      dataType: DataType<*>,
      shape: Shape?
  ): Variable {
    // TODO: [VARIABLES] When variables and partitioned variables are merged, makes sure this returns a normal variable.
    // TODO: [VARIABLES] When we support more variable types, match the returned variable type to the primary one.
    val slot = Variable.getVariable(scope, shape, dataType, initializer, trainable = false)
    primary.partitionInformation?.let {
      // Primary is a partitioned variable, and so we need to also indicate that the slot is also a partitioned
      // variable. Slots have the same partitioning as their primaries. For example, when using the Adam optimizer for a
      // linear model, 'slot.name' could be "linear//weights/Adam:0", while 'primary.op.name' is "linear//weight". We
      // want to get "Adam" as the real slot name, and so we remove "linear//weights/" and ":0".
      val realSlotName = slot.name.substring(primary.op.name.length + 1, slot.name.length - 2)
      slot.partitionInformation = Variable.PartitionInformation(
          fullName = "${it.fullName}/$realSlotName",
          fullShape = it.fullShape,
          partitionOffsets = it.partitionOffsets,
          partitionShape = it.partitionShape)
    }
    return slot
  }
  
  /** Helper function for creating slot variables. */
  private fun createSlotVariable(
      primary: Output,
      initializer: Initializer,
      scope: String,
      dataType: DataType<*>,
      shape: Shape?
  ): Variable =
      Variable.getVariable(scope, shape, dataType, initializer, trainable = false)
}
//
//import org.bytedeco.javacpp.tensorflow.DT_INVALID
//import wumo.sim.tensorflow.ops.Output
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.tensorflow.ops.*
//import wumo.sim.tensorflow.tf
//import wumo.sim.util.Shape
//
//fun create_slot_var(primary: Output,
//                    initializer: Initializer,
//                    name: String,
//                    validate_shape: Boolean,
//                    shape: Shape,
//                    dataType: Int): Variable {
//  //TODO partition
//  return tf.variable(shape, dataType, initializer, name, trainable = false, validate_shape = validate_shape)
//}
//
//fun create_slot(primary: Output, v: Output, name: String, colocate_with_primary: Boolean): Variable {
//  TODO("not implemented")
//}
//
//
///**
// * Creates a slot initialized using an [Initializer].
// *
// * The type of the slot is determined by the given value.
// *
// * @param primary The primary `Variable` or `Output`.
// * @param initializer An `Initializer`.  The initial value of the slot.
// * @param shape Shape of the initial value of the slot.
// * @param dataType Type of the value of the slot.
// * @param name Name to use for the slot variable.
// * @param colocate_with_primary If True the slot is located on the same device as `primary`.
// * @return  A `Variable` object.
// */
//fun create_slot_with_initializer(primary: Output,
//                                 initializer: Initializer,
//                                 shape: Shape, dataType: Int,
//                                 name: String,
//                                 colocate_with_primary: Boolean): Variable {
//  val validate_shape = shape.isFullyDefined
//  return if (colocate_with_primary)
//    tf.colocateWith(primary) {
//      create_slot_var(primary, initializer, name, validate_shape, shape, dataType)
//    }
//  else
//    create_slot_var(primary, initializer, name, validate_shape, shape, dataType)
//}
//
///**Create a slot initialized to 0 with same shape as the primary object
// * @param primary he primary `Variable` or [Output].
// * @param name Name to use for the slot variable.
// * @param dataType Type of the slot variable.  Defaults to the type of [primary].
// * @param colocate_with_primary f True the slot is located on the same device as [primary].
// * @return A [Variable] object.
// */
//fun create_zeros_slot(primary: Output, name: String, dataType: Int = DT_INVALID, colocate_with_primary: Boolean = true): Variable {
//  val dataType = if (dataType == DT_INVALID) primary.dataType else dataType
//  val slot_shape = primary.shape
//  return if (slot_shape.isFullyDefined) {
//    val initializer = tf.zerosInitializer(dataType)
//    create_slot_with_initializer(primary,
//                                 initializer,
//                                 slot_shape, dataType,
//                                 name,
//                                 colocate_with_primary = colocate_with_primary)
//  } else {
//    val slot_shape_t = if (primary is Variable) tf.shape(primary.initialized_value())
//    else tf.shape(primary)
//    val v = tf.zeros(slot_shape_t, dataType = dataType)
//    create_slot(primary, v, name, colocate_with_primary = colocate_with_primary)
//  }
//}



