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



