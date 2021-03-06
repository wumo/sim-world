package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.core.Graph.Graph
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.core.ShapeMismatchException
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.variables.variables.defaultInitializer
import wumo.sim.tensorflow.ops.variables.variables.makeGetter
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape

/**
 * Variable store that carries a number of named Variables.
 */
class VariableStore {
  
  /** Map with variable names as keys and the corresponding variables as values. */
  private val variables = hashMapOf<String, Variable>()
  /** Map with partitioned variable names as keys and the corresponding partitioned variables as values. */
  private val partitionedVariables = hashMapOf<String, PartitionedVariable>()
  
  /** Gets or creates a variable.
   *
   * @param  name          Variable name.
   * @param  dataType      Variable data type.
   * @param  shape         Variable shape.
   * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
   *                       passed in the constructor is used. If that one is `null` too, then we use a new
   *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
   *                       variable separately.
   * @param  regularizer   Variable regularizer.
   * @param  trainable     If `true`, the default, the variable is added to the graph collection
   *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
   *                       to use by the optimizers.
   * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
   *                       a new variable, or do either.
   * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
   *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
   * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
   *                       to the variable's device. Typical use is to cache on the device where the ops using the
   *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
   * @return Requested variable.
   * @throws IllegalArgumentException If any of the provided arguments are not compatible with each other, or with the
   *                                  variables stored in this variable store.
   * @throws ShapeMismatchException   If the provided shape does not match the shape of the corresponding variable
   *                                  stored in this variable store (if there exists one).
   * @throws InvalidDataTypeException If the provided data type does not match the data type of the corresponding
   *                                  variable stored in this variable store (if there exists one).
   */
  fun getVariable(
      name: String,
      shape: Shape? = null,
      dataType: DataType<*>? = FLOAT,
      initializer: Initializer? = null,
      regularizer: Regularizer? = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: MutableSet<Graph.Key<Variable>> = mutableSetOf(),
      cachingDevice: DeviceFunction? = null
  ): Variable {
    // Single variable case.
    if ("$name/part_0" in variables)
      throw IllegalArgumentException(
          "No partitioner was provided, but a partitioned version of the " +
              "variable was found: $name/part_0. Perhaps a variable of the same " +
              "name was already created with partitioning?")
    if (name in variables) {
      // Here we handle the case of returning an existing variable.
      if (reuse == CreateNewOnly)
        throw IllegalArgumentException(
            "Variable '$name' already exists, but variable native re-use was set to 'CreateNewOnly'.")
      val foundVariable = variables[name]!!
      if (shape != null && !shape.isCompatibleWith(foundVariable.shape))
        throw ShapeMismatchException(
            "Trying to share variable '$name', but the specified shape '$shape' is not compatible with the " +
                "existing variable shape '${foundVariable.shape}'.")
      if (dataType != foundVariable.dataType)
        throw InvalidDataTypeException(
            "Trying to share variable '$name', but the specified data type '$dataType' is not compatible with the " +
                "existing variable data type '${foundVariable.dataType}'.")
      return foundVariable
    } else {
      // Here we handle the case of creating a new variable.
      if (reuse == ReuseExistingOnly)
        throw IllegalArgumentException(
            "Variable '$name' does not exist, but variable native re-use was set to 'ReuseExistingOnly'.")
      if (shape != null && !shape.isFullyDefined)
        throw IllegalArgumentException(
            "The shape of a new variable ('$name') must be fully defined, but instead it was set to '$shape'.")
//      val acutalInitializer=
      val actualInitializer = tf.init_scope {
        initializer ?: defaultInitializer(name, dataType)
      }
      val variable = makeGetter()(name, dataType, shape, actualInitializer,
                                  regularizer, trainable, reuse, collections, cachingDevice, null)
      variables[name] = variable
      // Run the regularizer if specified and save the resulting loss.
      if (regularizer != null) {
        tf.colocateWith(variable.op) {
          val loss = tf.nameScope("$name/Regularizer") {
            regularizer(variable.value)
          }
          if (loss != null)
            tf.currentGraph.addToCollection(loss, Graph.Keys.REGULARIZATION_LOSSES)
        }
      }
      return variable
    }
  }
  
  companion object {
    val current get() = tf.currentGraph.variableStore
  }
}