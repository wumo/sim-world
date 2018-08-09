package wumo.sim.tensorflow.ops.variables

import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.core.Graph
import wumo.sim.algorithm.tensorflow.core.ShapeMismatchException
import wumo.sim.algorithm.tensorflow.ops.Initializer
import wumo.sim.algorithm.tensorflow.ops.OpSpecification
import wumo.sim.algorithm.tensorflow.types.DataType
import wumo.sim.algorithm.tensorflow.types.FLOAT32
import wumo.sim.util.Shape

/**
 * Variable store that carries a number of named Variables.
 */
internal class VariableStore {
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
      dataType: DataType<*> = FLOAT32,
      initializer: Initializer? = null,
      regularizer: Regularizer? = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set<Graph.Key<Variable>> = emptySet(),
      cachingDevice: ((OpSpecification) -> String)? = null
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
            "Variable '$name' already exists, but variable scope re-use was set to 'CreateNewOnly'.")
      val foundVariable = variables[name]!!
      if (shape != null && !shape.isCompatibleWith(foundVariable.shape))
        throw ShapeMismatchException(
            "Trying to share variable '$name', but the specified shape '$shape' is not compatible with the " +
                "existing variable shape '${foundVariable.shape}'.")
    }
    TODO()
  }
}