package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.core.Graph.Graph
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape

typealias Regularizer = TensorFunction

object variables {
  interface API {
    /**
     * @see "tensorflow.contrib.framework.python.ops.variables.modelVariable"
     */
    fun modelVariable(name: String,
                      shape: Shape? = null,
                      dataType: DataType<*>? = FLOAT,
                      initializer: Initializer? = null,
                      regularizer: Regularizer? = null,
                      trainable: Boolean = true,
                      reuse: Reuse = ReuseOrCreateNew,
                      collections: MutableSet<Graph.Key<Variable>> = mutableSetOf(),
                      cachingDevice: DeviceFunction? = null): Variable {
      collections += listOf(Graph.Keys.GLOBAL_VARIABLES, Graph.Keys.MODEL_VARIABLES)
      return variable(shape, dataType, initializer, regularizer, trainable,
                      reuse, collections, cachingDevice, name)
    }
    
    fun variable(
        shape: Shape? = null,
        dataType: DataType<*>? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew,
        collections: MutableSet<Graph.Key<Variable>> = mutableSetOf(),
        cachingDevice: DeviceFunction? = null,
        name: String
    ): Variable =
        Variable.getVariable(
            name, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    
    fun <R> variableScope(
        name: String,
        reuse: Reuse = ReuseOrCreateNew,
        dataType: DataType<*>? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        cachingDevice: DeviceFunction? = null,
        partitioner: Partitioner? = null,
        isDefaultName: Boolean = false,
        isPure: Boolean = false,
        block: () -> R
    ): R = VariableScope.scope(
        name, reuse, dataType, initializer, regularizer,
        cachingDevice, partitioner, isDefaultName,
        isPure, block)
    
    fun <R> updatedScope(
        variableScope: VariableScope = VariableScope.current,
        reuse: Reuse = ReuseOrCreateNew,
        dataType: DataType<*>? = null,
        initializer: Initializer? = null,
        regularizer: Regularizer? = null,
        cachingDevice: DeviceFunction? = null,
        partitioner: Partitioner? = null,
        isDefaultName: Boolean = false,
        isPure: Boolean = false,
        block: () -> R
    ): R = VariableScope.updatedScope(
        variableScope, reuse, dataType, initializer, regularizer,
        cachingDevice, partitioner, isDefaultName, isPure, block)
  }
  
  /**
   * Provide a default initializer and a corresponding value.
   *
   */
  fun defaultInitializer(name: String, dataType: DataType<*>? = FLOAT): Initializer =
      when {
        dataType!!.isFloatingPoint -> TODO()
        dataType.isInteger || dataType.isUnsigned || dataType.isBoolean -> ZerosInitializer()
        else -> throw IllegalArgumentException("A default initializer for variable '$name' of" +
                                                   " type '$dataType' is required.")
      }
  
  /** This function defines the main logic of `getVariable`. However, `underlyingGetter` may override this logic.
   * That is why we pass it as an argument to the `underlyingGetter`. */
  val defaultVariableCreator: VariableGetter = object : VariableGetter {
    override fun invoke(
        name: String,
        dataType: DataType<*>?,
        shape: Shape?,
        initializer: Initializer?,
        regularizer: Regularizer?,
        trainable: Boolean,
        reuse: Reuse,
        collections: MutableSet<Graph.Key<Variable>>,
        cachingDevice: DeviceFunction?,
        underlyingGetter: VariableGetter?): Variable {
      val acutalInitializer = tf.init_scope {
        initializer ?: defaultInitializer(name, dataType)
      }
      return Variable(acutalInitializer, shape, dataType, trainable, collections, cachingDevice, name)
    }
    
  }
  
  internal fun makeGetter(): VariableGetter {
    var currentGetter = defaultVariableCreator
    tf.currentGraph.variableCreatorStack.value.forEach { g ->
      currentGetter = object : VariableGetter {
        override fun invoke(name: String, dataType: DataType<*>?, shape: Shape?,
                            initializer: Initializer?, regularizer: Regularizer?,
                            trainable: Boolean, reuse: Reuse, collections: MutableSet<Graph.Key<Variable>>,
                            cachingDevice: DeviceFunction?, underlyingGetter: VariableGetter?): Variable =
            g(name, dataType, shape, initializer, regularizer,
              trainable, reuse, collections, cachingDevice, currentGetter)
      }
    }
    return currentGetter
  }
}