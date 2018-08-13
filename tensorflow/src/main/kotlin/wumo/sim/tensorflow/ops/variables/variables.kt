package wumo.sim.tensorflow.ops.variables

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.layers.TensorFunction
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.ops
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.types
import wumo.sim.util.Shape

typealias Regularizer = TensorFunction

/**
 * Gets an existing model variable with these parameters or creates a new one.
 * Args:
 * @param name: the name of the new or existing variable.
 * @param shape: shape of the new or existing variable.
 * @param dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
 * @param initializer: initializer for the variable if one is created.
 * @param regularizer: a (Output -> Output or None) function; the result of
applying it on a newly created variable will be added to the collection
 * @param trainable: If `True` also add the variable to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
 
 * @return:The created or existing variable.
 */
fun model_variable(name: String, shape: Shape, dtype: Int = DT_FLOAT,
                   initializer: Initializer,
                   trainable: Boolean = true): Variable {
  TODO()
}

object variables {
  fun variable() {
  
  }
  
  /**
   * Provide a default initializer and a corresponding value.
   *
   */
  fun defaultInitializer(name: String, dataType: DataType<*> = types.FLOAT32): Initializer =
      when {
        dataType.isFloatingPoint -> TODO()
        dataType.isInteger || dataType.isUnsigned || dataType.isBoolean -> ZerosInitializer()
        else -> throw IllegalArgumentException("A default initializer for variable '$name' of" +
                                                   " type '$dataType' is required.")
      }
  
  /** This function defines the main logic of `getVariable`. However, `underlyingGetter` may override this logic.
   * That is why we pass it as an argument to the `underlyingGetter`. */
  val defaultVariableCreator: VariableGetter = object : VariableGetter {
    override fun invoke(
        name: String,
        dataType: DataType<*>,
        shape: Shape?,
        initializer: Initializer?,
        regularizer: Regularizer?,
        trainable: Boolean,
        reuse: Reuse,
        collections: Set<Graph.Key<Variable>>,
        cachingDevice: DeviceFunction?,
        underlyingGetter: VariableGetter?): Variable {
      val acutalInitializer = ops.init_scope {
        initializer ?: defaultInitializer(name, dataType)
      }
      TODO()
    }
    
  }
  
  internal fun makeGetter(): VariableGetter {
    var currentGetter = defaultVariableCreator
    ops.currentGraph.variableCreatorStack.value.forEach { g ->
      currentGetter = object : VariableGetter {
        override fun invoke(name: String, dataType: DataType<*>, shape: Shape?,
                            initializer: Initializer?, regularizer: Regularizer?,
                            trainable: Boolean, reuse: Reuse, collections: Set<Graph.Key<Variable>>,
                            cachingDevice: DeviceFunction?, underlyingGetter: VariableGetter?): Variable =
            g(name, dataType, shape, initializer, regularizer,
              trainable, reuse, collections, cachingDevice, currentGetter)
      }
    }
    return currentGetter
  }
}