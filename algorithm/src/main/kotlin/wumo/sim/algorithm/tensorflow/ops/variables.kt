package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.layers.TensorFunction
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension

/**
 * Gets an existing model variable with these parameters or creates a new one.
 * Args:
 * @param name: the name of the new or existing variable.
 * @param shape: shape of the new or existing variable.
 * @param dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
 * @param initializer: initializer for the variable if one is created.
 * @param regularizer: a (Tensor -> Tensor or None) function; the result of
applying it on a newly created variable will be added to the collection
 * @param trainable: If `True` also add the variable to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
 
 * @return:The created or existing variable.
 */
fun model_variable(name: String, shape: Dimension, dtype: Int = DT_FLOAT,
                   initializer: Initializer,
                   trainable: Boolean = true): Variable {
  TODO()
}