package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.mode.*
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.types
import wumo.sim.util.Shape
import kotlin.math.sqrt

interface Initializer {
  val dtype: DataType<*>?
    get() = null
  val shape: Shape?
    get() = null
  val name: String
  val init: (Shape, DataType<*>, String) -> Output
  operator fun invoke(shape: Shape, dtype: DataType<*>) =
      tf.nameScope(name) { init(shape, dtype, tf.currentNameScope) }
}

class ZerosInitializer : Initializer {
  override val name: String
    get() = "zeroes_initializer"
  override val init: (Shape, DataType<*>, String) -> Output
    get() = { shape, dtype, name ->
      tf.zeros(shape, dtype, name)
    }
}

fun zeros_initializer(dtype: DataType<*> = FLOAT) = ZerosInitializer()
fun ones_initializer(dtype: DataType<*> = FLOAT) = object : Initializer {
  override val name: String
    get() = "oness_initializer"
  override val init: (Shape, DataType<*>, String) -> Output
    get() = { shape, dtype, name ->
      tf.ones(shape, dtype, "ones")
    }
}

class DynamicInitializer(val value: Output) : Initializer {
  override val dtype = value.dtype
  override val shape = value.shape
  override val name: String
    get() = "constant_initializer"
  override val init: (Shape, DataType<*>, String) -> Output
    get() = { shape, dtype, name ->
      value
    }
  
}

fun constant_initializer(value: Any, dtype: DataType<*> = FLOAT) = object : Initializer {
  override val name: String
    get() = "const_initializer"
  override val init: (Shape, DataType<*>, String) -> Output
    get() = { shape, dtype, name ->
      tf.const(shape, dtype, value, name = "Const")
    }
}

/**
 * Returns an initializer performing "Xavier" initialization for weights.
 *
 * This function implements the weight initialization from:
 *
 * Xavier Glorot and Yoshua Bengio (2010):
 * [Understanding the difficulty of training deep feedforward neural
 * networks. International conference on artificial intelligence and
 * statistics.](
 * http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
 *
 * This initializer is designed to keep the scale of the gradients roughly the
 * same in all layers. In uniform distribution this ends up being the range:
 * `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
 * deviation of `sqrt(2. / (in + out))` is used.
 * @param uniform Whether to use uniform or normal distributed random initialization.
 * @return An initializer for a weight matrix.
 *
 */
fun xavier_initializer(uniform: Boolean = true) =
    variance_scaling_initializer(factor = 1.0f, mode = FAN_AVG, uniform = uniform)

/**
 * Returns an initializer that generates tensors without scaling variance.
 *
When initializing a deep network, it is in principle advantageous to keep
the scale of the input variance constant, so it does not explode or diminish
by reaching the final layer. This initializer use the following formula:

```python
if mode='FAN_IN': # Count only number of input connections.
n = fan_in
elif mode='FAN_OUT': # Count only number of output connections.
n = fan_out
elif mode='FAN_AVG': # Average number of inputs and output connections.
n = (fan_in + fan_out)/2.0

truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
```

- To get [Delving Deep into Rectifiers](
     http://arxiv.org/pdf/1502.01852v1.pdf) (also know as the "MSRA
initialization"), use (Default):<br/>
`factor=2.0 mode='FAN_IN' uniform=False`
- To get [Convolutional Architecture for Fast Feature Embedding](
     http://arxiv.org/abs/1408.5093), use:<br/>
`factor=1.0 mode='FAN_IN' uniform=True`
- To get [Understanding the difficulty of training deep feedforward neural
    networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
use:<br/>
`factor=1.0 mode='FAN_AVG' uniform=True.`
 * To get `xavier_initializer` use either:<br/>
`factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
`factor=1.0 mode='FAN_AVG' uniform=False`.
 * @param factor A multiplicative factor.
 * @param mode 'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
 * @param uniform Whether to use uniform or normal distributed random initialization.
 * @return An initializer that generates tensors with unit variance.
 */
fun variance_scaling_initializer(factor: Float = 2.0f,
                                 mode: mode = FAN_IN,
                                 uniform: Boolean = false) =
    object : Initializer {
      override val dtype: DataType<*>?
        get() = types.FLOAT
      override val name: String
        get() = "variance_scaling_initializer"
      override val init: (Shape, DataType<*>, String) -> Output
        get() = { shape, dtype, name ->
          var fan_in = (if (shape.rank > 1) shape[-2] else shape[-1]).toFloat()
          var fan_out = shape[-1].toFloat()
          for (dim in 0 until shape.rank - 2) {
            fan_in *= shape[dim]
            fan_out *= shape[dim]
          }
          val n = when (mode) {
            FAN_IN -> fan_in //Count only number of input connections.
            FAN_OUT -> fan_out //Count only number of output connections.
            FAN_AVG -> (fan_in + fan_out) / 2 //Average number of inputs and output connections.
          }
          if (uniform) {
            val limit = sqrt(3.0 * factor / n).toFloat()
            tf.random_uniform(shape, -limit, limit)
          } else {
            val trunc_stddev = sqrt(1.3 * factor / n).toFloat()
            tf._truncatedNormal(tf.const(shape.asIntArray()!!), dtype, 0L, trunc_stddev.toLong())
          }
        }
    }

enum class mode {
  FAN_IN,
  FAN_OUT,
  FAN_AVG
}