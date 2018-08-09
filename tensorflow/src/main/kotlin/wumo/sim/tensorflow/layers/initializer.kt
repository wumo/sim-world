package wumo.sim.tensorflow.layers

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.layers.mode.*
import wumo.sim.tensorflow.ops.Initializer
import wumo.sim.tensorflow.ops.const
import wumo.sim.tensorflow.ops.random_uniform
import wumo.sim.tensorflow.ops.truncatedNormal
import wumo.sim.tensorflow.tf
import kotlin.math.sqrt

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
fun TF.xavier_initializer(uniform: Boolean = true) =
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
                                 uniform: Boolean = false) = Initializer(DT_FLOAT, "variance_scaling_initializer") { shape, dtype, name ->
  var fan_in = (if (shape.rank() > 1) shape[-2] else shape[-1]).toFloat()
  var fan_out = shape[-1].toFloat()
  for (dim in 0 until shape.rank() - 2) {
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
    tf.truncatedNormal(tf.const(shape.asIntArray()), 0f, trunc_stddev, dtype)
  }
}

enum class mode {
  FAN_IN,
  FAN_OUT,
  FAN_AVG
}