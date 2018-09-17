package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.ops.basic.toOutput
import wumo.sim.tensorflow.ops.gen.gen_linalg_ops
import wumo.sim.tensorflow.ops.variables.mode.*
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape
import wumo.sim.util.t2
import kotlin.math.max
import kotlin.math.sqrt

interface initializers {
  fun zerosInitializer(dtype: DataType<*> = FLOAT) = ZerosInitializer()
  fun onesInitializer(dtype: DataType<*> = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dtype
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output =
        tf.ones(shape, dtype)
  }
  
  fun constantInitializer(value: Any, dtype: DataType<*> = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dtype
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output =
        tf.const(shape, dtype, value)
  }
  
  fun randomUniformInitializer(minval: Float = -0.05f,
                               maxval: Float = 0.05f,
                               seed: Int? = null,
                               dataType: DataType<*>? = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dataType
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output =
        tf.randomUniform(shape, minval, maxval, dataType, seed)
  }
  
  fun randomNormalInitializer(mean: Float = 0f,
                              stddev: Float = 0.05f,
                              seed: Int? = null,
                              dataType: DataType<*>? = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dataType
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output =
        tf.randomNormal({ shape.toOutput(it) },
                        { tf.const(dataType, mean, it) },
                        { tf.const(dataType, stddev, it) },
                        dataType, seed)
  }
  
  fun truncatedNormalInitializer(mean: Float = 0f,
                                 stddev: Float = 0.05f,
                                 seed: Int? = null,
                                 dataType: DataType<*>? = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dataType
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output =
        tf.truncatedNormal(shape, mean, stddev, dataType, seed)
  }
  
  fun orthogonalInitializer(gain: Float = 0f,
                            seed: Int? = null,
                            dataType: DataType<*>? = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dataType
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output {
      require(shape.rank >= 2) { "The tensor to initialize must be at least two-dimensional" }
      val num_rows = shape.slice(0, -1).reduce { num_rows, dim -> num_rows * dim }
      val num_cols = shape[-1]
      val flat_shape = if (num_rows < num_cols) Shape(num_cols, num_rows)
      else Shape(num_rows, num_cols)
      
      val a = tf.randomNormal({ flat_shape.toOutput(it) }, dtype = dataType, seed = seed)
      var (q, r) = gen_linalg_ops.qr(a, fullMatrices = false)
      val d = tf.diagPart(r)
      q *= tf.sign(d)
      if (num_rows < num_cols)
        q = tf.matrixTranspose(q)
      return gain * tf.reshape(q, shape.toOutput())
    }
  }
  
  private fun computeFans(shape: Shape): t2<Int, Int> =
      when {
        shape.rank < 1 -> t2(1, 1)
        shape.rank == 1 -> t2(shape[0], shape[0])
        shape.rank == 2 -> t2(shape[0], shape[1])
        else -> {
          var receptive_field_size = 1
          for (dim in shape.slice(0, -2))
            receptive_field_size *= dim
          t2(shape[-2] * receptive_field_size,
             shape[-1] * receptive_field_size)
        }
      }
  
  fun glorotNormalInitializer(seed: Int? = null,
                              dataType: DataType<*>? = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dataType
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output {
      var scale = 1f
      val (fanIn, fanOut) = computeFans(shape)
      scale /= max(1f, (fanIn + fanOut) / 2f)
      //truncated_normal
      val stddev = sqrt(scale.toDouble()) / .87962566103423978
      return tf.truncatedNormal(shape, 0f, stddev.toFloat(), dataType, seed)
    }
  }
  
  fun glorotUniformInitializer(seed: Int? = null,
                               dataType: DataType<*>? = FLOAT) = object : Initializer {
    override val dataType: DataType<*>? = dataType
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output {
      var scale = 1f
      val (fanIn, fanOut) = computeFans(shape)
      scale /= max(1f, (fanIn + fanOut) / 2f)
      //uniform
      val limit = sqrt(3f * scale)
      return tf.randomUniform(shape, -limit, limit, dataType, seed)
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
                                   uniform: Boolean = false) = object : Initializer {
    override val dataType: DataType<*>? = FLOAT
    override fun init(shape: Shape, dataType: DataType<*>,
                      partitionInfo: PartitionInformation?): Output {
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
      return if (uniform) {
        val limit = sqrt(3.0 * factor / n).toFloat()
        tf.randomUniform(shape, -limit, limit)
      } else {
        val trunc_stddev = sqrt(1.3 * factor / n).toFloat()
        tf.truncatedNormal(shape, 0f, trunc_stddev, dtype = dataType)
      }
    }
  }
}

interface Initializer {
  val dataType: DataType<*>? get() = null
  val shape: Shape? get() = null
  
  fun init(shape: Shape, dataType: DataType<*>,
           partitionInfo: PartitionInformation? = null): Output
  
  operator fun invoke(shape: Shape, dtype: DataType<*>? = null,
                      partitionInfo: PartitionInformation? = null) =
      init(shape, dtype ?: this.dataType!!)
}

class ZerosInitializer : Initializer {
  override fun init(shape: Shape, dataType: DataType<*>,
                    partitionInfo: PartitionInformation?): Output =
      tf.zeros(shape, dataType)
}

class DynamicInitializer(val value: Output) : Initializer {
  override val dataType = value.dataType
  override val shape = value.shape
  
  override fun init(shape: Shape, dataType: DataType<*>,
                    partitionInfo: PartitionInformation?): Output = value
}

enum class mode {
  FAN_IN,
  FAN_OUT,
  FAN_AVG
}