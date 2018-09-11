package wumo.sim.algorithm.drl.common

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.plus
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.PartitionInformation
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.NONE
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.randomNormal
import wumo.sim.util.ndarray.svd
import wumo.sim.util.ndarray.timesAssign

fun ortho_init(scale: Float = 1f): Initializer =
    object : Initializer {
      override fun init(shape: Shape, dataType: DataType<*>,
                        partitionInfo: PartitionInformation?): Output {
        val flat_shape =
            when (shape.rank) {
              2 -> shape
              4 -> Shape(intArrayOf(shape.slice(0, -1)
                                        .reduce { acc, i -> acc * i },
                                    shape[shape.rank - 1])) //assumes NHWC
              else -> NONE()
            }
        val a = randomNormal(0f, 1f, flat_shape)
        val (u, _, v) = svd(a)
        var q = if (u.shape == flat_shape) u
        else v
        q = q.reshape(shape)
        q *= scale
        return tf.const(q)
      }
    }

fun fc(x: Output, scope: String, num_hidden: Int,
       init_scale: Float = 1f, init_bias: Float = 0f): Output =
    tf.variableScope(scope) {
      val nin = x.shape[1]
      val w = tf.variable(Shape(nin, num_hidden), initializer = ortho_init(init_scale), name = "w")
      val b = tf.variable(Shape(num_hidden), initializer = tf.constantInitializer(init_bias), name = "b")
      tf.matMul(x, w.toOutput()) + b
    }