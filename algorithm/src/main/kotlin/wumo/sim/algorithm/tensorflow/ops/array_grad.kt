package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.Op
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.gradients.noGradient
import wumo.sim.algorithm.tensorflow.ops.gradients.register_gradient_op
import wumo.sim.algorithm.tensorflow.ops.gradients.register_no_gradient_op
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.i

fun register_array_grad() {
  register_no_gradient_op("Const", "StopGradient",
                          "ConcatOffset",
                          "EditDistance",
                          "ZerosLike",
                          "InvertPermutation",
                          "Shape",
                          "ShapeN",
                          "Rank",
                          "Size",
                          "BroadcastGradientArgs",
                          "OneHot")
  
  register_gradient_op("Pack") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val N = op.attrLong("N")
//    val N = LongPointer(1)
    val axis = op.attrLong("axis")
    
    val grad_op = tf.unstack(grad, N, axis = axis)
    for (o in grad_op)
      grad_outputs.add(o)
  }
  register_gradient_op("Unpack") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val axis = op.attrLong("axis")
    grad_outputs.add(tf.pack(grad_inputs, axis = axis.toInt()))
  }
  register_gradient_op("Identity") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.identity(grad))
  }
  register_gradient_op("RefIdentity") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.identity(grad))
  }
  register_gradient_op("QuantizeAndDequantize") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.identity(grad))
  }
  register_gradient_op("QuantizeAndDequantizeV2") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.identity(grad))
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("QuantizeAndDequantizeV3") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.identity(grad))
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("Split") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(noGradient)
    grad_outputs.add(tf.concat(grad_inputs, op.inputs[0]))
  }
  register_gradient_op("Diag") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.diagPart(grad))
  }
  register_gradient_op("DiagPart") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.diag(grad))
  }
  register_gradient_op("MatrixDiag") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.matrixDiagPart(grad))
  }
  register_gradient_op("MatrixBandPart") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val num_lower = op.inputs[1]
    val num_upper = op.inputs[2]
    grad_outputs.add(tf.matrixBandPart(grad, num_lower, num_upper))
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("GatherNd") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val ref = op.inputs[0]
    val ref_shape = tf.shape(ref)
    val indices = op.inputs[1]
    grad_outputs.add(tf.scatterNd(indices, grad, ref_shape))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("CheckNumerics") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val message = op.attrString("message")
    grad_outputs.add(tf.checkNumerics(grad, "Not a number (NaN) or infinity (Inf) values detected in gradient. $message"))
  }
  register_gradient_op("Reshape") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val input_shape = tf.shape(op.inputs[0])
    grad_outputs.add(tf.reshape(grad, input_shape))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("ExpandDims") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val input_shape = tf.shape(op.inputs[0])
    grad_outputs.add(tf.reshape(grad, input_shape))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("Squeeze") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val input_shape = tf.shape(op.inputs[0])
    grad_outputs.add(tf.reshape(grad, input_shape))
  }
  register_gradient_op("Transpose") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val inverted_perm = tf.invertPermutation(op.inputs[1])
    grad_outputs.add(tf.transpose(grad, inverted_perm))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("ReverseSequence") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val seq_lengths = op.inputs[1]
    val batch_dim = op.attrLong("batch_dim")
    val seq_dim = op.attrLong("seq_dim")
    grad_outputs.add(
        tf.reverseSequence(grad, seq_lengths, seq_dim,
                           batch_dim = batch_dim))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("ReverseV2") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val reverse_dims = op.inputs[1]
    grad_outputs.add(tf.reverse(grad, reverse_dims))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("ScatterNd") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val indices = op.inputs[0]
    grad_outputs.add(noGradient);
    grad_outputs.add(tf.gatherNd(grad, indices))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("ScatterNdNonAliasingAdd") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val indices = op.inputs[1]
    grad_outputs.add(tf.identity(grad));
    grad_outputs.add(noGradient)
    grad_outputs.add(tf.gatherNd(grad, indices));
  }
  
  fun padGrad(op: Op, grad_inputs: List<Tensor>, grad_outputs: MutableList<Tensor>, isPadV2: Boolean) {
    val grad = grad_inputs[0]
    val x = op.inputs[0]
    val a = op.inputs[1]  // [Rank(x), 2]
    // Takes a slice of a. The 1st column. [Rank(x), 1].
    val size = tf.stack(listOf(tf.rank(x), tf.const(1)))
    val pad_before = tf.slice(a, tf.const(i(0, 0)), size)
    // Make it a 1-D tensor.
    val begin = tf.reshape(pad_before, tf.const(i(-1)))
    grad_outputs.add(tf.slice(grad, begin, tf.shape(x)));
    grad_outputs.add(noGradient)
    // PadV2 adds a "constant_values" input.
    if (isPadV2) {
      grad_outputs.add(noGradient)
    }
  }
  
  register_gradient_op("Pad") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    padGrad(op, grad_inputs, grad_outputs, false)
  }
  register_gradient_op("PadV2") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    padGrad(op, grad_inputs, grad_outputs, true)
  }
  register_gradient_op("SpaceToBatch") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val block_size = op.attrLong("block_size")
    grad_outputs.add(
        tf.batchToSpace(grad, op.inputs[1], block_size))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("SpaceToBatchND") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(
        tf.batchToSpaceND(grad, op.inputs[1], op.inputs[2]))
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("BatchToSpace") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val block_size = op.attrLong("block_size")
    grad_outputs.add(
        tf.spaceToBatch(grad, op.inputs[1], block_size))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("BatchToSpaceND") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(
        tf.spaceToBatchND(grad, op.inputs[1], op.inputs[2]))
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("SpaceToDepth") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val block_size = op.attrLong("block_size")
    grad_outputs.add(tf.depthToSpace(grad, block_size))
  }
  register_gradient_op("DepthToSpace") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val block_size = op.attrLong("block_size")
    grad_outputs.add(tf.spaceToDepth(grad, block_size))
  }
  register_gradient_op("MirrorPad") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val mode = op.attrString("mode")
    grad_outputs.add(tf.mirrorPadGrad(grad, op.inputs[1], mode))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("MirrorPadGrad") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val mode = op.attrString("mode")
    grad_outputs.add(tf.mirrorPad(grad, op.inputs[1], mode))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("StridedSlice") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val x = tf.shape(op.inputs[0])
    val begin = op.inputs[1]
    val end = op.inputs[2]
    val strides = op.inputs[3]
    val begin_mask = op.attrLong("begin_mask")
    val end_mask =op.attrLong("end_mask")
    val ellipsis_mask = op.attrLong("ellipsis_mask")
    val new_axis_mask = op.attrLong("new_axis_mask")
    val shrink_axis_mask = op.attrLong("shrink_axis_mask")
    grad_outputs.add(
        tf.stridedSliceGrad(x, begin, end, strides, grad,
                            begin_mask = begin_mask,
                            end_mask = end_mask,
                            ellipsis_mask = ellipsis_mask,
                            new_axis_mask = new_axis_mask,
                            shrink_axis_mask = shrink_axis_mask))
    // No gradients returned for begin, end and strides
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("Slice") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    // Propagate the incoming gradient along all the selected values,
    // and zero everywhere else. Use the Pad operator for this.
    //
    // First create an Nx2 padding where N is the number of input
    // dimensions. The first column is the number of prepended zeros
    // for each dimension, and the second column is the number of
    // appended zeros.
    //
    // The first column is just the begin vector.
    // The second column is the shape of the input element-wise
    // subtracted by begin+size
    
    // Running example:
    // input.shape = [3, 5, 3]
    // begin = [1, 2, 1], size = [1, 3, 2]
    val input_vec = op.inputs[0]
    val begin_vec = op.inputs[1]
    // input_rank = 3
    val input_rank = tf.rank(input_vec)
    // slice_size = [1, 3, 2]
    val slice_size = tf.shape(op.outputs[0])
    // padding_shape = [3, 1]
    val padding_shape = tf.stack(listOf(input_rank, tf.const(1)))
    // before_padding = [[1]
    //                   [2]
    //                   [1]]
    val before_padding = tf.reshape(begin_vec, padding_shape)
    // after_padding_sizes = shape(input) - slice_size - begin
    //                     = [3, 5, 3] - [1, 3, 2] - [1, 2, 1]
    //                     = [1, 0, 0]
    val after_padding_sizes =
        tf.sub(tf.sub(tf.shape(input_vec), slice_size), begin_vec)
    // after_padding = [[1]
    //                  [0]
    //                  [0]]
    val after_padding = tf.reshape(after_padding_sizes, padding_shape)
    // paddings = [[1 1]
    //             [2 0]
    //             [1 0]]
    val paddings =
        tf.concat(listOf(before_padding, after_padding), tf.const(1))
    grad_outputs.add(tf.pad(grad, paddings));
    // Nothing propagated for "begin" and "size" inputs
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
}