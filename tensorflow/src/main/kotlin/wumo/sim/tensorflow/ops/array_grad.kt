//package wumo.sim.tensorflow.ops
//
//import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
//import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
//import wumo.sim.tensorflow.tf
//import wumo.sim.util.i
//
//fun register_array_grad() {
//  registerNonDifferentiable("Const", "StopGradient",
//                            "ConcatOffset",
//                            "EditDistance",
//                            "ZerosLike",
//                            "InvertPermutation",
//                            "Shape",
//                            "ShapeN",
//                            "Rank",
//                            "Size",
//                            "BroadcastGradientArgs",
//                            "OneHot")
//
//  register("Pack") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val N = op.attrLong("N")
////    val N = LongPointer(1)
//    val axis = op.attrLong("axis")
//
//    val grad_op = tf._unpack(grad, N, axis = axis)
//    for (o in grad_op)
//      grad_outputs.add(o)
//  }
//  register("Unpack") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val axis = op.attrLong("axis")
//    grad_outputs.add(tf._pack(grad_inputs.map { it!!.toOutput() }, axis = axis))
//  }
//  register("Identity") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._identity(grad))
//  }
//  register("RefIdentity") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._identity(grad))
//  }
//  register("QuantizeAndDequantize") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._identity(grad))
//  }
//  register("QuantizeAndDequantizeV2") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._identity(grad))
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//  register("QuantizeAndDequantizeV3") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._identity(grad))
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//  register("Split") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(noGradient)
//    grad_outputs.add(tf._concatV2(grad_inputs.map { it!!.toOutput() }, op.inputs[0]))
//  }
//  register("Diag") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._diagPart(grad))
//  }
//  register("DiagPart") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._diag(grad))
//  }
//  register("MatrixDiag") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(tf._matrixDiagPart(grad))
//  }
//  register("MatrixBandPart") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val num_lower = op.inputs[1]
//    val num_upper = op.inputs[2]
//    grad_outputs.add(tf._matrixBandPart(grad, num_lower, num_upper))
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//  register("GatherNd") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val ref = op.inputs[0]
//    val ref_shape = tf._shape(ref)
//    val indices = op.inputs[1]
//    grad_outputs.add(tf._scatterNd(indices, grad, ref_shape))
//    grad_outputs.add(noGradient)
//  }
//  register("CheckNumerics") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val message = op.attrString("message")
//    grad_outputs.add(tf._checkNumerics(grad, "Not a number (NaN) or infinity (Inf) values detected in gradient. $message"))
//  }
//  register("Reshape") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val input_shape = tf._shape(op.inputs[0])
//    grad_outputs.add(tf._reshape(grad, input_shape))
//    grad_outputs.add(noGradient)
//  }
//  register("ExpandDims") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val input_shape = tf._shape(op.inputs[0])
//    grad_outputs.add(tf._reshape(grad, input_shape))
//    grad_outputs.add(noGradient)
//  }
//  register("Squeeze") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val input_shape = tf._shape(op.inputs[0])
//    grad_outputs.add(tf._reshape(grad, input_shape))
//  }
//  register("Transpose") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val inverted_perm = tf._invertPermutation(op.inputs[1])
//    grad_outputs.add(tf._transpose(grad, inverted_perm))
//    grad_outputs.add(noGradient)
//  }
//  register("ReverseSequence") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val seq_lengths = op.inputs[1]
//    val batch_dim = op.attrLong("batch_dim")
//    val seq_dim = op.attrLong("seq_dim")
//    grad_outputs.add(
//        tf._reverseSequence(grad, seq_lengths, seq_dim,
//                            batch_dim = batch_dim))
//    grad_outputs.add(noGradient)
//  }
//  register("ReverseV2") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val reverse_dims = op.inputs[1]
//    grad_outputs.add(tf._reverseV2(grad, reverse_dims))
//    grad_outputs.add(noGradient)
//  }
//  register("ScatterNd") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val indices = op.inputs[0]
//    grad_outputs.add(noGradient);
//    grad_outputs.add(tf._gatherNd(grad, indices))
//    grad_outputs.add(noGradient)
//  }
//  register("ScatterNdNonAliasingAdd") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val indices = op.inputs[1]
//    grad_outputs.add(tf._identity(grad));
//    grad_outputs.add(noGradient)
//    grad_outputs.add(tf._gatherNd(grad, indices));
//  }
//
//  fun padGrad(op: Op, grad_inputs: List<OutputLike?>, grad_outputs: MutableList<OutputLike?>, isPadV2: Boolean) {
//    val grad = grad_inputs[0]!!.toOutput()
//    val x = op.inputs[0]
//    val a = op.inputs[1]  // [Rank(x), 2]
//    // Takes a slice of a. The 1st column. [Rank(x), 1].
//    val size = tf.stack(listOf(tf._rank(x), tf.const(1)))
//    val pad_before = tf._slice(a, tf.const(i(0, 0)), size)
//    // Make it a 1-D tensor.
//    val begin = tf._reshape(pad_before, tf.const(i(-1)))
//    grad_outputs.add(tf._slice(grad, begin, tf._shape(x)));
//    grad_outputs.add(noGradient)
//    // PadV2 adds a "constant_values" input.
//    if (isPadV2) {
//      grad_outputs.add(noGradient)
//    }
//  }
//
//  register("Pad") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    padGrad(op, grad_inputs, grad_outputs, false)
//  }
//  register("PadV2") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    padGrad(op, grad_inputs, grad_outputs, true)
//  }
//  register("SpaceToBatch") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val block_size = op.attrLong("block_size")
//    grad_outputs.add(
//        tf._batchToSpace(grad, op.inputs[1], block_size))
//    grad_outputs.add(noGradient)
//  }
//  register("SpaceToBatchND") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(
//        tf._batchToSpaceND(grad, op.inputs[1], op.inputs[2]))
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//  register("BatchToSpace") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val block_size = op.attrLong("block_size")
//    grad_outputs.add(
//        tf._spaceToBatch(grad, op.inputs[1], block_size))
//    grad_outputs.add(noGradient)
//  }
//  register("BatchToSpaceND") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    grad_outputs.add(
//        tf._spaceToBatchND(grad, op.inputs[1], op.inputs[2]))
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//  register("SpaceToDepth") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val block_size = op.attrLong("block_size")
//    grad_outputs.add(tf._depthToSpace(grad, block_size))
//  }
//  register("DepthToSpace") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val block_size = op.attrLong("block_size")
//    grad_outputs.add(tf._spaceToDepth(grad, block_size))
//  }
//  register("MirrorPad") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val mode = op.attrString("mode")
//    grad_outputs.add(tf._mirrorPadGrad(grad, op.inputs[1], mode))
//    grad_outputs.add(noGradient)
//  }
//  register("MirrorPadGrad") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val mode = op.attrString("mode")
//    grad_outputs.add(tf._mirrorPad(grad, op.inputs[1], mode))
//    grad_outputs.add(noGradient)
//  }
//  register("StridedSlice") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    val x = tf._shape(op.inputs[0])
//    val begin = op.inputs[1]
//    val end = op.inputs[2]
//    val strides = op.inputs[3]
//    val begin_mask = op.attrLong("begin_mask")
//    val end_mask = op.attrLong("end_mask")
//    val ellipsis_mask = op.attrLong("ellipsis_mask")
//    val new_axis_mask = op.attrLong("new_axis_mask")
//    val shrink_axis_mask = op.attrLong("shrink_axis_mask")
//    grad_outputs.add(
//        tf._stridedSliceGrad(x, begin, end, strides, grad,
//                             begin_mask = begin_mask,
//                             end_mask = end_mask,
//                             ellipsis_mask = ellipsis_mask,
//                             new_axis_mask = new_axis_mask,
//                             shrink_axis_mask = shrink_axis_mask))
//    // No gradients returned for begin, end and strides
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//  register("Slice") { op, grad_inputs, grad_outputs ->
//    val grad = grad_inputs[0]!!.toOutput()
//    // Propagate the incoming gradient along all the selected values,
//    // and zero everywhere else. Use the Pad operator for this.
//    //
//    // First create an Nx2 padding where N is the number of input
//    // dimensions. The first column is the number of prepended zeros
//    // for each dimension, and the second column is the number of
//    // appended zeros.
//    //
//    // The first column is just the begin vector.
//    // The second column is the shape of the input element-wise
//    // subtracted by begin+size
//
//    // Running example:
//    // input.shape = [3, 5, 3]
//    // begin = [1, 2, 1], size = [1, 3, 2]
//    val input_vec = op.inputs[0]
//    val begin_vec = op.inputs[1]
//    // input_rank = 3
//    val input_rank = tf._rank(input_vec)
//    // slice_size = [1, 3, 2]
//    val slice_size = tf._shape(op.outputs[0])
//    // padding_shape = [3, 1]
//    val padding_shape = tf.stack(listOf(input_rank, tf.const(1)))
//    // before_padding = [[1]
//    //                   [2]
//    //                   [1]]
//    val before_padding = tf._reshape(begin_vec, padding_shape)
//    // after_padding_sizes = shape(input) - slice_size - begin
//    //                     = [3, 5, 3] - [1, 3, 2] - [1, 2, 1]
//    //                     = [1, 0, 0]
//    val after_padding_sizes =
//        tf._sub(tf._sub(tf._shape(input_vec), slice_size), begin_vec)
//    // after_padding = [[1]
//    //                  [0]
//    //                  [0]]
//    val after_padding = tf._reshape(after_padding_sizes, padding_shape)
//    // paddings = [[1 1]
//    //             [2 0]
//    //             [1 0]]
//    val paddings =
//        tf._concatV2(listOf(before_padding, after_padding), tf.const(1))
//    grad_outputs.add(tf._pad(grad, paddings));
//    // Nothing propagated for "begin" and "size" inputs
//    grad_outputs.add(noGradient)
//    grad_outputs.add(noGradient)
//  }
//}