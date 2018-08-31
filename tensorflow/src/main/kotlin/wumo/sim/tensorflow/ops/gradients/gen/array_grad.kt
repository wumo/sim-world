import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register

//import wumo.sim.tensorflow.ops.*
//import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops
//import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
//import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
//import wumo.sim.tensorflow.tensor.constantValue
//import wumo.sim.tensorflow.tf
//import wumo.sim.util.append
//import wumo.sim.util.i
//
fun register_array_grad() {
//  /**Gradients for operators defined in array_ops.py.*/
///* from__future__importabsolute_import */
///* from__future__importdivision */
///* from__future__importprint_function */
///* fromtensorflow.pythonimportpywrap_tensorflow */
///* fromtensorflow.python.eagerimportcontext */
///* fromtensorflow.python.frameworkimportconstant_op */
///* fromtensorflow.python.frameworkimportops */
///* fromtensorflow.python.frameworkimportsparse_tensor */
///* fromtensorflow.python.frameworkimporttensor_util */
///* fromtensorflow.python.opsimportarray_ops */
///* fromtensorflow.python.opsimportcontrol_flow_util */
///* fromtensorflow.python.opsimportgen_array_ops */
///* fromtensorflow.python.opsimportmath_ops */
///* fromtensorflow.python.opsimportsparse_ops */
//  register("Pack") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for pack op.*/
//    tf.unstack(grad, num = op.attrLong("N").toInt(), axis = op.attrLong("axis").toInt())  //return@register
//  }
//  register("Unpack") { op, grad ->
//    /**Gradient for unpack op.*/
//    listOf(tf.stack(grad.map { it!!.toOutput() }, axis = op.attrLong("axis")))  //return@register
//  }
//  fun concatGradHelper(op: Op, grad: OutputLike, startValueIndex: Int, endValueIndex: Int, dimIndex: Int): List<OutputLike?> {
//    /**Gradient for concat op.
//
//    Args:
//    op: An operation.
//    grad: `Tensor` or `IndexedSlices` representing the gradients with respect
//    to each output of the op.
//    start_value_index: An integer index of the first value in the op.inputs.
//    end_value_index: An integer index of the last value in the op.inputs.
//    dim_index: An interger index of concat_dim or axis parameter in op.inputs.
//
//    Returns:
//    Tensors representing the partial gradients with respect to each input
//    of the op.
//
//    Raises:
//    ValueError: if concat_dim/axis is not statically known.
//     */
//    fun createDenseMaskAndBegin(sizes, concatDim) {
//      /**Create variables for iteratively slicing a dense gradients tensor.*/
//      val shapeOfShape = tf.shape(sizes[0])
//      val mask = tf.concat(listOf(tf.fill(tf.expandDims(concatDim, 0), 0), listOf(1), tf.fill(shapeOfShape - concatDim - 1, 0)), 0)
//      val begin = tf.fill(shapeOfShape, 0)
//      return mask, begin
//    }
//
//    fun extractInputShapes(inputs: List<Output>): List<Output> {
//      /**Extract the shapes of a set of input tensors.*/
//      val sizes = mutableListOf<Output>()
//      var fullyKnown = true
//
//      for (x in inputs) {
//        val inputShape = tf.shape(x)
//        if (inputShape.op.opType != "Const") {
//          fullyKnown = false
//          break
//        }
//        sizes.append(inputShape)
//      }
//      return if (fullyKnown)
//        sizes
//      else
//        tf.shapeN(inputs)
//    }
//    if ((op.inputs).size == 2) {
//      return if (endValueIndex <= dimIndex)
//        listOf(grad, null)
//      else listOf(null, grad)
//    }
//    val concatDim = op.inputs[dimIndex]
//    val inputValues = op.inputs.subList(startValueIndex, endValueIndex)
//    val outGrads = listOf()
//    when {
//      grad is Output -> {
//        if (const_ops.isConstant(concatDim.op)) {
//          val gradContext = control_flow_ops.getOutputContext(grad.op)
//          val dimContext = control_flow_ops.getOutputContext(concatDim.op)
//          if (dimContext != gradContext) {
//            val value = constantValue(concatDim)
//            concatDim = tf.const(value = value, dtype = concatDim.dataType)
//          }
//        }
//        val nonNegConcatDim = concatDim % tf.rank(inputValues[0])
//        val sizes = extractInputShapes(inputValues)
//        if (sizes.size > 16) {
//          val sizes = tf.squeeze(tf.slice(tf.stack(sizes, axis = 1),
//                                      tf.stack(listOf(nonNegConcatDim, tf.const(0))),
//                                      tf.const(i(1, -1))))
//          tf.split(grad, sizes, nonNegConcatDim)
//        } else {
//          val offset = tf.concatenateDataset(nonNegConcatDim, sizes)
//
//          for ((begin, size) in zip(offset, sizes)) {
//            outGrads.append(tf.slice(grad, begin, size))
//          }
//        }
//      }
//      grad is IndexedSlices -> {
//        val nonNegConcatDim = concatDim % tf.rank(inputValues[0])
//        val concatDimStatic = tf.constantValue(concatDim)
//        if (concatDimStatic == null) {
//          raiseValueError("Can only compute IndexedSlices gradient with ""statically-known concat_dim")
//
//        }
//        if (concatDimStatic < 0) {
//          val rank = tf.constantValue(tf.rank(inputValues[0]))
//          if (rank == null) {
//            raiseValueError("Can only compute IndexedSlices gradient with ""negative concat_dim when first value rank is ""statically-known.")
//
//          }
//          concatDimStatic %= rank
//        }
//        val sizes = listOf(tf.shape(x) forxininputValues)
//        if (concatDimStatic > 0) {
//          val (mask, begin) = createDenseMaskAndBegin(sizes, nonNegConcatDim)
//
//          for (size in sizes) {
//            val newValues = tf.slice(grad.values, begin, tf.concat(listOf(listOf(-1), tf.slice(size, listOf(1), listOf(-1))), 0))
//            outGrads.append(tf.IndexedSlices(newValues, grad.indices, size))
//            val begin = tf.add(begin, size * mask)
//
//          }
//
//        } else {
//          val start = tf.const(0, dtype = grad.indices.dataType)
//
//          for (size in sizes) {
//            val sizeConcatDim = tf.gather(size, nonNegConcatDim)
//            if (sizeConcatDim.dataType != grad.indices.dataType) {
//              sizeConcatDim = tf.cast(sizeConcatDim, dtype = grad.indices.dataType)
//
//            }
//            val end = start + sizeConcatDim
//            val indicesToSelect = tf.squeeze(tf.where(tf.logicalAnd(grad.indices >= start, grad.indices < end)), axis = listOf(1))
//            val newIndices = tf.gather(grad.indices, indicesToSelect) - start
//            val newValues = tf.gather(grad.values, indicesToSelect)
//            outGrads.append(tf.IndexedSlices(newValues, newIndices, size))
//            start = end
//
//          }
//
//        }
//
//      }
//      else -> throw IllegalArgumentException(
//          "Only 'Output' and 'IndexedSlices' gradients are supported for the concatenation op.")
//    }
//    return (outGrads + listOf(null) ifendValueIndex <= dimIndexelselistOf (null) + outGrads)
//  }
//  register("Concat") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    concatGradHelper(op, grad, startValueIndex = 1, endValueIndex = (op.inputs).size, dimIndex = 0)  //return@register
//  }
//  register("ConcatV2") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    concatGradHelper(op, grad, startValueIndex = 0, endValueIndex = op.inputs.lastIndex - 1, dimIndex = -1) //return@register
//  }
//  registerNonDifferentiable("ConcatOffset")
//  register("Slice") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for Slice op.*/
//    val inputVec = op.inputs[0]
//    val beginVec = op.inputs[1]
//    val inputRank = tf.rank(inputVec)
//    val sliceSize = tf.shape(op.outputs[0])
//    val shape = tf.stack(listOf(inputRank, 1))
//    val beforePad = tf.reshape(beginVec, shape)
//    val afterPad = tf.reshape(tf.shape(inputVec) - sliceSize - beginVec, shape)
//    val paddings = tf.concat(listOf(beforePad, afterPad), 1)
//    listOf(tf.pad(grad, paddings), null, null)  //return@register
//  }
//  register("StridedSlice") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for StridedSlice op.*/
//    val begin = op.inputs[1]
//    val end = op.inputs[2]
//    val strides = op.inputs[3]
//    val x = tf.shape(op.inputs[0], outType = begin.dataType)
//    listOf(tf.stridedSliceGrad(x, begin, end, strides, grad, beginMask = op.getAttr("begin_mask"), endMask = op.getAttr("end_mask"), ellipsisMask = op.getAttr("ellipsis_mask"), newAxisMask = op.getAttr("new_axis_mask"), shrinkAxisMask = op.getAttr("shrink_axis_mask")), null, null, null)  //return@register
//  }
//  register("StridedSliceGrad") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for StridedSliceGrad op.*/
//    val begin = op.inputs[1]
//    val end = op.inputs[2]
//    val strides = op.inputs[3]
//    listOf(null, null, null, null, tf.stridedSlice(grad, begin, end, strides, beginMask = op.getAttr("begin_mask"), endMask = op.getAttr("end_mask"), ellipsisMask = op.getAttr("ellipsis_mask"), newAxisMask = op.getAttr("new_axis_mask"), shrinkAxisMask = op.getAttr("shrink_axis_mask")))  //return@register
//  }
//  register("Split") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(null, tf.concat(list(grads), op.inputs[0]))  //return@register
//  }
//  register("SplitV") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val returnval = tf.concat(list(grads), op.inputs[2])
//    returnval = listOf(returnval) + listOf(null, ) * ((op.inputs).size - 1)
//    listOf(returnval)  //return@register
//  }
//  registerNonDifferentiable("Const")
//  register("Diag") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(tf.diagPart(grad))  //return@register
//  }
//  register("DiagPart") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(tf.diag(grad))  //return@register
//  }
//  register("MatrixDiag") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(tf.matrixDiagPart(grad))  //return@register
//  }
//  register("MatrixDiagPart") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val matrixShape = op.inputs[0].get_shape()[-2:]
//    if (matrixShape.isFullyDefined() && matrixShape[0] == matrixShape[1]) {
//      listOf(tf.matrixDiag(grad))  //return@register
//    } else {
//      listOf(tf.matrixSetDiag(tf.zerosLike(op.inputs[0]), grad))  //return@register
//
//    }
//
//  }
//  register("MatrixSetDiag") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for MatrixSetDiag.*/
//    val inputShape = op.inputs[0].get_shape().merge_with(grad.getShape())
//    val diagShape = op.inputs[1].get_shape()
//    val batchShape = inputShape[:-2].merge_with(diagShape[:-1])
//    val matrixShape = inputShape[-2:]
//    if (batchShape.isFullyDefined() && matrixShape.isFullyDefined()) {
//      diagShape = batchShape.asList() + listOf(min(matrixShape.asList()))
//
//    } else {
//      tf.colocateWith(grad) {
//        val gradShape = tf.shape(grad)
//        val gradRank = tf.rank(grad)
//        batchShape = tf.slice(gradShape, listOf(0), listOf(gradRank - 2))
//        matrixShape = tf.slice(gradShape, listOf(gradRank - 2), listOf(2))
//        val minDim = tf.reduceMin(matrixShape)
//        diagShape = tf.concat(listOf(batchShape, listOf(minDim)), 0)
//
//      }
//
//    }
//    val gradInput = tf.matrixSetDiag(grad, tf.zeros(diagShape, dtype = grad.dataType))
//    val gradDiag = tf.matrixDiagPart(grad)
//    listOf((gradInput, gradDiag))  //return@register
//  }
//  register("MatrixBandPart") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val numLower = op.inputs[1]
//    val numUpper = op.inputs[2]
//    listOf((tf.matrixBandPart(grad, numLower, numUpper), null, null))  //return@register
//  }
//  registerNonDifferentiable("EditDistance")
//  register("Fill") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(null, tf.sum(grad))  //return@register
//  }
//  registerNonDifferentiable("ZerosLike")
//  registerNonDifferentiable("OnesLike")
//  register("PreventGradient") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    raiseLookupError("Gradient explicitly disabled. Reason: %s" % op.get_attr("message"))
//
//  }
//  register("Gather") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for Gather op.*/
//    val params = op.inputs[0]
//    tf.colocateWith(params) {
//      val paramsShape = tf.shape(params, outType = INT64)
//      paramsShape = tf.toInt32(paramsShape)
//
//    }
//    val indices = op.inputs[1]
//    val size = tf.expandDims(tf.size(indices), 0)
//    val valuesShape = tf.concat(listOf(size, paramsShape[1:]), 0)
//    val values = tf.reshape(grad, valuesShape)
//    indices = tf.reshape(indices, size)
//    listOf(tf.IndexedSlices(values, indices, paramsShape), null) //return@register
//  }
//  register("GatherV2") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for GatherV2 op.*/
//    val params = op.inputs[0]
//    tf.colocateWith(params) {
//      val paramsShape = tf.shape(params, outType = INT64)
//      paramsShape = tf.toInt32(paramsShape)
//
//    }
//    val indices = op.inputs[1]
//    val indicesSize = tf.expandDims(tf.size(indices), 0)
//    val axis = op.inputs[2]
//    val axisStatic = tf.constantValue(axis)
//    if (axisStatic == 0) {
//      if (context.executingEagerly()) {
//        val paramsTailShape = paramsShape.cpu()[1:]
//
//      } else {
//        val paramsTailShape = paramsShape[1:]
//
//      }
//      val valuesShape = tf.concat(listOf(indicesSize, paramsTailShape), 0)
//      val values = tf.reshape(grad, valuesShape)
//      indices = tf.reshape(indices, indicesSize)
//      listOf(tf.IndexedSlices(values, indices, paramsShape), null, null) //return@register
//    }
//    val outerShape = paramsShape[:axis]
//    val outerDims = tf.size(outerShape)
//    val innerShape = paramsShape[axis:][1:]
//    val innerDims = tf.size(innerShape)
//    val outerAxesIndices = tf.range(outerDims)
//    val innerAxesIndices = tf.range(outerDims + 1, outerDims + 1 + innerDims)
//    val valuesShape = tf.concat(listOf(outerShape, indicesSize, innerShape), 0)
//    val values = tf.reshape(grad, valuesShape)
//    indices = tf.reshape(indices, indicesSize)
//    val transposeDims = tf.concat(listOf(listOf(outerDims), outerAxesIndices, innerAxesIndices), 0)
//    val valuesTranspose = tf.transpose(values, transposeDims)
//    val numSegments = paramsShape[axis]
//    val paramsGrad = tf.unsortedSegmentSum(valuesTranspose, indices, numSegments)
//    val invertTransposeDims = tf.concat(listOf(outerAxesIndices + 1, listOf(0), innerAxesIndices), 0)
//    paramsGrad = tf.transpose(paramsGrad, invertTransposeDims)
//    listOf(paramsGrad, null, null) //return@register
//  }
//  register("GatherNd") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val ref = op.inputs[0]
//    val indices = op.inputs[1]
//    val refShape = tf.shape(ref, outType = indices.dataType)
//    if (indices.shape.ndims == 2 && indices.shape[-1].value == 1) {
//      val refGrad = tf.IndexedSlices(grad, tf.squeeze(indices, axis = -1), refShape)
//
//    } else {
//      val refGrad = tf.scatterNd(indices, grad, refShape)
//
//    }
//    listOf(refGrad, null) //return@register
//  }
//  register("CheckNumerics") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Gradient for check_numerics op.*/
//    listOf(tf.checkNumerics(grad, "Not a number (NaN) or infinity (Inf) values detected in gradient."))  //return@register
//  }
  register("PlaceholderWithDefault", "Identity") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad)  //return@register
  }
//  register("RefIdentity") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(grad)  //return@register
//  }
//  register("IdentityN") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(grad)  //return@register
//  }
//  registerNonDifferentiable("StopGradient")
//  register("Reshape") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(tf.reshape(grad, tf.shape(op.inputs[0])), null) //return@register
//  }
//  registerNonDifferentiable("InvertPermutation")
//  fun reshapeToInput(op, grad) {
//    /**Reshapes the gradient to the shape of the original input.*/
//    return tf.reshape(grad, tf.shape(op.inputs[0]))
//  }
//  register("ExpandDims") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(reshapeToInput(op, grad), null) //return@register
//  }
//  register("Squeeze") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(reshapeToInput(op, grad))  //return@register
//  }
//  register("Transpose") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Returns unshuffle(grad).*/
//    val p = op.inputs[1]
//    listOf(tf.transpose(grad, tf.invertPermutation(p)), null) //return@register
//  }
//  register("ConjugateTranspose") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Returns conj(unshuffle(grad)).*/
//    val p = op.inputs[1]
//    listOf(tf.transpose(grad, tf.invertPermutation(p), conjugate = true), null) //return@register
//  }
//  registerNonDifferentiable("Shape")
//  registerNonDifferentiable("ShapeN")
//  registerNonDifferentiable("Rank")
//  registerNonDifferentiable("Size")
//  register("Tile") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    /**Sum reduces grad along the tiled dimensions.*/
//    val inputShape = tf.shape(op.inputs[0])
//    val splitShape = tf.reshape(tf.transpose(tf.stack(listOf(op.inputs[1], inputShape))), listOf(-1))
//    val axes = tf.range(0, tf.size(splitShape), 2)
//    if ((grad is ops.IndexedSlices)) {
//      grad = tf.unsortedSegmentSum(grad.values, tf.mod(grad.indices, inputShape[0]), inputShape[0])
//      splitShape = tf.concat(listOf(listOf(1), splitShape[1:]), axis = 0)
//
//    }
//    val inputGrad = tf.sum(tf.reshape(grad, splitShape), axes)
//    if (!context.executingEagerly()) {
//      inputGrad.setShape(op.inputs[0].get_shape())
//
//    }
//    listOf(inputGrad, null) //return@register
//  }
//  registerNonDifferentiable("BroadcastGradientArgs")
//  fun padGrad(op, grad) {
//    /**Gradient for Pad.*/
//    val x = op.inputs[0]
//    val a = op.inputs[1]
//    val padBefore = tf.slice(a, listOf(0, 0), tf.stack(listOf(tf.rank(x), 1)))
//    val begin = tf.reshape(padBefore, listOf(-1))
//    val sizes = tf.shape(x)
//    val xGrad = tf.slice(grad, begin, sizes)
//    if ((op.inputs).size == 3) {
//      return xGrad, null, null
//    } else {
//      return xGrad, null
//
//    }
//
//  }
//  register("Pad")(padGrad)
//  register("PadV2")(padGrad)
//  register("ReverseSequence") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val seqLengths = op.inputs[1]
//    listOf(tf.reverseSequence(grad, batchAxis = op.getAttr("batch_dim"), seqAxis = op.getAttr("seq_dim"), seqLengths = seqLengths), null) //return@register
//  }
//  register("Reverse") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val reverseDims = op.inputs[1]
//    listOf(tf.reverse(grad, reverseDims), null)  //return@register
//  }
//  register("ReverseV2") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val axis = op.inputs[1]
//    listOf(tf.reverseV2(grad, axis), null)  //return@register
//  }
//  register("SpaceToBatch") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val blockSize = op.getAttr("block_size")
//    listOf(tf.batchToSpace(grad, op.inputs[1], blockSize = blockSize), null) //return@register
//  }
//  register("SpaceToBatchND") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(tf.batchToSpaceNd(grad, op.inputs[1], op.inputs[2]), null, null) //return@register
//  }
//  register("BatchToSpace") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val blockSize = op.getAttr("block_size")
//    listOf(tf.spaceToBatch(grad, op.inputs[1], blockSize = blockSize), null) //return@register
//  }
//  register("BatchToSpaceND") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(tf.spaceToBatchNd(grad, op.inputs[1], op.inputs[2]), null, null) //return@register
//  }
//  register("SpaceToDepth") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val blockSize = op.getAttr("block_size")
//    val dataFormat = op.getAttr("data_format")
//    if (dataFormat == "NCHW_VECT_C") {
//      raiseValueError("Cannot compute SpaceToDepth gradient with NCHW_VECT_C. ""NCHW_VECT_C requires qint8 data type.")
//
//    }
//    listOf(tf.depthToSpace(grad, blockSize, dataFormat = dataFormat))  //return@register
//  }
//  register("DepthToSpace") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val blockSize = op.getAttr("block_size")
//    val dataFormat = op.getAttr("data_format")
//    if (dataFormat == "NCHW_VECT_C") {
//      raiseValueError("Cannot compute DepthToSpace gradient with NCHW_VECT_C. ""NCHW_VECT_C requires qint8 data type.")
//
//    }
//    listOf(tf.spaceToDepth(grad, blockSize, dataFormat = dataFormat))  //return@register
//  }
//  registerNonDifferentiable("OneHot")
//  register("MirrorPad") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val mode = op.getAttr("mode")
//    listOf(tf.mirrorPadGrad(grad, op.inputs[1], mode = mode), null) //return@register
//  }
//  register("MirrorPadGrad") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val mode = op.getAttr("mode")
//    listOf(tf.mirrorPad(grad, op.inputs[1], mode = mode), null) //return@register
//  }
//  register("QuantizeAndDequantize") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(grad)  //return@register
//  }
//  register("QuantizeAndDequantizeV2") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(grad, null, null) //return@register
//  }
//  register("QuantizeAndDequantizeV3") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    listOf(grad, null, null, null) //return@register
//  }
//  register("ExtractImagePatches") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val (batchSize, rowsIn, colsIn, channels) = listOf(dim.valuefordiminop.inputs[0].get_shape())
//    val inputBhwc = tf.shape(op.inputs[0])
//    val batchSize = inputBhwc[0]
//    val channels = inputBhwc[3]
//    val inputIndicesNum = 1 + rowsIn * colsIn
//    val inputIdx = tf.reshape(tf.range(1, inputIndicesNum, dtype = INT64), (1, rowsIn, colsIn, 1))
//    val inputIdxPatched = tf.extractImagePatches(inputIdx, op.getAttr("ksizes"), op.getAttr("strides"), op.getAttr("rates"), op.getAttr("padding"))
//    val (_, rowsOut, colsOut, _) = listOf(dim.valuefordiminop.outputs[0].get_shape())
//    val (_, ksizeR, ksizeC, _) = op.getAttr("ksizes")
//    val outputIndicesNum = rowsOut * colsOut * ksizeR * ksizeC
//    val outputIdx = tf.reshape(tf.range(outputIndicesNum, dtype = INT64), (1, rowsOut, colsOut, ksizeR * ksizeC))
//    val idxMatrix = tf.concat(listOf(tf.expandDims(inputIdxPatched, axis = -1), tf.expandDims(outputIdx, axis = -1)), axis = -1)
//    val idxMap = tf.reshape(idxMatrix, (-1, 2))
//    val spShape = (inputIndicesNum, outputIndicesNum)
//    val spMatFull = sparseTensor.SparseTensor(idxMap, tf.ones(listOf(outputIndicesNum), dtype = grad.dataType), spShape)
//    val spMat = tf.sparseSlice(spMatFull, (1, 0), (inputIndicesNum-1, outputIndicesNum))
//    val gradExpanded = tf.transpose(tf.reshape(grad, (batchSize, rowsOut, colsOut, ksizeR, ksizeC, channels)), (1, 2, 3, 4, 0, 5))
//    val gradFlat = tf.reshape(gradExpanded, (-1, batchSize * channels))
//    val jac = tf.sparseTensorDenseMatmul(spMat, gradFlat)
//    val gradOut = tf.reshape(jac, (rowsIn, colsIn, batchSize, channels))
//    gradOut = tf.transpose(gradOut, (2, 0, 1, 3))
//    listOf(gradOut) //return@register
//  }
//  register("ScatterNd") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val indices = op.inputs[0]
//    val updatesGrad = tf.gatherNd(grad, indices)
//    listOf(null, updatesGrad, null) //return@register
//  }
//  register("ScatterNdNonAliasingAdd") { op, grad ->
//    val grad = grad[0]!!.toOutput()
//    val indices = op.inputs[1]
//    val updatesGrad = tf.gatherNd(grad, indices)
//    listOf(grad, null, updatesGrad) //return@register
//  }
}