import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.basic.*
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.INT32
import wumo.sim.tensorflow.types.INT64
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.i
import wumo.sim.util.ndarray.NDArray

fun register_array_grad() {
//  /**Gradients for operators defined in array_ops.py.*/
  register("Pack") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for pack op.*/
    tf.unstack(grad, num = op.attrLong("N").toInt(), axis = op.attrLong("axis").toInt())  //return@register
  }
  register("Unpack") { op, grad ->
    /**Gradient for unpack op.*/
    listOf(tf.stack(grad.map { it!!.toOutput() }, axis = op.attrLong("axis")))  //return@register
  }
  fun concatGradHelper(op: Op, grad: OutputLike, startValueIndex: Int, endValueIndex: Int, dimIndex: Int): List<OutputLike?> {
    /**Gradient for concat op.
    
    Args:
    op: An operation.
    grad: `Tensor` or `IndexedSlices` representing the gradients with respect
    to each output of the op.
    start_value_index: An integer index of the first value in the op.inputs.
    end_value_index: An integer index of the last value in the op.inputs.
    dim_index: An interger index of concat_dim or axis parameter in op.inputs.
    
    Returns:
    Tensors representing the partial gradients with respect to each input
    of the op.
    
    Raises:
    ValueError: if concat_dim/axis is not statically known.
     */
    
    fun extractInputShapes(inputs: List<Output>): List<Output> {
      /**Extract the shapes of a set of input tensors.*/
      val sizes = mutableListOf<Output>()
      var fullyKnown = true
      
      for (x in inputs) {
        val inputShape = tf.shape(x)
        if (inputShape.op.opType != "Const") {
          fullyKnown = false
          break
        }
        sizes += inputShape
      }
      return if (fullyKnown)
        sizes
      else
        tf.shapeN(inputs)
    }
    if ((op.inputs).size == 2) {
      return if (endValueIndex <= dimIndex)
        listOf(grad, null)
      else listOf(null, grad)
    }
    var concatDim = op.inputs[dimIndex]
    val inputValues = op.inputs.subList(startValueIndex, endValueIndex)
    val outGrads = mutableListOf<OutputLike?>()
    when (grad) {
      is Output -> {
        if (const_ops.isConstant(concatDim.op)) {
          val gradContext = control_flow_ops.getOutputContext(grad.op)
          val dimContext = control_flow_ops.getOutputContext(concatDim.op)
          if (dimContext != gradContext) {
            val value = constantValue<Any>(concatDim)
            concatDim = tf.const(value = value!!)
          }
        }
        val nonNegConcatDim = concatDim % tf.rank(inputValues[0])
        val shapes = extractInputShapes(inputValues)
        if (shapes.size > 16) {
          val sizes = tf.squeeze(tf.slice(tf.stack(shapes, axis = 1),
                                          tf.stack(listOf(nonNegConcatDim, tf.const(0))),
                                          tf.const(i(1, -1))))
          tf.split(grad, sizes, nonNegConcatDim)
        } else {
          val offset = tf.concatOffset(nonNegConcatDim, shapes)
          
          for ((begin, size) in offset.zip(shapes))
            outGrads += tf.slice(grad, begin, size)
        }
      }
      is IndexedSlices -> {
        val nonNegConcatDim = concatDim % tf.rank(inputValues[0])
        val concatDimStatic = constantValue<Any>(concatDim) as NDArray<Int>?
            ?: throw IllegalArgumentException("Can only compute IndexedSlices gradient with ")
        var realNumericAxis: Int = concatDimStatic.get()
        if (realNumericAxis < 0) {
          val rank = constantValue<Int>(tf.rank(inputValues[0]))
              ?: throw  IllegalArgumentException("Can only compute IndexedSlices gradient with " +
                                                     "negative concat_dim when first value rank is " +
                                                     "statically-known.")
          realNumericAxis %= rank.get()
        }
        val shapes = inputValues.map { tf.shape(it) }
        if (realNumericAxis > 0) {
          /**Create variables for iteratively slicing a dense gradients tensor.*/
          val shapeOfShape = tf.shape(shapes[0])
          val zero = tf.const(0)
          val mask = tf.concat(listOf(
              tf.fill(tf.expandDims(concatDim, zero), zero), tf.const(i(1)),
              tf.fill(shapeOfShape - concatDim - 1, zero)),
                               zero)
          var begin = tf.fill(shapeOfShape, zero)
          
          for (size in shapes) {
            val newValues = tf.slice(grad.values, begin,
                                     tf.concat(listOf(tf.const(i(-1)),
                                                      tf.slice(size, tf.const(i(1)), tf.const(i(-1)))),
                                               zero))
            outGrads += IndexedSlices(grad.indices, newValues, size)
            begin = tf.add(begin, size * mask)
          }
        } else {
          var start = tf.const(grad.indices.dataType, 0)
          
          for (size in shapes) {
            var sizeConcatDim = tf.gather(size, nonNegConcatDim)
            if (sizeConcatDim.dataType != grad.indices.dataType)
              sizeConcatDim = tf.cast(sizeConcatDim, grad.indices.dataType)
            val end = start + sizeConcatDim
            val indicesToSelect = tf.squeeze(
                tf.where(tf.logicalAnd(tf.greaterEqual(grad.indices, start), tf.less(grad.indices, end))),
                arrayOf(1L))
            val newIndices = tf.gather(grad.indices, indicesToSelect) - start
            val newValues = tf.gather(grad.values, indicesToSelect)
            outGrads += IndexedSlices(newIndices, newValues, size)
            start = end
          }
        }
        
      }
      else -> throw IllegalArgumentException(
          "Only 'Output' and 'IndexedSlices' gradients are supported for the concatenation op.")
    }
    return if (endValueIndex <= dimIndex)
      outGrads + listOf(null)
    else
      listOf(null) + outGrads
  }
  register("Concat") { op, grad ->
    val grad = grad[0]!!.toOutput()
    concatGradHelper(op, grad, startValueIndex = 1, endValueIndex = (op.inputs).size, dimIndex = 0)  //return@register
  }
  register("ConcatV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    concatGradHelper(op, grad, startValueIndex = 0, endValueIndex = op.inputs.lastIndex - 1, dimIndex = -1) //return@register
  }
  registerNonDifferentiable("ConcatOffset")
  register("Slice") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for Slice op.*/
    val inputVec = op.inputs[0]
    val beginVec = op.inputs[1]
    val inputRank = tf.rank(inputVec)
    val sliceSize = tf.shape(op.outputs[0])
    val one = tf.const(1)
    val shape = tf.stack(listOf(inputRank, one))
    val beforePad = tf.reshape(beginVec, shape)
    val afterPad = tf.reshape(tf.shape(inputVec) - sliceSize - beginVec, shape)
    val paddings = tf.concat(listOf(beforePad, afterPad), one)
    listOf(tf.pad(grad, paddings), null, null)  //return@register
  }
  register("StridedSlice") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for StridedSlice op.*/
    val begin = op.inputs[1]
    val end = op.inputs[2]
    val strides = op.inputs[3]
    val x = tf.shape(op.inputs[0], outType = begin.dataType)
    listOf(tf.stridedSliceGrad(x,
                               begin,
                               end,
                               strides,
                               grad,
                               beginMask = op.attrLong("begin_mask"),
                               endMask = op.attrLong("end_mask"),
                               ellipsisMask = op.attrLong("ellipsis_mask"),
                               newAxisMask = op.attrLong("new_axis_mask"),
                               shrinkAxisMask = op.attrLong("shrink_axis_mask")),
           null, null, null)  //return@register
  }
  register("StridedSliceGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for StridedSliceGrad op.*/
    val begin = op.inputs[1]
    val end = op.inputs[2]
    val strides = op.inputs[3]
    listOf(null, null, null, null,
           tf.stridedSlice(grad, begin, end, strides,
                           beginMask = op.attrLong("begin_mask"),
                           endMask = op.attrLong("end_mask"),
                           ellipsisMask = op.attrLong("ellipsis_mask"),
                           newAxisMask = op.attrLong("new_axis_mask"),
                           shrinkAxisMask = op.attrLong("shrink_axis_mask")))  //return@register
  }
  register("Split") { op, grads ->
    listOf(null, tf.concat(grads.map { it!!.toOutput() }, op.inputs[0]))  //return@register
  }
  register("SplitV") { op, grads ->
    val returnval = tf.concat(grads.map { it!!.toOutput() }, op.inputs[2])
    MutableList<OutputLike?>(op.inputs.size) { null }.apply { this[0] = returnval }
  }
  registerNonDifferentiable("Const")
  register("Diag") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.diagPart(grad))  //return@register
  }
  register("DiagPart") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.diag(grad))  //return@register
  }
  register("MatrixDiag") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.matrixDiagPart(grad))  //return@register
  }
  register("MatrixDiagPart") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val matrixShape = op.inputs[0].shape[-2..-1]
    if (matrixShape.isFullyDefined && matrixShape[0] == matrixShape[1])
      listOf(tf.matrixDiag(grad))  //return@register
    else
      listOf(tf.matrixSetDiag(tf.zerosLike(op.inputs[0]), grad))  //return@register
  }
  register("MatrixSetDiag") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for MatrixSetDiag.*/
    val inputShape = op.inputs[0].shape.mergeWith(grad.shape)
    val diagShape = op.inputs[1].shape
    val batchShape = inputShape[0 until -2].mergeWith(diagShape[0 until -1])
    val matrixShape = inputShape[-2..-1]
    val gradInput = if (batchShape.isFullyDefined && matrixShape.isFullyDefined) {
      val diagShape = Shape(batchShape.asIntArray()!! + matrixShape.asIntArray()!!.min()!!)
      tf.matrixSetDiag(grad, tf.zeros(diagShape, dtype = grad.dataType))
    } else {
      tf.colocateWith(grad) {
        val gradShape = tf.shape(grad)
        val gradRank = tf.rank(grad)
        val batchShape = tf.slice(gradShape, tf.const(0), gradRank - 2)
        val matrixShape = tf.slice(gradShape, gradRank - 2, tf.const(2))
        val minDim = tf.min(matrixShape)
        val diagShape = tf.concat(listOf(batchShape, minDim), tf.const(0))
      }
      tf.matrixSetDiag(grad, tf.zeros(diagShape, dtype = grad.dataType))
    }
    
    val gradDiag = tf.matrixDiagPart(grad)
    listOf(gradInput, gradDiag)  //return@register
  }
  register("MatrixBandPart") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val numLower = op.inputs[1]
    val numUpper = op.inputs[2]
    listOf(tf.matrixBandPart(grad, numLower, numUpper), null, null)  //return@register
  }
  registerNonDifferentiable("EditDistance")
  register("Fill") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(null, tf.sum(grad, null as Output?))  //return@register
  }
  registerNonDifferentiable("ZerosLike")
  registerNonDifferentiable("OnesLike")
  register("PreventGradient") { op, grad ->
    val grad = grad[0]!!.toOutput()
    throw IllegalArgumentException("Gradient explicitly disabled. Reason: ${op.attrString("message")}.")
    
  }
  register("Gather") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for Gather op.*/
    val params = op.inputs[0]
    val paramsShape = tf.colocateWith(params) {
      val paramsShape = tf.shape(params, outType = INT64)
      tf.cast(paramsShape, INT32)
    }
    var indices = op.inputs[1]
    val size = tf.expandDims(tf.size(indices), tf.const(0))
    val valuesShape = tf.concat(listOf(size, paramsShape.slice(1)), tf.const(0))
    val values = tf.reshape(grad, valuesShape)
    indices = tf.reshape(indices, size)
    listOf(IndexedSlices(indices, values, paramsShape), null) //return@register
  }
  register("GatherV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for GatherV2 op.*/
    val params = op.inputs[0]
    val paramsShape = tf.colocateWith(params) {
      val paramsShape = tf.shape(params, outType = INT64)
      tf.cast(paramsShape, INT32)
    }
    var indices = op.inputs[1]
    val indicesSize = tf.expandDims(tf.size(indices), tf.const(0))
    val axis = op.inputs[2]
    val axisStatic = constantValue<Int>(axis)
    if (axisStatic?.get() == 0) {
      val paramsTailShape = paramsShape.slice(1)
      val valuesShape = tf.concat(listOf(indicesSize, paramsTailShape), tf.const(0))
      val values = tf.reshape(grad, valuesShape)
      indices = tf.reshape(indices, indicesSize)
      return@register listOf(IndexedSlices(indices, values, paramsShape), null, null) //return@register
    }
    val outerShape = paramsShape.slice(0, axis)
    val outerDims = tf.size(outerShape)
    val innerShape = paramsShape.slice(axis).slice(1)
    val innerDims = tf.size(innerShape)
    
    val zero = tf.const(0)
    val outerAxesIndices = tf.range(zero, outerDims)
    val innerAxesIndices = tf.range(outerDims + 1, outerDims + 1 + innerDims)
    
    val valuesShape = tf.concat(listOf(outerShape, indicesSize, innerShape), zero)
    val values = tf.reshape(grad, valuesShape)
    indices = tf.reshape(indices, indicesSize)
    
    val transposeDims = tf.concat(listOf(tf.expandDims(outerDims, zero), outerAxesIndices, innerAxesIndices), zero)
    val valuesTranspose = tf.transpose(values, transposeDims)
    val numSegments = paramsShape[axis]
    var paramsGrad = tf.unsortedSegmentSum(valuesTranspose, indices, numSegments)
    val invertTransposeDims = tf.concat(listOf(outerAxesIndices + 1, zero, innerAxesIndices), zero)
    paramsGrad = tf.transpose(paramsGrad, invertTransposeDims)
    listOf(paramsGrad, null, null) //return@register
  }
  register("GatherNd") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val ref = op.inputs[0]
    val indices = op.inputs[1]
    val refShape = tf.shape(ref, outType = indices.dataType)
    val refGrad = if (indices.shape.rank == 2 && indices.shape[-1] == 1)
      IndexedSlices(tf.squeeze(indices, a(-1)), grad, refShape)
    else
      tf.scatterNd(indices, grad, refShape)
    listOf(refGrad, null) //return@register
  }
  register("CheckNumerics") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Gradient for check_numerics op.*/
    listOf(tf.checkNumerics(grad, "Not a number (NaN) or infinity (Inf) values detected in gradient."))  //return@register
  }
  register("PlaceholderWithDefault", "Identity") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad)  //return@register
  }
  register("RefIdentity") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad)  //return@register
  }
  register("IdentityN") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad)  //return@register
  }
  registerNonDifferentiable("StopGradient")
  register("Reshape") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.reshape(grad, tf.shape(op.inputs[0])), null) //return@register
  }
  registerNonDifferentiable("InvertPermutation")
  fun reshapeToInput(op: Op, grad: Output): Output {
    /**Reshapes the gradient to the shape of the original input.*/
    return tf.reshape(grad, tf.shape(op.inputs[0]))
  }
  register("ExpandDims") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(reshapeToInput(op, grad), null) //return@register
  }
  register("Squeeze") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(reshapeToInput(op, grad))  //return@register
  }
  register("Transpose") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Returns unshuffle(grad).*/
    val p = op.inputs[1]
    listOf(tf.transpose(grad, tf.invertPermutation(p)), null) //return@register
  }
  register("ConjugateTranspose") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Returns conj(unshuffle(grad)).*/
    val p = op.inputs[1]
    listOf(tf.transpose(grad, tf.invertPermutation(p), conjugate = true), null) //return@register
  }
  registerNonDifferentiable("Shape")
  registerNonDifferentiable("ShapeN")
  registerNonDifferentiable("Rank")
  registerNonDifferentiable("Size")
  register("Tile") { op, grad ->
    var grad = grad[0]!!
    /**Sum reduces grad along the tiled dimensions.*/
    val inputShape = tf.shape(op.inputs[0])
    var splitShape = tf.reshape(tf.transpose(tf.stack(listOf(op.inputs[1], inputShape))), tf.const(i(-1)))
    val zero = tf.const(0)
    val axes = tf.range(zero, tf.size(splitShape), tf.const(2))
    if (grad is IndexedSlices) {
      grad = tf.unsortedSegmentSum(grad.values, tf.mod(grad.indices, inputShape[0]), inputShape[0])
      splitShape = tf.concat(listOf(tf.const(1), splitShape.slice(1)), axis = zero)
    }
    val inputGrad = tf.sum(tf.reshape(grad as Output, splitShape), axes)
    inputGrad.setShape(op.inputs[0].shape)
    listOf(inputGrad, null) //return@register
  }
  registerNonDifferentiable("BroadcastGradientArgs")
  fun padGrad(op: Op, grad: Output): List<OutputLike?> {
    /**Gradient for Pad.*/
    val x = op.inputs[0]
    val a = op.inputs[1]
    val padBefore = tf.slice(a, tf.const(i(0, 0)), tf.stack(listOf(tf.rank(x), 1)))
    val begin = tf.reshape(padBefore, tf.const(i(-1)))
    val sizes = tf.shape(x)
    val xGrad = tf.slice(grad, begin, sizes)
    return if (op.inputs.size == 3)
      listOf(xGrad, null, null)
    else
      listOf(xGrad, null)
  }
  register("Pad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    padGrad(op, grad)
  }
  register("PadV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    padGrad(op, grad)
  }
  register("ReverseSequence") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val seqLengths = op.inputs[1]
    listOf(tf.reverseSequence(grad,
                              batchDim = op.attrLong("batch_dim"),
                              seqDim = op.attrLong("seq_dim"),
                              seqLengths = seqLengths), null) //return@register
  }
  register("Reverse") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val reverseDims = op.inputs[1]
    listOf(tf.reverse(grad, reverseDims), null)  //return@register
  }
  register("ReverseV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val axis = op.inputs[1]
    listOf(tf.reverseV2(grad, axis), null)  //return@register
  }
  register("SpaceToBatch") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val blockSize = op.attrLong("block_size")
    listOf(tf.batchToSpace(grad, op.inputs[1], blockSize = blockSize), null) //return@register
  }
  register("SpaceToBatchND") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.batchToSpaceND(grad, op.inputs[1], op.inputs[2]), null, null) //return@register
  }
  register("BatchToSpace") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val blockSize = op.attrLong("block_size")
    listOf(tf.spaceToBatch(grad, op.inputs[1], blockSize = blockSize), null) //return@register
  }
  register("BatchToSpaceND") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(tf.spaceToBatchND(grad, op.inputs[1], op.inputs[2]), null, null) //return@register
  }
  register("SpaceToDepth") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val blockSize = op.attrLong("block_size")
    val dataFormat = op.attrString("data_format")
    if (dataFormat == "NCHW_VECT_C")
      throw InvalidArgumentException("Cannot compute SpaceToDepth gradient with NCHW_VECT_C. " +
                                         "NCHW_VECT_C requires qint8 data type.")
    listOf(tf.depthToSpace(grad, blockSize, dataFormat = dataFormat))  //return@register
  }
  register("DepthToSpace") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val blockSize = op.attrLong("block_size")
    val dataFormat = op.attrString("data_format")
    if (dataFormat == "NCHW_VECT_C")
      throw InvalidArgumentException("Cannot compute DepthToSpace gradient with NCHW_VECT_C. " +
                                         "NCHW_VECT_C requires qint8 data type.")
    listOf(tf.spaceToDepth(grad, blockSize, dataFormat = dataFormat))  //return@register
  }
  registerNonDifferentiable("OneHot")
  register("MirrorPad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val mode = op.attrString("mode")
    listOf(tf.mirrorPadGrad(grad, op.inputs[1], mode = mode), null) //return@register
  }
  register("MirrorPadGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val mode = op.attrString("mode")
    listOf(tf.mirrorPad(grad, op.inputs[1], mode = mode), null) //return@register
  }
  register("QuantizeAndDequantize") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad)  //return@register
  }
  register("QuantizeAndDequantizeV2") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad, null, null) //return@register
  }
  register("QuantizeAndDequantizeV3") { op, grad ->
    val grad = grad[0]!!.toOutput()
    listOf(grad, null, null, null) //return@register
  }
  register("ExtractImagePatches") { op, grad ->
    TODO()
//    val grad = grad[0]!!.toOutput()
//
//    var (_, rowsIn, colsIn, _) = op.inputs[0].shape.map { it }
//    val inputBhwc = tf.shape(op.inputs[0])
//    val batchSize = inputBhwc[0]
//    val channels = inputBhwc[3]
//    val inputIndicesNum = 1 + rowsIn * colsIn
//    val inputIdx = tf.reshape(tf.range(tf.const(1), tf.const(inputIndicesNum), dtype = INT64), (1, rowsIn, colsIn, 1))
//    val inputIdxPatched = tf.extractImagePatches(inputIdx,
//                                                 op.att("ksizes"),
//                                                 op.getAttr("strides"),
//                                                 op.getAttr("rates"),
//                                                 op.getAttr("padding"))
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
  }
  register("ScatterNd") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val indices = op.inputs[0]
    val updatesGrad = tf.gatherNd(grad, indices)
    listOf(null, updatesGrad, null) //return@register
  }
  register("ScatterNdNonAliasingAdd") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val indices = op.inputs[1]
    val updatesGrad = tf.gatherNd(grad, indices)
    listOf(grad, null, updatesGrad) //return@register
  }
}