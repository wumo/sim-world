
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.slice
import wumo.sim.tensorflow.ops.basic.toOutput
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DOUBLE
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.FLOAT16

fun register_image_grad() {
  /**Contains Gradient functions for image ops.*/
  register("ResizeNearestNeighbor") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**The derivatives for nearest neighbor resizing.
    
    Args:
    op: The ResizeNearestNeighbor op.
    grad: The tensor representing the gradient w.r.t. the output.
    
    Returns:
    The gradients w.r.t. the input and the output.
     */
    val image = op.inputs[0]
    val imageShape = if (image.shape[1 until 3].isFullyDefined)
      image.shape[1 until 3].toOutput()
    else
      tf.shape(image).slice(1, 3)
    val grads = tf.resizeNearestNeighborGrad(grad, imageShape,
                                             alignCorners = op.attrBool("align_corners"))
    listOf(grads, null) //return@register
  }
  register("ResizeBilinear") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**The derivatives for bilinear resizing.
    
    Args:
    op: The ResizeBilinear op.
    grad: The tensor representing the gradient w.r.t. the output.
    
    Returns:
    The gradients w.r.t. the input.
     */
    val grad0 = tf.resizeBilinearGrad(grad, op.inputs[0],
                                      alignCorners = op.attrBool("align_corners"))
    listOf(grad0, null) //return@register
  }
  register("ResizeBicubic") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**The derivatives for bicubic resizing.
    
    Args:
    op: The ResizeBicubic op.
    grad: The tensor representing the gradient w.r.t. the output.
    
    Returns:
    The gradients w.r.t. the input.
     */
    val allowedTypes = listOf(FLOAT, DOUBLE)
    var grad0: Output? = null
    if (op.inputs[0].dataType in allowedTypes)
      grad0 = tf.resizeBicubicGrad(grad, op.inputs[0], alignCorners = op.attrBool("align_corners"))
    listOf(grad0, null) //return@register
  }
  register("CropAndResize") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**The derivatives for crop_and_resize.
    
    We back-propagate to the image only when the input image tensor has floating
    point dtype but we always back-propagate to the input boxes tensor.
    
    Args:
    op: The CropAndResize op.
    grad: The tensor representing the gradient w.r.t. the output.
    
    Returns:
    The gradients w.r.t. the input image, boxes, as well as the always-None
    gradients w.r.t. box_ind and crop_size.
     */
    val image = op.inputs[0]
    val imageShape = if (image.shape.isFullyDefined)
      image.shape.toOutput()
    else
      tf.shape(image)
    val allowedTypes = listOf(FLOAT16, FLOAT, DOUBLE)
    val grad0 = if (op.inputs[0].dataType in allowedTypes)
      tf.cropAndResizeGradImage(grad, op.inputs[1], op.inputs[2], imageShape,
                                t = op.attrDataType("T"),
                                method = op.attrString("method"))
    else
      null
    val grad1 = tf.cropAndResizeGradBoxes(grad, op.inputs[0], op.inputs[1], op.inputs[2])
    listOf(grad0, grad1, null, null) //return@register
  }
}