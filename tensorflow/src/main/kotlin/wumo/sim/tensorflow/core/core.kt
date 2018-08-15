package wumo.sim.tensorflow.core

import org.bytedeco.javacpp.tensorflow
import wumo.sim.tensorflow.ops.Output

object core {
  internal val defaultGraph = Graph()
}

class ShapeMismatchException(msg: String) : Exception(msg)
class GraphMismatchException(msg: String) : Exception(msg)
class IllegalNameException(msg: String) : Exception(msg)
class InvalidDeviceException(msg: String) : Exception(msg)
class InvalidShapeException(msg: String) : Exception(msg)
class InvalidIndexerException(msg: String) : Exception(msg)
class OpBuilderUsedException(msg: String) : Exception(msg)
class CheckpointNotFoundException(msg: String) : Exception(msg)
class CancelledException(msg: String) : Exception(msg)
class UnknownException(msg: String) : Exception(msg)
class InvalidArgumentException(msg: String) : Exception(msg)
class DeadlineExceededException(msg: String) : Exception(msg)
class NotFoundException(msg: String) : Exception(msg)
class AlreadyExistsException(msg: String) : Exception(msg)
class PermissionDeniedException(msg: String) : Exception(msg)
class UnauthenticatedException(msg: String) : Exception(msg)
class ResourceExhaustedException(msg: String) : Exception(msg)
class FailedPreconditionException(msg: String) : Exception(msg)
class AbortedException(msg: String) : Exception(msg)
class OutOfRangeException(msg: String) : Exception(msg)
class UnimplementedException(msg: String) : Exception(msg)
class InternalException(msg: String) : Exception(msg)
class UnavailableException(msg: String) : Exception(msg)
class DataLossException(msg: String) : Exception(msg)
class InvalidDataTypeException(msg: String) : Exception(msg)

internal fun tensorflow.TF_Status.check() {
  val code = tensorflow.TF_GetCode(this)
  if (code == tensorflow.TF_OK) return
  val msg = tensorflow.TF_Message(this).string
  throw when (code) {
    tensorflow.TF_INVALID_ARGUMENT -> IllegalArgumentException(msg)
    tensorflow.TF_UNAUTHENTICATED, tensorflow.TF_PERMISSION_DENIED -> SecurityException(msg)
    tensorflow.TF_RESOURCE_EXHAUSTED, tensorflow.TF_FAILED_PRECONDITION -> IllegalStateException(msg)
    tensorflow.TF_OUT_OF_RANGE -> IndexOutOfBoundsException(msg)
    tensorflow.TF_UNIMPLEMENTED -> UnsupportedOperationException(msg)
    else -> Exception(msg)
  }
}

typealias TensorFunction = (Output) -> Output?