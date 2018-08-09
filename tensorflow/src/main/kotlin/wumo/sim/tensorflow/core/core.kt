package wumo.sim.tensorflow.core

object core {
  internal val defaultGraph = Graph()
}

class ShapeMismatchException(msg: String) : Exception(msg)
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