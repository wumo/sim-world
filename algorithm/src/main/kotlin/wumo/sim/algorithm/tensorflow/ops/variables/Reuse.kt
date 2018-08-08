package wumo.sim.algorithm.tensorflow.ops.variables

sealed class Reuse
sealed class ReuseAllowed : Reuse()

object ReuseExistingOnly : ReuseAllowed()
object CreateNewOnly : Reuse()
object ReuseOrCreateNew : ReuseAllowed()