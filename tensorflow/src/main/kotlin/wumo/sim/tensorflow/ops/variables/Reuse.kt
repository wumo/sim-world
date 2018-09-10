package wumo.sim.tensorflow.ops.variables

sealed class Reuse
sealed class ReuseAllowed : Reuse()

object ReuseExistingOnly : ReuseAllowed() //-->True in tensorflow
object CreateNewOnly : Reuse()
object ReuseOrCreateNew : ReuseAllowed() //-->None in tensorflow