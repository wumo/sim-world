package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.ops.ops

/** A thread-local score for the current variable scope and scope counts.
 */
class VariableScopeStore {
  var scope = VariableScope(CreateNewOnly)
  
  /** Map with variable scope names as keys and the corresponding use counts as values. */
  var variableScopeCounts = mutableMapOf<String, Int>()
  
  fun enterVariableScope(scope: String) {
    variableScopeCounts.compute(scope) { _, v ->
      val visits = (v ?: 0)
      visits + 1
    }
  }
  
  fun closeVariableSubScopes(scope: String) {
    variableScopeCounts.keys.removeIf { it.startsWith("$scope/") }
  }
  
  fun variableScopeCount(scope: String) = variableScopeCounts.getOrDefault(scope, 0)
  
  companion object {
    val current get() = ops.currentGraph.variableScopeStore.value
  }
}