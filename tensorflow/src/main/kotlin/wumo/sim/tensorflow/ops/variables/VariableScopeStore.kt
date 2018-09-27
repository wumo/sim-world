package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.tf

/** A thread-local score for the current variable native and native counts.
 */
class VariableScopeStore {
  
  internal var scope = VariableScope(CreateNewOnly)
  /** Map with variable native names as keys and the corresponding use counts as values. */
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
    /**Returns the variable native store for current thread.*/
    val current get() = tf.currentGraph.variableScopeStore.value
  }
}