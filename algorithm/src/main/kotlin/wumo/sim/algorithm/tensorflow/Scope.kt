package wumo.sim.algorithm.tensorflow

import sun.audio.AudioDevice.device
import wumo.sim.algorithm.tensorflow.ops.CondContext
import java.util.*
import kotlin.collections.HashMap

typealias NameMap = HashMap<String, Int>

interface ControlFlowContext {
  fun addOp(op: Operation)
}

/**
 * @param name_map A NameMap is used to keep track of suffixes for names used in a scope. A
 * parentName that has not been used so far in a scope will get no suffix. Later
 * uses of the same parentName will get suffixes _1, _2, _3, etc. Multiple scopes
 * can share the same NameMap. For instance, a new scope created using
 * WithControlDependencies() should would share the same NameMap with the
 * parent.
 * @param parentName The fully-qualified parentName of this scope (i.e. includes any parent scope names).
 *
 */
class Scope(val name_map: NameMap = NameMap(), val subscopes: HashMap<String, Scope> = HashMap(),
            val parentName: String = "", var device: String = "", val reuse: Boolean = false) {
  private var useParentScopeName = false
  private var usedOnce = false
  var colocate_with: Operation? = null
  val control_ops = ArrayDeque<Operation>()
  var control_flow_ctx: ControlFlowContext? = null
  
  fun getUniqueName(prefix: String): String {
    var unique_name = prefix
    name_map.compute(prefix) { _, n ->
      if (n == null)
        0
      else
        (n + 1).apply { unique_name = "${unique_name}_$this" }
    }
    return unique_name
  }
  
  /**需要确保此函数调用后，[getUniqueFullName]函数只调用一次，否则报错*/
  fun borrowParentName(): String {
    useParentScopeName = true
    return ""
  }
  
  fun getUniqueFullName(default_name: String): String {
    if (useParentScopeName) {
      assert(!usedOnce)
      usedOnce = true
      return parentName
    }
    val unique_name = getUniqueName(default_name)
    val sep = if (parentName.isEmpty() || unique_name.isEmpty()) "" else "/"
    return "$parentName$sep$unique_name"
  }
  
  fun newSubscope(name: String, device: String = "", reuse: Boolean = false) =
      if (name.isEmpty())
        Scope(this.name_map, this.subscopes, this.parentName, device, reuse)
      else
        subscopes.computeIfAbsent(name) { Scope(parentName = getUniqueFullName(name), device = device, reuse = reuse) }
  
  inline fun <R> with_device(dev: String, block: () -> R): R {
    val tmp = device
    device = dev
    try {
      return block()
    } finally {
      device = tmp
    }
  }
  
  inline fun <R> colocate_with(colocate_with: Tensor, block: () -> R) =
      colocate_with(colocate_with.op!!, block)
  
  inline fun <R> colocate_with(colocate_with: Operation, block: () -> R): R {
    val tmp = this.colocate_with
    this.colocate_with = colocate_with
    try {
      return block()
    } finally {
      this.colocate_with = tmp
    }
  }
  
  inline fun <R> control_dependencies(control_inputs: List<Operation>, block: () -> R): R {
    val size = control_inputs.size
    control_ops += control_inputs
    try {
      return block()
    } finally {
      repeat(size) {
        control_ops.removeLast()
      }
    }
  }
  
  /**
   * Creates a `CondContext`.
   *
   * @param pred  The `boolean` tensor for the conditional predicate.
   * @param pivot The predicate tensor in this branch.
   * @param branch 0 or 1 representing this branch.
   */
  inline fun <R> condCtx(pred: Tensor, pivot: Tensor, branch: Int, block: () -> R): R {
    control_flow_ctx = CondContext(pred, pivot, branch)
    try {
      return block()
    } finally {
      control_flow_ctx = null
    }
  }
}