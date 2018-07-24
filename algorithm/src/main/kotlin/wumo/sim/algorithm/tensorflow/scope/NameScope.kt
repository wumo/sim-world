@file:Suppress("OVERRIDE_BY_INLINE", "NOTHING_TO_INLINE")

package wumo.sim.algorithm.tensorflow.scope

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import java.util.*
import kotlin.collections.HashMap

class NameScope(val name: String, parentScope: NameScope?) : enter_exit {
  /**这一层已使用的name，[subscopes]的name是[name_map]的子集*/
  private val name_map = HashMap<String, Int>()
  /**下一层的[NameScope]*/
  private val subscopes = HashMap<String, NameScope>()
  /**本层的全称*/
  val fullName: String = if (parentScope == null) name else concate(parentScope.fullName, name)
  private var useParentScopeName = false
  private var usedOnce = false
  
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
  
  fun getUniqueFullName(default_name: String): String {
    if (useParentScopeName) {
      assert(!usedOnce)
      usedOnce = true
      return fullName
    }
    val unique_name = getUniqueName(default_name)
    return concate(fullName, unique_name)
  }
  
  private fun concate(a: String, b: String): String {
    val sep = if (a.isEmpty() || b.isEmpty()) "" else "/"
    return "$a$sep$b"
  }
  
  /**需要确保此函数调用后，[getUniqueFullName]函数只调用一次，否则报错*/
  fun scopeNameForOnce(): String {
    useParentScopeName = true
    return ""
  }
  
  var device: String = ""
  var colocate_with: Operation? = null
  val control_ops = ArrayDeque<Operation>()
  
  /**
   * 新建subscope
   * @param name subscope的名字，不能为空
   */
  fun new_subscope(name: String): NameScope {
    assert(name.isNotEmpty())
    val subname = getUniqueName(name)
    val sub = NameScope(subname, this)
    subscopes[subname] = sub
    return sub
  }
  
  /**
   * 寻找名为[name]的subscope，如果存在则直接返回，否则新建subscope，并重命名
   */
  internal fun reuse_or_new_subscope(name: String): NameScope {
    assert(name.isNotEmpty())
    return subscopes.computeIfAbsent(name) { NameScope(name, this) }
  }
  
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
  
  inline fun <R> control_dependencies(vararg control_inputs: Operation, block: () -> R): R {
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
}