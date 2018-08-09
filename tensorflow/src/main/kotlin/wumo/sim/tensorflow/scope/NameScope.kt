@file:Suppress("OVERRIDE_BY_INLINE", "NOTHING_TO_INLINE")

package wumo.sim.tensorflow.scope

import wumo.sim.tensorflow.scopeChar

class NameScope(val name: String, parentScope: NameScope? = null) : enter_exit {
  /**这一层已使用的name，[subscopes]的name是[name_map]的子集*/
  private val name_map = HashMap<String, Int>()
  /**下一层的[NameScope]*/
  private val subscopes = HashMap<String, NameScope>()
  /**本层的全称以$特殊标志*/
  val scopeName: String = if (parentScope == null) name else concate(parentScope.scopeName, name)
  
  private fun getUniqueName(prefix: String): String {
    var unique_name = prefix
    name_map.compute(prefix) { _, n ->
      if (n == null)
        0
      else
        (n + 1).apply { unique_name = "${unique_name}_$this" }
    }
    return unique_name
  }
  
  private fun concate(a: String, b: String): String {
    assert(a.isNotEmpty() && b.isNotEmpty())
    return "$a/$b"
  }
  
  /**
   * 新建subscope
   * @param name subscope的名字，不能为空
   */
  fun new_subscope(name: String): NameScope {
    assert(name.isNotEmpty() && !name.startsWith(scopeChar))
    val subname = getUniqueName(name)
    val sub = NameScope(subname, this)
    subscopes[subname] = sub
    return sub
  }
  
  /**
   * 寻找名为[name]的subscope，如果存在则直接返回，否则新建subscope，并重命名
   */
  internal fun reuse_or_new_subscope(name: String): NameScope {
    assert(name.isNotEmpty() && !name.startsWith(scopeChar))
    return subscopes.computeIfAbsent(name) { NameScope(name, this) }
  }
}