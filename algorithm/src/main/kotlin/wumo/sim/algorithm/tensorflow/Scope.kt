package wumo.sim.algorithm.tensorflow

typealias NameMap = HashMap<String, Int>

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
class Scope(val name_map: NameMap = NameMap(),
            val parentName: String = "", var device: String = "") {
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
  
  fun newSubscope(name: String, device: String = "") =
      if (name.isEmpty())
        Scope(this.name_map, this.parentName, device)
      else
        Scope(parentName = getUniqueFullName(name), device = device)
  
  inline fun with_device(dev: String, block: () -> Unit) {
    val tmp = device
    device = dev
    block()
    device = tmp
  }
}