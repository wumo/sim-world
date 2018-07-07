package wumo.sim.algorithm.tensorflow

typealias NameMap = HashMap<String, Int>

/**
 * @param name_map A NameMap is used to keep track of suffixes for names used in a scope. A
 * name that has not been used so far in a scope will get no suffix. Later
 * uses of the same name will get suffixes _1, _2, _3, etc. Multiple scopes
 * can share the same NameMap. For instance, a new scope created using
 * WithControlDependencies() should would share the same NameMap with the
 * parent.
 * @param name The fully-qualified name of this scope (i.e. includes any parent scope names).
 *
 */
class Scope(val name_map: NameMap = NameMap(),
            val name: String = "") {
  var useName = false
  
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
  
  fun useContextName(): String {
    useName = true
    return ""
  }
  
  fun getUniqueFullName(default_name: String): String {
    if (useName) return name
    val unique_name = getUniqueName(default_name)
    val sep = if (name.isEmpty() || unique_name.isEmpty()) "" else "/"
    return "$name$sep$unique_name"
  }
  
  fun newSubscope(name: String) = Scope(name = getUniqueFullName(name))
}