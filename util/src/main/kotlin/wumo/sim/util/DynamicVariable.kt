package wumo.sim.util

/** [DynamicVariable]s provide a binding mechanism where the current
 *  value is found through dynamic scope, but where access to the
 *  variable itself is resolved through static scope.
 *
 *  The current value can be retrieved with the value method. New values
 *  should be pushed using the [withValue] method. Values pushed via
 *  `withValue` only stay valid while the `withValue`'s second argument, a
 *  parameterless closure, executes. When the second argument finishes,
 *  the variable reverts to the previous value.
 *
 *  ```
 *  someDynamicVariable.withValue(newValue) {
 *    // ... code called in here that calls value ...
 *    // ... will be given back the newValue ...
 *  }
 *  ```
 *
 *  Each thread gets its own stack of bindings.  When a
 *  new thread is created, the `DynamicVariable` gets a copy
 *  of the stack of bindings from the parent thread, and
 *  from then on the bindings for the new thread
 *  are independent of those for the original thread.
 *
 */
class DynamicVariable<T>(init: T) {
  val tl = object : InheritableThreadLocal<T>() {
    override fun initialValue() = init
  }
  
  var value: T
    /** Retrieve the current value */
    get() = tl.get()
    /** Change the currently bound value, discarding the old value.
     * Usually withValue() gives better semantics.
     */
    set(newval) = tl.set(newval)
  
  /** Set the value of the variable while executing the specified
   * block.
   *
   * @param newval The value to which to set the variable
   * @param block The code to evaluate under the new setting
   */
  inline fun <S> with(newval: T, block: () -> S): S {
    val oldval = value
    tl.set(newval)
    try {
      return block()
    } finally {
      tl.set(oldval)
    }
  }
  
  override fun toString() = "DynamicVariable($value)"
}