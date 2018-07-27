package wumo.sim.algorithm.tensorflow.scope

import wumo.sim.algorithm.tensorflow.Variable

/**
 * @param namescope 此[VariableScope]对应的[NameScope]，通常是复用的
 */
class VariableScope(val name: String, val namescope: NameScope) : enter_exit {
  override fun enter() {
    var_scope_count.clear()
  }
  
  override fun exit() {
    var_scope_count.clear()
  }
  
  /**是否允许重复进入[VariableScope]时自动递增后缀*/
  var reenter_increment = false
  var reuse = false
  /***/
  private val var_scope_count = HashMap<String, Int>()
  /**这一层存储的所有变量*/
  val variables = HashMap<String, Variable>()
  /**这一层下的sub variable scope, */
  val variable_subscopes = HashMap<String, VariableScope>()
  
  fun all_variables(): List<Variable> {
    val a_variables = mutableListOf<Variable>()
    a_variables += variables.values
    for (variableScope in variable_subscopes.values) {
      a_variables += variableScope.all_variables()
    }
    return a_variables
  }
  
  /**
   * 在[enter]后，[exit]之前，调用[variable_scope]进入
   * 某个[name]的sub[VariableScope]时，第一次进入时复用已有的
   * [VariableScope]，之后再次进入重复[name]的sub[VariableScope]时，
   * 则自动递增[name]的后缀,递增条件如下：
   *
   * 1. 如[name]="a"，而之前为访问过，则返回"a"，记录访问过1次;
   * 2. 如果"a"之前访问过1次，则更改为"a_1"，记录访问过2次；
   * 3. 如果"a"之前访问过2此，则更改为"a_2"，记录访问过3次
   *
   * [exit]之后，计数清空，重新开始
   */
  fun variable_scope(name: String, reuse: Boolean, reenter_increment: Boolean): VariableScope {
    assert(name.isNotEmpty())
    val suffix =
        if (!reenter_increment) ""
        else {
          val visits = var_scope_count.compute(name) { _, v ->
            val visits = (v ?: 0)
            visits + 1
          }!!
          if (visits == 1) "" else "_${visits - 1}"
        }
    val real_name = "$name$suffix"
    return variable_subscopes.compute(real_name) { _, v ->
      v ?: VariableScope(real_name, namescope.reuse_or_new_subscope(real_name))
    }!!.apply {
      this.reuse = reuse
      this.reenter_increment = reenter_increment
    }
  }
}