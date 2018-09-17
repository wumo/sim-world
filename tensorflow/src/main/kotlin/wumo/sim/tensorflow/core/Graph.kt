@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.tensorflow.core

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.SizeTPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Buffer.newBuffer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Graph.newGraph
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.GraphDef
import org.tensorflow.framework.OpDef
import wumo.sim.tensorflow.OperationBuilder
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.Saver
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.ops.variables.VariableScopeStore
import wumo.sim.tensorflow.ops.variables.VariableStore
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.util.isNotNull
import wumo.sim.util.DynamicVariable
import java.util.*

/**
 * A TensorFlow computation, represented as a dataflow graph.
 *
 * A `Graph` contains a set of [Op] objects,
 * which represent units of computation; and
 * [Output] objects, which represent
 * the units of data that flow between operations.
 */
open class Graph {
  
  val c_graph = newGraph()!!
  
  val nextIdCounter: Int = 0
  
  /** Indicates whether this graph has been frozen (i.e., no more ops can be added to it). */
  var frozen = false
    private set
  
  /** Freezes this graph, meaning that no more ops can be added to it after a call to this function. This method is used
   * to ensure that no operations are added to a graph when it is shared between multiple threads. */
  fun freeze() {
    frozen = true
  }
  
  /** Unfreezes this graph. */
  fun unfreeze() {
    frozen = false
  }
  
  /** Asserts that this graph has not been frozen. */
  inline fun assertNotFrozen() {
    assert(!frozen) { "This graph has already been frozen." }
  }
  
  /** Map from native op handle to op object in the Kotlin side. Used for caching ops that have already been obtained
   * from the native library. */
  internal val opsCache = hashMapOf<Long, Op>()
  
  internal fun cache(op: tensorflow.TF_Operation) = opsCache[op]
  operator fun HashMap<Long, Op>.get(op: tensorflow.TF_Operation) =
      getOrPut(op.address()) { Op(this@Graph, op) }
  
  /** Variable store object of this graph, used to store created variables and keep track of variable scope usages. */
  internal val variableStore = VariableStore()
  /** Variable scope store object of this graph. */
  internal val variableScopeStore = DynamicVariable(VariableScopeStore())
  internal val variableCreatorStack = DynamicVariable(listOf<VariableGetter>())
  /** Set that contains the current names in use in this graph. */
  private val namesInUse = mutableMapOf<String, Int>()
  
  /** Map from collection key to set of values in that collection. */
  private val collections: MutableMap<Graph.Key<*>, MutableSet<*>> = mutableMapOf()
  /**Set of tensors that are dangerous to feed!*/
  private val unfeedableOutputs = mutableSetOf<Output>()
  /**Set of operations that are dangerous to fetch!*/
  private val unfetchableOps = mutableSetOf<Op>()
  
  private val functions = LinkedHashSet<String>()
  
  /** Returns a unique op name in this graph, based on the provided `name`.
   *
   * @note Operation names are displayed in error messages reported by the TensorFlow runtime, and in various
   *       visualization tools such as TensorBoard.
   * @note You rarely need to call `uniqueName` directly. Most of the time you just need to create
   *       `Op.createWithNameScope(...)` (which is also thread-safe) blocks to generate structured names.
   * @param  name       Name in which to base the generated unique name.
   * @param  markAsUsed If `true`, which is the default, a new unique name is created and marked as in use. If `false`,
   *                    the unique name is returned without actually being marked as used. This is useful when the
   *                    caller simply wants to know what the name to be created will be.
   * @return Unique name.
   * @see "tensorflow.python.framework.ops.Graph#unique_name"
   */
  internal fun uniqueName(name: String, markAsUsed: Boolean = true): String {
    val nameScope = ops.convertNameScopeToName(tf.currentNameScope)
    val fullName = if (nameScope == "") name
    else "$nameScope/$name"
    var count = namesInUse.getOrDefault(fullName, 0)
    //Increment the counter for the provided name.
    if (markAsUsed)
      namesInUse[fullName] = count + 1
    return if (count > 0) {
      var uniqueName = fullName
      //Make sure the composed name is not already being used.
      while (uniqueName in namesInUse) {
        uniqueName = "${fullName}_$count"
        count++
      }
      // Mark the composed name_key as used in case someone wants
      // to call unique_name("name_1").
      if (markAsUsed)
        namesInUse[uniqueName] = 1
      uniqueName
    } else
      fullName
  }
  
  /** Removes the specified collection from this graph.
   *
   * @param  key Collection key.
   */
  fun <K> clearCollection(key: Graph.Key<K>) {
    assertNotFrozen()
    collections -= key
  }
  
  /** Adds [value] to the collection with name [key].
   *
   * @param  value Value to add to the collection.
   * @param  key   Collection name.
   */
  fun <K> addToCollection(value: K, key: Graph.Key<K>) {
    assertNotFrozen()
    collections.compute(key) { _, _v ->
      val v = _v ?: mutableSetOf<K>()
      v as MutableSet<K>
      v += value
      v
    }
  }
  
  /** Gets the set of values contained in the collection with name `key`.
   *
   * Note that this method returns an immutable copy of the set.
   *
   * @param  key Collection name.
   * @return Set of values contained in the collection with name `collection`.
   */
  fun <K> getCollection(key: Graph.Key<K>) =
      collections.getOrElse(key) { mutableSetOf<K>() } as MutableSet<K>
  
  /**
   * Returns a list of values in the collection with the given `name`.
  
  This is different from `get_collection_ref()` which always returns the
  actual collection list if it exists in that it returns a new list each time
  it is called.
  
  Args:
   * @param key: The key for the collection. For example, the `GraphKeys` class
  contains many standard names for collections.
   * @param scope: (Optional.) A string. If supplied, the resulting list is filtered
  to include only items whose `name` attribute matches `scope` using
  [Regex.containsMatchIn]. Items without a `name` attribute are never returned if a
  scope is supplied. The choice of `re.match` means that a `scope` without
  special tokens filters by prefix.
   
   * @return:
  The list of values in the collection with the given `name`, or
  an empty list if no value has been added to that collection. The
  list contains the values in the order under which they were
  collected.
   */
  fun <K : HasName> getCollection(key: Graph.Key<K>, scope: String? = null): Set<K> {
    val collection = collections.getOrElse(key) { setOf<K>() } as Set<K>
    if (collection.isEmpty()) return collection
    return if (scope == null) collection
    else {
      val regex = Regex(scope)
      collection.filterTo(mutableSetOf()) {
        regex.containsMatchIn(it.name)
      }
    }
  }
  
  var randomSeed: Int?
    /** Gets the random seed of this graph. */
    get() = (collections.getOrPut(Graph.Keys.RANDOM_SEEDS) { mutableSetOf<Int>() } as Set<Int>)?.firstOrNull()
    /** Sets the random seed of this graph to the provided value. */
    set(value) {
      assertNotFrozen()
      value?.let {
        collections[Graph.Keys.RANDOM_SEEDS] = mutableSetOf(it)
      }
    }
  
  /** Returns the set of global variables in this graph.
   *
   * Global variables are variables that are shared across machines in a distributed environment. The `Variable()`
   * constructor and the function `getVariable()` automatically add new variables to the graph collection with key
   * `Graph.Keys.GLOBAL_VARIABLES`. This convenience function returns the contents of that collection.
   *
   * An alternative to global variables are local variables.
   */
  val globalVariables: Set<Variable>
    get() = getCollection(Graph.Keys.GLOBAL_VARIABLES)
  
  /** Returns the set of local variables in this graph.
   *
   * Local variables (or per-process variables), are usually not saved/restored to/from checkpoints and are used for
   * temporary or intermediate values. For example, they can be used as counters for metrics computations or number of
   * epochs this machine has read data. This convenience function returns the contents of that collection.
   *
   * An alternative to local variables are global variables.
   */
  val localVariables: Set<Variable> get() = getCollection(Graph.Keys.LOCAL_VARIABLES)
  
  /** Returns the subset of `Variable` objects that are used in models for inference (feed forward), in this graph. */
  val modelVariables: Set<Variable> get() = getCollection(Graph.Keys.MODEL_VARIABLES)
  
  /** Returns the set of metric variables in the current graph.
   *
   * Metric variables are usually not saved/restored to/from checkpoints and are used for temporary or intermediate
   * values used for computing metrics (e.g., streaming metrics). This convenience function returns the contents of
   * that collection.
   */
//  val metricVariables: Set<Variable> = getCollection(Metric.METRIC_VARIABLES)
  
  /** Returns the set of all variables created with `trainable = true`.
   *
   * When passed `trainable = true`, the `Variable()` constructor automatically adds new variables to the graph
   * collection with key `Graph.Keys.TRAINABLE_VARIABLES`. This convenience function returns the contents of that
   * collection.
   */
  val trainableVariables: Set<Variable> get() = getCollection(Graph.Keys.TRAINABLE_VARIABLES)
  
  /** Creates an op that returns a tensor containing the names of all uninitialized variables among all global and local
   * variables of this graph. If all variables have been initialized, then an empty tensor is returned.
   *
   * @param  name Name for the created op.
   * @return Created op output, which contains the names of the handles of all variables which have not yet been
   *         initialized.
   */
//  fun uninitializedVariables(name: String = "UninitializedVariables"): Output = {
//    Variable.uninitializedVariables(name = name)
//  }
  
  /** Returns the set of all the summary `Output`s that have been created in the graph. */
  val summaries: Set<Output> get() = getCollection(Graph.Keys.SUMMARIES)
  
  /** Returns the set of all the table initializers that have been created in the graph. */
  val tableInitializers: Set<Op> get() = getCollection(Graph.Keys.TABLE_INITIALIZERS)
  
  /** Returns the set of all savers that have been created in the graph. */
  val savers: Set<Saver> get() = getCollection(Graph.Keys.SAVERS)
  
  /** Returns the set of all shared resources used by the graph which need to be initialized once per cluster. */
  val sharedResources: Set<Resource> get() = getCollection(Graph.Keys.SHARED_RESOURCES)
  
  /** Returns the set of all local resources used by the graph which need to be initialized once per cluster. */
  val localResources: Set<Resource> get() = getCollection(Graph.Keys.LOCAL_RESOURCES)
  
  /** Creates an op that returns a tensor containing the names of all uninitialized resources among all shared and local
   * resources of this graph. If all resources have been initialized, then an empty tensor is returned.
   *
   * @param  name Name for the created op.
   * @return Created op output, which contains the names of the handles of all resources which have not yet been
   *         initialized.
   */
  fun uninitializedResources(name: String = "UninitializedResources"): Output = run {
    TODO()
//    Resources.uninitializedResources(name = name)
  }
  
  /** Returns the set of all the train `Op`s (i.e., optimizer update ops) that have been created in the graph. */
  val trainOps: Set<Op> get() = getCollection(Graph.Keys.TRAIN_OP)
  
  /** Returns an op that initializes all global variables of this graph.
   *
   * For more information, refer to [[globalVariables]] and [[Variable.initializer]].
   *
   * @param  name Name for the created op.
   * @return Created op.
   */
  fun globalVariablesInitializer(name: String = "global_variables_initializer"): Op =
      Variable.initializer(globalVariables)
  
  /** Returns an op that initializes all local variables of this graph.
   *
   * For more information, refer to [[localVariables]] and [[Variable.initializer]].
   *
   * @param  name Name for the created op.
   * @return Created op.
   */
  fun localVariablesInitializer(name: String = "local_variables_initializer"): Op =
      Variable.initializer(localVariables, name)
  
  /** Returns an op that initializes all model variables of this graph.
   *
   * For more information, refer to [[modelVariables]] and [[Variable.initializer]].
   *
   * @param  name Name for the created op.
   * @return Created op.
   */
  fun modelVariablesInitializer(name: String = "ModelVariablesInitializer"): Op =
      Variable.initializer(modelVariables, name)
  
  /** Returns an op that initializes all trainable variables of this graph.
   *
   * For more information, refer to [[trainableVariables]] and [[Variable.initializer]].
   *
   * @param  name Name for the created op.
   * @return Created op.
   */
  fun trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): Op =
      Variable.initializer(trainableVariables, name)
  
  /**
   * Returns the [OpDef] proto for [opType].
   */
  fun getOpDef(opType: String): OpDef {
    val buf = newBuffer()
    val status = newStatus()
    TF_GraphGetOpDef(c_graph, opType, buf, status)
    status.check()
    val data = buf.data()
    data.limit<Pointer>(buf.length())
    return OpDef.parseFrom(data.asByteBuffer())
  }
  
  fun num_node_ids() = c_graph.graph().num_node_ids()
  fun nodeBuilder(opType: String, name: String) = OperationBuilder(opType, name)
  fun isFetchable(op: Op) {
  }
  
  fun findOp(name: String): Op? {
    val op = TF_GraphOperationByName(c_graph, name)
    return if (op.isNull) null
    else opsCache[op]
  }
  
  fun ops(): List<Op> {
    val pos = SizeTPointer(1)
    val ops = arrayListOf<Op>()
    do {
      val op = TF_GraphNextOperation(c_graph, pos) ?: break
      ops += opsCache[op]
    } while (op.isNotNull)
    return ops
  }
  
  fun getTensor(name: String): Output {
    val parts = name.split(':')
    val opName = parts[0]
    val idx = if (parts.size > 1) parts[1] else "0"
    val valueIdx = idx.toInt()
    val op = findOp(opName)!!
    if (valueIdx > op.numOutputs - 1)
      throw IllegalArgumentException("valueIdx > op.numOutputs - 1")
    return Output(op, valueIdx)
  }
  
  /** Returns `true` if [name] is registered in this graph's function library. */
  fun isFunction(name: String) = name in functions
  
  fun toGraphDef() = GraphDef.parseFrom(toGraphDefBytes())
  
  fun toGraphDefBytes(): ByteArray {
    val buf = newBuffer()
    val status = newStatus()
    TF_GraphToGraphDef(c_graph, buf, status)
    status.check()
    val len = buf.length()
    val bytes = ByteArray(len.toInt())
    val d = buf.data()
    d.capacity<Pointer>(len)
    val data = d.asByteBuffer()
    data.get(bytes)
    return bytes
  }
  
  fun debugString(): String {
    return toGraphDef().toString()
  }
  
  fun import(act_graph_def: ByteArray, prefix: String = "") {
    assertNotFrozen()
    val buf = TF_NewBufferFromString(BytePointer(*act_graph_def), act_graph_def.size.toLong())
    val status = newStatus()
    val opt = TF_NewImportGraphDefOptions()
    if (prefix.isNotBlank())
      TF_ImportGraphDefOptionsSetPrefix(opt, prefix)
    TF_GraphImportGraphDef(c_graph, buf, opt, status)
    status.check()
    TF_DeleteImportGraphDefOptions(opt)
    TF_DeleteBuffer(buf)
    
    ops().forEach { namesInUse[it.name] = 1 }
  }
  
  /** Prevents the feeding of values to the provided op output, while running in a session.
   *
   * @param  output Op output whose feeding is prevented.
   * @throws GraphMismatchException If the provided op output does not belong to this graph.
   */
  fun preventFeeding(output: Output) {
    assertNotFrozen()
    if (output.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    unfeedableOutputs += output
  }
  
  /** Prevents the fetching of values to the provided op, while running in a session.
   *
   * @param  op Op whose fetching is prevented.
   * @throws GraphMismatchException If the provided op does not belong to this graph.
   */
  fun preventFetching(op: Op) {
    assertNotFrozen()
    if (op.graph != this)
      throw GraphMismatchException("The provided op does not belong to this graph.")
    unfetchableOps += op
  }
  
  companion object Graph {
    interface Key<K> {
      val name: String
    }
    
    object Keys {
      interface StringCollectionKey : Key<String>
      interface IntCollectionKey : Key<Int>
      interface OpCollectionKey : Key<Op>
      interface OutputCollectionKey : Key<Output>
      interface VariableCollectionKey : Key<Variable>
      interface SaverCollectionKey : Key<Saver>
      interface ResourceCollectionKey : Key<Resource>
      /** Key to collect the graph random seed values. The seed values collection should have only one element
       * representing the graph random seed value. */
      object RANDOM_SEEDS : IntCollectionKey {
        
        override val name: String = "random_seeds"
      }
      
      /** Key to collect the default collection of `Variable` objects, shared across distributed environment (model
       * variables are subset of these). Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`, and
       * all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`. */
      object GLOBAL_VARIABLES : VariableCollectionKey {
        
        override val name: String = "variables"
      }
      
      /** Key to collect the subset of `Variable` objects that are local to each machine. Usually used for temporary
       * variables, like counters. */
      object LOCAL_VARIABLES : VariableCollectionKey {
        
        override val name: String = "local_variables"
      }
      
      /** Key to collect the subset of `Variable` objects that are used in models for inference (feed forward).
       * TODO: Note: use `tf.contrib.framework.modelVariable` to add to this collection. */
      object MODEL_VARIABLES : VariableCollectionKey {
        
        override val name: String = "model_variables"
      }
      
      /** Key to collect the subset of `Variable` objects that will be trained using an optimizer. */
      object TRAINABLE_VARIABLES : VariableCollectionKey {
        
        override val name: String = "trainable_variables"
      }
      
      /** Key to collect the summary `Output` objects that have been created in the graph. */
      object SUMMARIES : OutputCollectionKey {
        
        override val name: String = "summaries"
      }
      
      // /** Key to collect the `QueueRunner` objects that are used to produce inputs for a computation. */
      // object QUEUE_RUNNERS : Key {override val name: String = "queue_runners"}
      
      /** Key to collect the table initializers that have been created in the graph. */
      object TABLE_INITIALIZERS : OpCollectionKey {
        
        override val name: String = "table_initializer"
      }
      
      /** Key to collect asset filepaths. An asset represents an external resource like a vocabulary file. */
      object ASSET_FILEPATHS : OutputCollectionKey {
        
        override val name: String = "asset_filepaths"
      }
      
      /** Key to collect the subset of `Variable` objects that will also keep moving averages. */
      object MOVING_AVERAGE_VARIABLES : VariableCollectionKey {
        
        override val name: String = "moving_average_variables"
      }
      
      /** Key to collect regularization losses at graph construction. */
      object REGULARIZATION_LOSSES : OutputCollectionKey {
        
        override val name: String = "regularization_losses"
      }
      
      // /** Key to collect concatenated sharded variables. */
      // object CONCATENATED_VARIABLES : Key {override val name: String = "concatenated_variables"}
      
      /** Key to collect savers. */
      object SAVERS : SaverCollectionKey {
        
        override val name: String = "savers"
      }
      
      /** Key to collect weights. */
      object WEIGHTS : VariableCollectionKey {
        
        override val name: String = "weights"
      }
      
      /** Key to collect biases. */
      object BIASES : VariableCollectionKey {
        
        override val name: String = "biases"
      }
      
      /** Key to collect activations. */
      object ACTIVATIONS : OpCollectionKey {
        
        override val name: String = "activations"
      }
      
      /** Key to collect update ops. */
      object UPDATE_OPS : OpCollectionKey {
        
        override val name: String = "update_ops"
      }
      
      /** Key to collect losses. */
      object LOSSES : OutputCollectionKey {
        
        override val name: String = "losses"
      }
      
      // /** Key to collect saveable objects used for checkpoints. */
      // object SAVEABLE_OBJECTS : Key {override val name: String = "saveable_objects"}
      
      /** Key to collect all shared resources used by the graph which need to be initialized once per cluster. */
      object SHARED_RESOURCES : ResourceCollectionKey {
        
        override val name: String = "resources"
      }
      
      /** Key to collect all local resources used in this graph which need to be initialized once per session. */
      object LOCAL_RESOURCES : ResourceCollectionKey {
        
        override val name: String = "local_resources"
      }
      
      // Keys to indicate various ops.
      
      object INIT_OP : OpCollectionKey {
        override val name: String = "init_op"
      }
      
      object LOCAL_INIT_OP : OpCollectionKey {
        override val name: String = "local_init_op"
      }
      
      object READY_OP : OutputCollectionKey {
        override val name: String = "ready_op"
      }
      
      object READY_FOR_LOCAL_INIT_OP : OutputCollectionKey {
        override val name: String = "ready_for_local_init_op"
      }
      
      object SUMMARY_OP : OutputCollectionKey {
        override val name: String = "summary_op"
      }
      
      object GLOBAL_EPOCH : VariableCollectionKey {
        override val name: String = "global_epoch"
      }
      
      object GLOBAL_STEP : VariableCollectionKey {
        override val name: String = "global_step"
      }
      
      object EVAL_STEP : VariableCollectionKey {
        override val name: String = "eval_step"
      }
      
      object TRAIN_OP : OpCollectionKey {
        override val name: String = "train_op"
      }
      
      // Keys for control flow management.
      // object COND_CONTEXT : Key {override val name: String = "cond_context"}
      // object WHILE_CONTEXT : Key {override val name: String = "while_context"}
      
      /** Key to collect streaming model ports. */
      object STREAMING_MODEL_PORTS : VariableCollectionKey {
        
        override val name: String = "streaming_model_ports"
      }
      
      /** Key to collect the unbound inputs when serializing/deserializing graphs. */
      object UNBOUND_INPUTS : StringCollectionKey {
        
        override val name: String = "unbound_inputs"
      }
    }
  }
}
