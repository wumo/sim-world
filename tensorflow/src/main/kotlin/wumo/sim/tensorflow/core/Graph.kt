@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.tensorflow.core

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.SizeTPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Graph.newGraph
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.OperationBuilder
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.Resource
import wumo.sim.tensorflow.ops.variables.Saver
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.ops.variables.VariableScopeStore
import wumo.sim.tensorflow.ops.variables.VariableStore
import wumo.sim.tensorflow.throwExceptionIfNotOk
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
class Graph {
  
  val c_graph = newGraph()!!
  
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
  
  internal fun cache(op: TF_Operation) = opsCache[op]
  operator fun HashMap<Long, Op>.get(op: TF_Operation) =
      getOrPut(op.address()) { Op(this@Graph, op) }
  
  /** Variable store object of this graph, used to store created variables and keep track of variable scope usages. */
  internal val variableStore = VariableStore()
  /** Variable scope store object of this graph. */
  internal val variableScopeStore = DynamicVariable(VariableScopeStore())
  internal val variableCreatorStack = DynamicVariable(listOf<VariableGetter>())
  
  /** Map from collection key to set of values in that collection. */
  private val collections: MutableMap<Graph.Key<*>, MutableSet<*>> = mutableMapOf()
  /**Set of tensors that are dangerous to feed!*/
  private val unfeedableOutputs = mutableSetOf<Output>()
  /**Set of operations that are dangerous to fetch!*/
  private val unfetchable_ops = mutableSetOf<Op>()
  
  private val functions = LinkedHashSet<String>()
  
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
  
  fun num_node_ids() = c_graph.graph().num_node_ids()
  fun nodeBuilder(opType: String, name: String) = OperationBuilder(this, opType, name)
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
      val op = TF_GraphNextOperation(c_graph, pos)
      ops += opsCache[op]
    } while (op.isNotNull)
    return ops
  }

//  fun getOp(name: String): Op {
//
//    findOp(name)
//  }
  
  fun getTensor(name: String): Output {
    val (opName, idx) = name.split(':')
    val valueIdx = idx.toInt()
    val op = findOp(opName)!!
    if (valueIdx > op.numOutputs - 1)
      throw IllegalArgumentException("valueIdx > op.numOutputs - 1")
    return Output(op, valueIdx)
  }
  
  fun is_function(name: String) = name in functions
  fun toGraphDef(): ByteArray {
    val buf = TF_NewBuffer()
    val status = TF_NewStatus()
    TF_GraphToGraphDef(c_graph, buf, status)
    throwExceptionIfNotOk(status)
    val len = buf.length()
    val bytes = ByteArray(len.toInt())
    val d = buf.data()
    d.capacity<Pointer>(len)
    val data = d.asByteBuffer()
    data.get(bytes)
    TF_DeleteStatus(status)
    TF_DeleteBuffer(buf)
    return bytes
  }
  
  fun import(act_graph_def: ByteArray, prefix: String = "") {
    val buf = TF_NewBufferFromString(BytePointer(*act_graph_def), act_graph_def.size.toLong())
    val status = newStatus()
    val opt = TF_NewImportGraphDefOptions()
    if (prefix.isNotBlank())
      TF_ImportGraphDefOptionsSetPrefix(opt, prefix)
    TF_GraphImportGraphDef(c_graph, buf, opt, status)
    throwExceptionIfNotOk(status)
    TF_DeleteImportGraphDefOptions(opt)
    TF_DeleteBuffer(buf)
  }
  
  fun create_op(new_op_type: String,
                new_op_inputs: MutableList<Output>,
                output_types: List<Int>,
                name: String,
                attrs: Map<String, Any>): Op {
    TODO("not implemented")
  }
  
  fun prevent_fetching(op: Op) {
    unfetchable_ops += op
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
       * TODO: Note: use `tf.contrib.framework.model_variable` to add to this collection. */
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
