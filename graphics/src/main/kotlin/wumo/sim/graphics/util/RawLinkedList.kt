package wumo.sim.graphics.util


/**
 * Constructs an empty list.
 */
internal class RawLinkedList<E> : Iterable<E?> {
  var size = 0
  
  /**
   * Pointer to first node.
   * Invariant: (first == null && last == null) ||
   * (first.prev == null && first.item != null)
   */
  var first: Node<E>? = null
  
  /**
   * Pointer to last node.
   * Invariant: (first == null && last == null) ||
   * (last.next == null && last.item != null)
   */
  var last: Node<E>? = null
  
  /**
   * Links e as last element.
   */
  fun linkLast(e: E): Node<E> {
    val l = last
    val newNode = Node(l, e, null)
    last = newNode
    if (l == null)
      first = newNode
    else
      l.next = newNode
    size++
    return newNode
  }
  
  /**
   * Unlinks non-null node x.
   */
  fun unlink(x: Node<E>): E? {
    val element = x.item
    val next = x.next
    val prev = x.prev
    
    if (prev == null) {
      first = next
    } else {
      prev.next = next
      x.prev = null
    }
    
    if (next == null) {
      last = prev
    } else {
      next.prev = prev
      x.next = null
    }
    
    x.item = null
    size--
    return element
  }
  
  override fun iterator() = object : Iterator<E?> {
    internal var x = first
    
    override fun hasNext(): Boolean {
      return x != null
    }
    
    override fun next(): E? {
      val tmp = x!!.item
      x = x!!.next
      return tmp
    }
  }
  
  class Node<E>(var prev: Node<E>?, var item: E?, var next: Node<E>?)
  
}
