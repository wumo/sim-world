package wumo.sim.graphics

import com.badlogic.gdx.math.Vector2
import wumo.sim.graphics.ShapeType.Triangles

open class Geom(val type: ShapeType = Triangles,
                shape: MeshPartBuilder.() -> Unit) {
  internal var changed = true
  var shape = shape
    set(value) {
      field = value
      changed = true
    }
  val translation = Vector2()
  var z = 0f
  var rotation = 0f
  var scale = Vector2(1f, 1f)
  internal var attr: Geom.() -> Unit = {}
  fun attr(attr: Geom.() -> Unit): Geom {
    this.attr = attr
    return this
  }
  
  internal val mesh = InternalMeshPart()
  
  internal fun prepareMesh(builder: MeshPartBuilder) {
    if (changed) {
      builder.begin(mesh, type)
      shape(builder)
      builder.end()
      changed = false
    }
  }
}
