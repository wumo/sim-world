package wumo.sim.graphics

import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture

class Image(val scale:Float,var img: () -> Pixmap) {
  private var changed = true
  internal lateinit var tex: Texture
  
  fun change(img: () -> Pixmap) {
    changed = true
    this.img = img
  }
  
  fun prepare() {
    if (changed) {
      if (::tex.isInitialized)
        tex.dispose()
      val pixmap = img()
      tex = Texture(pixmap)
      pixmap.dispose()
    }
  }
  
  fun dispose() {
    if (::tex.isInitialized)
      tex.dispose()
  }
}