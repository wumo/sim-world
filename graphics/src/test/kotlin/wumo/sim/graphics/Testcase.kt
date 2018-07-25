package wumo.sim.graphics

import com.badlogic.gdx.graphics.Color
import org.junit.Test

class Testcase {
  @Test
  fun `test1`() {
    val viewer = Viewer(Config(width = 800, height = 800, isContinousRendering = false))
    val car = Geom {
      color(Color.RED)
      rect_filled(0f, 0f, 100f, 0f, 100f, 100f, 0f, 100f)
    }
    viewer.add(car)
    viewer.startAsync()
    Thread.sleep(1000 / 60)
    viewer.requestRender()
    Thread.sleep(2000)
    viewer.close()
  }
  
  @Test
  fun `test2`() {
    val viewer = Viewer(Config(width = 800, height = 800, isContinousRendering = false))
    val car = Geom {
      color(Color.RED)
      circle(100f, 100f, 100f)
      
    }.apply { z = -99f }
    viewer.add(car)
    viewer.start()
  }
  
  @Test
  fun test3() {
    val viewer = Viewer(Config(width = 800, height = 800, isContinousRendering = false, maxVertices = 15))
    val car = Geom {
      color(Color.RED)
      rect_filled(0f, 0f, 100f, 0f, 100f, 100f, 0f, 100f)
    }
    val car2 = Geom {
      color(Color.BLUE)
      rect_filled(100f, 0f, 200f, 0f, 200f, 100f, 100f, 100f)
    }
    viewer.add(car)
    viewer += car2
    viewer.startAsync()
    Thread.sleep(2000)
    viewer.requestRender()
    car.shape = {
      color(Color.GREEN)
      rect_filled(0f, 0f, 100f, 0f, 100f, 100f, 0f, 100f)
      triangle_filled(0f, 100f, 100f, 100f, 50f, 200f)
    }
    viewer.requestRender()
    Thread.sleep(2000)
    viewer.remove(car)
    viewer.requestRender()
    Thread.sleep(2000)
    viewer.add(car)
    viewer.requestRender()
    Thread.sleep(2000)
    viewer.requestRender()
    Thread.sleep(2000)
    viewer.close()
  }
  
  @Test
  fun test4() {
    val viewer = Viewer(Config(width = 800, height = 800, isContinousRendering = false, maxVertices = 15))
    val car = Geom {
      color(Color.RED)
      rect_filled(0f, 0f, 100f, 0f, 100f, 100f, 0f, 100f)
    }
    val car2 = Geom {
      color(Color.BLUE)
      rect_filled(100f, 0f, 200f, 0f, 200f, 100f, 100f, 100f)
    }
    viewer.add(car)
    viewer += car2
    viewer.startAsync()
    Thread.sleep(2000)
    viewer.requestRender()
    car2.shape = {
      color(Color.BLUE)
      rect_filled(100f, 0f, 200f, 0f, 200f, 100f, 100f, 100f)
      triangle_filled(100f, 100f, 200f, 100f, 150f, 200f)
    }
    viewer.requestRender()
    Thread.sleep(2000)
    viewer.close()
  }
  
}
