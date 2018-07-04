package wumo.sim.algorithm.util.cpp_api

import org.junit.Before

open class BaseTest {
  lateinit var tf: TF_CPP
  @Before
  fun setup() {
    tf = TF_CPP()
  }
  
}