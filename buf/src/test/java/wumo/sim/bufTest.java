package wumo.sim;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class bufTest {
  
  @Before
  public void startup() {
    buf.buf_init();
  }
  
  @Test
  public void maxOf() {
    long size = 1000;
    BytePointer a = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      a.put(i, (byte) (i % 128));
    }
    BytePointer b = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      b.put(i, (byte) (i + 1 % 128));
    }
    BytePointer c = new BytePointer(size);
    buf.maxOf(a, b, c, size);
    for (long i = 0; i < size; i++) {
      System.out.println(a.get(i) + "," + b.get(i) + "->" + c.get(i));
    }
  }
  
  @Test
  public void copy() {
    long size = 1000;
    BytePointer a = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      a.put(i, (byte) i);
    }
    FloatPointer b = new FloatPointer(size);
    buf.castchar2float(a, b, size);
    for (long i = 0; i < size; i++) {
      System.out.println(a.get(i) + "," + b.get(i));
    }
    for (long i = 0; i < size; i++) {
      b.put(i, i);
    }
    buf.castfloat2char(b, a, size);
    for (long i = 0; i < size; i++) {
      System.out.println(a.get(i) + "," + b.get(i));
    }
  }
}