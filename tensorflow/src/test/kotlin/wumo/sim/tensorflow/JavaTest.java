package wumo.sim.tensorflow;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.tensorflow;
import org.bytedeco.javacpp.tensorflow.Deallocator_Pointer_long_Pointer;
import org.bytedeco.javacpp.tensorflow.TF_Tensor;
import org.junit.Test;

import static org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.newTensor;
import static org.bytedeco.javacpp.tensorflow.DT_UINT8;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteTensor;
import static org.bytedeco.javacpp.tensorflow.TF_NewTensor;

public class JavaTest {
  @Test
  public void testDeallocate() {
    Object a = tf.INSTANCE;
    while (true) {
      PointerScope scope = new PointerScope();
      BytePointer data = new BytePointer(100L);
      TF_Tensor t = newTensor(DT_UINT8, new long[]{data.limit()}, data);
      scope.close();
    }
  }
  
  
  @Test
  public void testDeallocate2() {
    Object a = tf.INSTANCE;
    Deallocator_Pointer_long_Pointer dummyDeallocator = new Deallocator_Pointer_long_Pointer() {
      @Override
      public void call(Pointer data, long len, Pointer arg) {
      }
    };
    BytePointer data = new BytePointer(100L);
    while (true) {
      PointerScope scope = new PointerScope();
      
      long[] dims = new long[]{data.limit()};
      System.out.println(dummyDeallocator.isNull());
      System.out.println(dummyDeallocator.address());
      TF_Tensor t = TF_NewTensor(DT_UINT8, dims, dims.length, data, data.limit(), dummyDeallocator, null);
      TF_DeleteTensor(t);
      scope.close();
      System.out.println(dummyDeallocator.isNull());
      System.out.println(dummyDeallocator.address());
    }
  }
  
  @Test
  public void testDeallocate3() {
    Object a = tf.INSTANCE;
    BytePointer data = new BytePointer(100L);
    while (true) {
      PointerScope scope = new PointerScope();
      
      long[] dims = new long[]{data.limit()};
      Deallocator_Pointer_long_Pointer dummyDeallocator = new Deallocator_Pointer_long_Pointer() {
        @Override
        public void call(Pointer data, long len, Pointer arg) {
        }
      };
      TF_Tensor t = TF_NewTensor(DT_UINT8, dims, dims.length, data, data.limit(), dummyDeallocator, null);
      TF_DeleteTensor(t);
      scope.close();
      System.out.println(dummyDeallocator.isNull());
      System.out.println(dummyDeallocator.address());
    }
  }
}
