package wumo.sim.helper;

import org.bytedeco.javacpp.Pointer;

public class ale extends wumo.sim.presets.ale {
    public static class ALEInterface extends Pointer {
        public ALEInterface(Pointer p) {
            super(p);
        }
    }

    public static class ALEState extends Pointer {
        public ALEState(Pointer p) {
            super(p);
        }
    }
}
