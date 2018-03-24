package ActivationFunctions;

public class Boolean implements ActivationFunction{

    @Override
    public float activation(float input) {
        return input < 0 ? 0 : 1;
    }
}
