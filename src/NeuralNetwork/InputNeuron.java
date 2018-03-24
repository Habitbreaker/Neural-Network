package NeuralNetwork;

public class InputNeuron extends Neuron {

    private float value;

    public void setValue(float value) {
        this.value = value;
    }

    @Override
    public float getValue() {
        return this.value;
    }
}
