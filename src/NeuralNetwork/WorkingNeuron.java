package NeuralNetwork;

import ActivationFunctions.ActivationFunction;
import ActivationFunctions.Boolean;
import ActivationFunctions.Identity;

import java.util.ArrayList;
import java.util.List;

public class WorkingNeuron extends  Neuron{
    private List<Connection> connections = new ArrayList<>();
    private ActivationFunction activationFunction = ActivationFunction.ActivationIdentity;

    @Override
    public float getValue() {
        float sum = 0;
        for (Connection c : connections) {
            sum += c.getValue();
        }
        return activationFunction.activation(sum);
    }

    public void addConnection(Connection c) {
        connections.add(c);
    }

    public void deltaLearning(float epsilon, float smallDelta) {
        for(int i = 0; i < connections.size(); i++) {
            float bigDelta = epsilon * smallDelta * connections.get(i).getValue();
            connections.get(i).addWeight(bigDelta);
        }
    }
}
