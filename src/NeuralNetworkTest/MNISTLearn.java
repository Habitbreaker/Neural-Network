package NeuralNetworkTest;

import NeuralNetwork.InputNeuron;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.WorkingNeuron;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MNISTLearn {
    public static List<MNISTDecoder.Digit> digits;
    public static List<MNISTDecoder.Digit> digitsTest;
    public static  NeuralNetwork nn = new NeuralNetwork();
    public static InputNeuron[][] inputs = new InputNeuron[28][28];
    public static WorkingNeuron[] outputs = new WorkingNeuron[10];

    public static void main(String[] args) throws IOException {
        digits = MNISTDecoder.loadDataSet("C:\\dev_src\\MNIST Datenbank\\data\\train-images.idx3-ubyte", "C:\\dev_src\\MNIST Datenbank\\data\\train-labels.idx1-ubyte");
        digitsTest = MNISTDecoder.loadDataSet("C:\\dev_src\\MNIST Datenbank\\data\\t10k-images.idx3-ubyte", "C:\\dev_src\\MNIST Datenbank\\data\\t10k-labels.idx1-ubyte");

        for(int i = 0; i < 28; i++) {
            for(int k = 0; k < 28; k++) {
                inputs[i][k] = nn.createNewInput();
            }
        }

        for(int i = 0; i < 10; i++) {
            outputs[i] = nn.createNewOutput();
        }

        Random rand = new Random();
        float[] weights = new float[28 * 28 * 10];
        for(int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextFloat();
        }
        nn.createFullMesh(weights);

        float epsilon = 0.01f;
        while(true) {
            test();
            for(int i = 0; i < digits.size(); i++) {
                for(int x = 0; x < 28; x++) {
                   for(int y = 0; y < 28; y++) {
                       inputs[x][y].setValue(MNISTDecoder.toUnsignedByte(digits.get(i).data[x][y]) / 255f);
                   }
                }
                float[] shoulds = new float[10];
                shoulds[digits.get(i).label] = 1;
                nn.deltaLearning(shoulds, epsilon);
            }
            epsilon *= 0.9f;
        }
    }

    public static void test() {
        int correct = 0;
        int incorrect = 0;

        for(int i = 0; i < digitsTest.size(); i++) {
            for(int x = 0; x < 28; x++) {
                for(int y = 0; y < 28; y++) {
                    inputs[x][y].setValue(MNISTDecoder.toUnsignedByte(digitsTest.get(i).data[x][y]) / 255f);
                }
            }
            ProbabilityDigit[] probs = new ProbabilityDigit[10];
            for(int k = 0; k < probs.length; k++) {
                probs[k] = new ProbabilityDigit(k, outputs[k].getValue());
            }

            Arrays.sort(probs, Collections.reverseOrder());

            boolean wasCorrect = false;

            if(digitsTest.get(i).label == probs[0].DIGIT) {
                correct++;
            } else {
                incorrect++;
            }
        }

        float percentage = (float) correct / (float)(correct + incorrect);
        System.out.println(percentage);
    }

    public static class ProbabilityDigit implements Comparable<ProbabilityDigit> {
        public final int DIGIT;
        public float probability;

        public ProbabilityDigit(int digit, float probability) {
            this.DIGIT = digit;
            this. probability = probability;
        }

        @Override
        public int compareTo(ProbabilityDigit other) {
            if(probability == other.probability) {
                return 0;
            } else if(probability > other.probability) {
                return 1;
            } else {
                return -1;
            }
        }
    }
}
