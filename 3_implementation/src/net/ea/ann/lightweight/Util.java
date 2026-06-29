package net.ea.ann.lightweight;
import java.util.Random;

/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class Util {
    private static final Random rand = new Random();

    public static double randomWeight(int fanIn) {
        // He initialization
        return rand.nextGaussian() * Math.sqrt(2.0 / fanIn);
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    public static double[] softmax(double[] x) {
        double max = x[0];
        for (double v : x) max = Math.max(max, v);

        double sum = 0;
        double[] out = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - max);
            sum += out[i];
        }

        for (int i = 0; i < x.length; i++) out[i] /= sum;

        return out;
    }

    public static double crossEntropy(double[] probs, int label) {
        return -Math.log(probs[label] + 1e-8);
    }
}