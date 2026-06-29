package net.ea.ann.lightweight;
import java.util.Random;

/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class DropoutLayer {
    double rate;
    boolean[] mask;
    Random rand = new Random();

    public DropoutLayer(double rate) {
        this.rate = rate;
    }

    public double[] forward(double[] x, boolean training) {
        double[] out = new double[x.length];

        if (training) {
            mask = new boolean[x.length];
            for (int i = 0; i < x.length; i++) {
                mask[i] = rand.nextDouble() > rate;
                out[i] = mask[i] ? x[i] : 0;
            }
        } else {
            for (int i = 0; i < x.length; i++)
                out[i] = x[i] * (1 - rate);
        }

        return out;
    }

    public double[] backward(double[] grad) {
        for (int i = 0; i < grad.length; i++)
            grad[i] = mask[i] ? grad[i] : 0;
        return grad;
    }
}