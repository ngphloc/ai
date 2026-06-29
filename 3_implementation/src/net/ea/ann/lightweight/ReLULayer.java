package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class ReLULayer {
    double[][][] cache3D;
    double[] cache1D;

    public double[][][] forward(double[][][] input) {
        cache3D = input;
        double[][][] out = new double[input.length][input[0].length][input[0][0].length];

        for (int c = 0; c < input.length; c++)
            for (int i = 0; i < input[0].length; i++)
                for (int j = 0; j < input[0][0].length; j++)
                    out[c][i][j] = Util.relu(input[c][i][j]);

        return out;
    }

    public double[][][] backward(double[][][] grad) {
        for (int c = 0; c < grad.length; c++)
            for (int i = 0; i < grad[0].length; i++)
                for (int j = 0; j < grad[0][0].length; j++)
                    grad[c][i][j] *= Util.reluDerivative(cache3D[c][i][j]);

        return grad;
    }

    public double[] forward(double[] input) {
        cache1D = input;
        double[] out = new double[input.length];
        for (int i = 0; i < input.length; i++)
            out[i] = Util.relu(input[i]);
        return out;
    }

    public double[] backward(double[] grad) {
        for (int i = 0; i < grad.length; i++)
            grad[i] *= Util.reluDerivative(cache1D[i]);
        return grad;
    }
}