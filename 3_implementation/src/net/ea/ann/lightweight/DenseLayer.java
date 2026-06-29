package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class DenseLayer {
    int in, out;
    double[][] W, gradW;
    double[] B, gradB;
    double[] cache;

    public DenseLayer(int in, int out) {
        this.in = in;
        this.out = out;

        W = new double[out][in];
        gradW = new double[out][in];
        B = new double[out];
        gradB = new double[out];

        for (int i = 0; i < out; i++)
            for (int j = 0; j < in; j++)
                W[i][j] = Util.randomWeight(in);
    }

    public double[] forward(double[] x) {
        cache = x;
        double[] y = new double[out];

        for (int i = 0; i < out; i++) {
            y[i] = B[i];
            for (int j = 0; j < in; j++)
                y[i] += W[i][j] * x[j];
        }

        return y;
    }

    public double[] backward(double[] gradOut) {
        double[] gradIn = new double[in];

        for (int i = 0; i < out; i++) {
            for (int j = 0; j < in; j++) {
                gradW[i][j] += gradOut[i] * cache[j];
                gradIn[j] += gradOut[i] * W[i][j];
            }
            gradB[i] += gradOut[i];
        }

        return gradIn;
    }

    public void update(double lr, int batchSize) {
        for (int i = 0; i < out; i++) {
            for (int j = 0; j < in; j++) {
                W[i][j] -= lr * gradW[i][j] / batchSize;
                gradW[i][j] = 0;
            }
            B[i] -= lr * gradB[i] / batchSize;
            gradB[i] = 0;
        }
    }
}