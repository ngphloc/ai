package net.ea.ann.lightweight;

/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class BatchNorm2D {
    int channels;

    double[] gamma;
    double[] beta;

    double[] gradGamma;
    double[] gradBeta;

    double[] runningMean;
    double[] runningVar;

    double[] batchMean;
    double[] batchVar;

    double[][][] xHatCache;

    double momentum = 0.9;
    double eps = 1e-5;

    public BatchNorm2D(int channels) {
        this.channels = channels;

        gamma = new double[channels];
        beta = new double[channels];

        gradGamma = new double[channels];
        gradBeta = new double[channels];

        runningMean = new double[channels];
        runningVar = new double[channels];

        for (int c = 0; c < channels; c++) {
            gamma[c] = 1.0;
            beta[c] = 0.0;
            runningVar[c] = 1.0;
        }
    }

    public double[][][] forward(double[][][] input, boolean training) {
        int h = input[0].length;
        int w = input[0][0].length;

        double[][][] out = new double[channels][h][w];
        xHatCache = new double[channels][h][w];

        batchMean = new double[channels];
        batchVar = new double[channels];

        int N = h * w;

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    batchMean[c] += input[c][y][x];

            batchMean[c] /= N;

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    double diff = input[c][y][x] - batchMean[c];
                    batchVar[c] += diff * diff;
                }

            batchVar[c] /= N;

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    xHatCache[c][y][x] =
                        (input[c][y][x] - batchMean[c]) /
                        Math.sqrt(batchVar[c] + eps);

                    out[c][y][x] =
                        gamma[c] * xHatCache[c][y][x] + beta[c];
                }
            }

            runningMean[c] =
                momentum * runningMean[c] +
                (1 - momentum) * batchMean[c];

            runningVar[c] =
                momentum * runningVar[c] +
                (1 - momentum) * batchVar[c];
        }

        return out;
    }

    public double[][][] backward(double[][][] gradOut) {
        int h = gradOut[0].length;
        int w = gradOut[0][0].length;

        int N = h * w;

        double[][][] gradIn = new double[channels][h][w];

        for (int c = 0; c < channels; c++) {
            double sumDxhat = 0;
            double sumDxhatXhat = 0;

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    gradGamma[c] += gradOut[c][y][x] * xHatCache[c][y][x];
                    gradBeta[c] += gradOut[c][y][x];

                    double dxhat = gradOut[c][y][x] * gamma[c];

                    sumDxhat += dxhat;
                    sumDxhatXhat += dxhat * xHatCache[c][y][x];
                }
            }

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    double dxhat = gradOut[c][y][x] * gamma[c];

                    gradIn[c][y][x] =
                        (1.0 / N) *
                        (1.0 / Math.sqrt(batchVar[c] + eps)) *
                        (N * dxhat - sumDxhat
                         - xHatCache[c][y][x] * sumDxhatXhat);
                }
            }
        }

        return gradIn;
    }

    public void update(double lr, int batchSize) {
        for (int c = 0; c < channels; c++) {
            gamma[c] -= lr * gradGamma[c] / batchSize;
            beta[c] -= lr * gradBeta[c] / batchSize;

            gradGamma[c] = 0;
            gradBeta[c] = 0;
        }
    }
}