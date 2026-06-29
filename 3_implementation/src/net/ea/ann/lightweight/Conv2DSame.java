package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class Conv2DSame {
    int channels;
    int kernel;
    int pad;

    double[][][][] W;
    double[][][][] gradW;

    double[][][] inputCache;

    public Conv2DSame(int channels, int kernel) {
        this.channels = channels;
        this.kernel = kernel;
        this.pad = kernel / 2;

        W = new double[channels][channels][kernel][kernel];
        gradW = new double[channels][channels][kernel][kernel];

        for (int o = 0; o < channels; o++)
            for (int i = 0; i < channels; i++)
                for (int r = 0; r < kernel; r++)
                    for (int c = 0; c < kernel; c++)
                        W[o][i][r][c] = Util.randomWeight(channels * kernel * kernel);
    }

    public double[][][] forward(double[][][] input) {
        inputCache = input;

        int h = input[0].length;
        int w = input[0][0].length;

        double[][][] out = new double[channels][h][w];

        for (int o = 0; o < channels; o++) {
            for (int i = 0; i < channels; i++) {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        for (int ky = 0; ky < kernel; ky++) {
                            for (int kx = 0; kx < kernel; kx++) {
                                int iy = y + ky - pad;
                                int ix = x + kx - pad;

                                if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                    out[o][y][x] += input[i][iy][ix] * W[o][i][ky][kx];
                                }
                            }
                        }
                    }
                }
            }
        }

        return out;
    }

    public double[][][] backward(double[][][] gradOut) {
        int h = inputCache[0].length;
        int w = inputCache[0][0].length;

        double[][][] gradIn = new double[channels][h][w];

        for (int o = 0; o < channels; o++) {
            for (int i = 0; i < channels; i++) {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        for (int ky = 0; ky < kernel; ky++) {
                            for (int kx = 0; kx < kernel; kx++) {
                                int iy = y + ky - pad;
                                int ix = x + kx - pad;

                                if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                    gradW[o][i][ky][kx] += gradOut[o][y][x] * inputCache[i][iy][ix];
                                    gradIn[i][iy][ix] += gradOut[o][y][x] * W[o][i][ky][kx];
                                }
                            }
                        }
                    }
                }
            }
        }

        return gradIn;
    }

    public void update(double lr, int batchSize) {
        for (int o = 0; o < channels; o++)
            for (int i = 0; i < channels; i++)
                for (int ky = 0; ky < kernel; ky++)
                    for (int kx = 0; kx < kernel; kx++) {
                        W[o][i][ky][kx] -= lr * gradW[o][i][ky][kx] / batchSize;
                        gradW[o][i][ky][kx] = 0;
                    }
    }
}