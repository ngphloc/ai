package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class FlattenLayer {
    int channels, h, w;

    public double[] forward(double[][][] input) {
        channels = input.length;
        h = input[0].length;
        w = input[0][0].length;

        double[] out = new double[channels * h * w];
        int idx = 0;

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    out[idx++] = input[c][y][x];

        return out;
    }

    public double[][][] backward(double[] grad) {
        double[][][] out = new double[channels][h][w];
        int idx = 0;

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    out[c][y][x] = grad[idx++];

        return out;
    }
}