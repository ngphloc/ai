package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class MaxPoolLayer {
    int size;
    int[][][] maxY;
    int[][][] maxX;
    int channels;

    public MaxPoolLayer(int size) {
        this.size = size;
    }

    public double[][][] forward(double[][][] input) {
        channels = input.length;
        int h = input[0].length / size;
        int w = input[0][0].length / size;

        maxY = new int[channels][h][w];
        maxX = new int[channels][h][w];

        double[][][] out = new double[channels][h][w];

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    double max = -Double.MAX_VALUE;

                    for (int py = 0; py < size; py++) {
                        for (int px = 0; px < size; px++) {
                            int iy = y * size + py;
                            int ix = x * size + px;

                            if (input[c][iy][ix] > max) {
                                max = input[c][iy][ix];
                                maxY[c][y][x] = iy;
                                maxX[c][y][x] = ix;
                            }
                        }
                    }

                    out[c][y][x] = max;
                }
            }
        }

        return out;
    }

    public double[][][] backward(double[][][] gradOut, int inH, int inW) {
        double[][][] gradIn = new double[channels][inH][inW];

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < gradOut[0].length; y++)
                for (int x = 0; x < gradOut[0][0].length; x++)
                    gradIn[c][maxY[c][y][x]][maxX[c][y][x]] += gradOut[c][y][x];

        return gradIn;
    }
}