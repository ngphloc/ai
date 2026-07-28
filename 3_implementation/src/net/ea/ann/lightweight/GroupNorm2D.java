package net.ea.ann.lightweight;

/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class GroupNorm2D {

    private final int channels;
    private final int groups;
    private final int channelsPerGroup;

    private final double[] gamma;
    private final double[] beta;

    private final double[] gradGamma;
    private final double[] gradBeta;

    private double[][][] xHatCache;
    private double[] groupMean;
    private double[] groupVar;

    private static final double EPS = 1e-5;

    public GroupNorm2D(int channels, int groups) {

        if (channels % groups != 0)
            throw new IllegalArgumentException(
                "channels must be divisible by groups");

        this.channels = channels;
        this.groups = groups;
        this.channelsPerGroup = channels / groups;

        gamma = new double[channels];
        beta = new double[channels];

        gradGamma = new double[channels];
        gradBeta = new double[channels];

        for (int c = 0; c < channels; c++) {
            gamma[c] = 1.0;
            beta[c] = 0.0;
        }
    }

    public double[][][] forward(double[][][] input) {

        int H = input[0].length;
        int W = input[0][0].length;

        double[][][] output =
            new double[channels][H][W];

        xHatCache =
            new double[channels][H][W];

        groupMean = new double[groups];
        groupVar = new double[groups];

        for (int g = 0; g < groups; g++) {

            int cStart = g * channelsPerGroup;
            int cEnd = cStart + channelsPerGroup;

            int N = channelsPerGroup * H * W;

            //----------------------------------------
            // mean
            //----------------------------------------

            double mean = 0;

            for (int c = cStart; c < cEnd; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        mean += input[c][y][x];

            mean /= N;

            groupMean[g] = mean;

            //----------------------------------------
            // variance
            //----------------------------------------

            double var = 0;

            for (int c = cStart; c < cEnd; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++) {

                        double d =
                            input[c][y][x] - mean;

                        var += d * d;
                    }

            var /= N;

            groupVar[g] = var;

            //----------------------------------------
            // normalize
            //----------------------------------------

            double std =
                Math.sqrt(var + EPS);

            for (int c = cStart; c < cEnd; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++) {

                        double xHat =
                            (input[c][y][x]-mean)/std;

                        xHatCache[c][y][x] = xHat;

                        output[c][y][x] =
                            gamma[c]*xHat
                            + beta[c];
                    }
        }

        return output;
    }

    public double[][][] backward(double[][][] gradOut) {

        int H = gradOut[0].length;
        int W = gradOut[0][0].length;

        double[][][] gradIn =
            new double[channels][H][W];

        for (int g = 0; g < groups; g++) {

            int cStart = g * channelsPerGroup;
            int cEnd = cStart + channelsPerGroup;

            int N = channelsPerGroup * H * W;

            double invStd =
                1.0 / Math.sqrt(groupVar[g] + EPS);

            double sumDxHat = 0;
            double sumDxHatXHat = 0;

            //--------------------------------------
            // gamma beta gradient
            //--------------------------------------

            for (int c = cStart; c < cEnd; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++) {

                        gradGamma[c] +=
                            gradOut[c][y][x]
                            * xHatCache[c][y][x];

                        gradBeta[c] +=
                            gradOut[c][y][x];

                        double dxHat =
                            gradOut[c][y][x]
                            * gamma[c];

                        sumDxHat += dxHat;

                        sumDxHatXHat +=
                            dxHat
                            * xHatCache[c][y][x];
                    }

            //--------------------------------------
            // input gradient
            //--------------------------------------

            for (int c = cStart; c < cEnd; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++) {

                        double dxHat =
                            gradOut[c][y][x]
                            * gamma[c];

                        gradIn[c][y][x] =
                            invStd
                            *
                            (
                                N*dxHat
                                - sumDxHat
                                - xHatCache[c][y][x]
                                * sumDxHatXHat
                            )
                            / N;
                    }
        }

        return gradIn;
    }

    public void update(double lr) {

        for (int c = 0; c < channels; c++) {

            gamma[c] -= lr * gradGamma[c];
            beta[c] -= lr * gradBeta[c];

            gradGamma[c] = 0;
            gradBeta[c] = 0;
        }
    }
}