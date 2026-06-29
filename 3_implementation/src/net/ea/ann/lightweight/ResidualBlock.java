package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class ResidualBlock {
    Conv2DSame conv1;
    Conv2DSame conv2;
    ReLULayer relu1;
    ReLULayer relu2;

    double[][][] inputCache;

    public ResidualBlock(int channels) {
        conv1 = new Conv2DSame(channels, 3);
        conv2 = new Conv2DSame(channels, 3);
        relu1 = new ReLULayer();
        relu2 = new ReLULayer();
    }

    public double[][][] forward(double[][][] x) {
        inputCache = x;

        double[][][] out = conv1.forward(x);
        out = relu1.forward(out);
        out = conv2.forward(out);

        // skip connection
        for (int c = 0; c < out.length; c++)
            for (int y = 0; y < out[0].length; y++)
                for (int x2 = 0; x2 < out[0][0].length; x2++)
                    out[c][y][x2] += x[c][y][x2];

        out = relu2.forward(out);

        return out;
    }

    public double[][][] backward(double[][][] grad) {
        grad = relu2.backward(grad);

        double[][][] skipGrad = new double[grad.length][grad[0].length][grad[0][0].length];
        for (int c = 0; c < grad.length; c++)
            for (int y = 0; y < grad[0].length; y++)
                for (int x = 0; x < grad[0][0].length; x++)
                    skipGrad[c][y][x] = grad[c][y][x];

        grad = conv2.backward(grad);
        grad = relu1.backward(grad);
        grad = conv1.backward(grad);

        // split and merge residual gradient
        for (int c = 0; c < grad.length; c++)
            for (int y = 0; y < grad[0].length; y++)
                for (int x = 0; x < grad[0][0].length; x++)
                    grad[c][y][x] += skipGrad[c][y][x];

        return grad;
    }

    public void update(double lr, int batchSize) {
        conv1.update(lr, batchSize);
        conv2.update(lr, batchSize);
    }
}