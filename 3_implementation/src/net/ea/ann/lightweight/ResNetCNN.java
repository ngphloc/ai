package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class ResNetCNN {
    ResidualBlock r1;
    ResidualBlock r2;
    MaxPoolLayer pool;
    FlattenLayer flatten;
    DenseLayer d1;
    ReLULayer relu;
    DropoutLayer dropout;
    DenseLayer d2;

    public ResNetCNN() {
        r1 = new ResidualBlock(1);
        r2 = new ResidualBlock(1);

        pool = new MaxPoolLayer(2);
        flatten = new FlattenLayer();

        d1 = new DenseLayer(14 * 14, 64);
        relu = new ReLULayer();
        dropout = new DropoutLayer(0.5);
        d2 = new DenseLayer(64, 10);
    }

    public double train(double[][] image, int label) {
        double[][][] x = new double[1][28][28];
        x[0] = image;

        x = r1.forward(x);
        x = r2.forward(x);

        x = pool.forward(x);

        double[] flat = flatten.forward(x);

        double[] h = d1.forward(flat);
        h = relu.forward(h);
        h = dropout.forward(h, true);

        double[] logits = d2.forward(h);
        double[] probs = Util.softmax(logits);

        double loss = Util.crossEntropy(probs, label);

        double[] grad = probs.clone();
        grad[label] -= 1.0;

        grad = d2.backward(grad);
        grad = dropout.backward(grad);
        grad = relu.backward(grad);
        grad = d1.backward(grad);

        double[][][] grad3 = flatten.backward(grad);
        grad3 = pool.backward(grad3, 28, 28);
        grad3 = r2.backward(grad3);
        grad3 = r1.backward(grad3);

        return loss;
    }

    public void update(double lr, int batchSize) {
        r1.update(lr, batchSize);
        r2.update(lr, batchSize);
        d1.update(lr, batchSize);
        d2.update(lr, batchSize);
    }
    
    public int predict(double[][] image) {
        double[][][] x = new double[1][28][28];
        x[0] = image;

        x = r1.forward(x);
        x = r2.forward(x);

        x = pool.forward(x);

        double[] flat = flatten.forward(x);

        double[] h = d1.forward(flat);
        h = relu.forward(h);

        // inference mode: dropout disabled
        h = dropout.forward(h, false);

        double[] logits = d2.forward(h);
        double[] probs = Util.softmax(logits);

        int best = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[best]) {
                best = i;
            }
        }

        return best;
    }

}