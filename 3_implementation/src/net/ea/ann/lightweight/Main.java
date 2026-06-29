package net.ea.ann.lightweight;
/**
 * 
 * @author ChatGPT
 * @version 1.0
 *
 */
public class Main {
    public static void main(String[] args) {
        ResNetCNN model = new ResNetCNN();

        int epochs = 5;
        int batchSize = 32;
        double lr = 0.001;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int correct = 0;
            int total = 320;

            for (int i = 0; i < total; i++) {
                double[][] image = new double[28][28];

                for (int y = 0; y < 28; y++)
                    for (int x = 0; x < 28; x++)
                        image[y][x] = Math.random();

                int label = (int)(Math.random() * 10);

                totalLoss += model.train(image, label);

                int prediction = model.predict(image);

                if (prediction == label) {
                    correct++;
                }

                System.out.println(
                    "Sample " + i +
                    " | Predicted: " + prediction +
                    " | Actual: " + label
                );

                if ((i + 1) % batchSize == 0) {
                    model.update(lr, batchSize);
                }
            }

            double accuracy = (double) correct / total * 100.0;

            System.out.println(
                "Epoch " + epoch +
                " | Loss = " + (totalLoss / total) +
                " | Accuracy = " + accuracy + "%"
            );
        }
    }
}