/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.Scanner;

import net.ea.ann.core.Network;
import net.ea.ann.core.function.Function;
import net.ea.ann.mane.MatrixNetworkAbstract;

/**
 * This class provides utility methods to create classifier model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public final class ClassifierBuilder implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This enum represents classifier model.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static enum ClassifierModel {
		
		/**
		 * Matrix neural network classifier.
		 */
		mac,
		
		/**
		 * Stack network classifier.
		 */
		stack,
		
	}
	
	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Classifier model.
	 */
	ClassifierModel model = ClassifierModel.mac;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Learning rate.
	 */
	protected double learningRate = Network.LEARN_RATE_DEFAULT;
	
	
	/**
	 * Number of batches.
	 */
	protected int batches = Network.LEARN_MAX_ITERATION_DEFAULT;
	
	
	/**
	 * Including convolutional neural network.
	 */
	protected boolean conv = MatrixClassifier0.CONV_DEFAULT;
	
	
	/**
	 * Vectorization mode.
	 */
	protected boolean vectorized = MatrixNetworkAbstract.VECTORIZED_DEFAULT;
	
	
	/**
	 * Adjustment mode.
	 */
	protected boolean adjust = MatrixClassifier.ADJUST_DEFAULT;
	
	
	/**
	 * Dual mode.
	 */
	protected boolean dual = MatrixClassifier0.DUAL_DEFAULT;
	
	
	/**
	 * Baseline mode.
	 */
	protected boolean baseline = MatrixClassifier0.BASELINE_DEFAULT;

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ClassifierBuilder(int neuronChannel) {
		this.neuronChannel = neuronChannel;
	}

	
	/**
	 * Setting activation reference.
	 * @param activateRef activation reference.
	 * @return this builder.
	 */
	public ClassifierBuilder setActivateRef(Function activateRef) {
		this.activateRef = activateRef;
		return this;
	}
	
	
	/**
	 * Getting classifier model.
	 * @return classifier model.
	 */
	public ClassifierModel getModel() {
		return model;
	}
	
	
	/**
	 * Setting classifier model.
	 * @param model classifier model.
	 * @return this builder.
	 */
	public ClassifierBuilder setModel(ClassifierModel model) {
		this.model = model;
		return this;
	}
	
	
	/**
	 * Getting learning rate.
	 * @return learning rate.
	 */
	public double getLearningRate() {
		return learningRate;
	}
	
	
	/**
	 * Setting learning rate.
	 * @param learningRate learning rate.
	 * @return this builder.
	 */
	public ClassifierBuilder setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}
	
	
	/**
	 * Getting batches.
	 * @return batches.
	 */
	public int getBatches() {
		return batches;
	}
	
	
	/**
	 * Setting batches.
	 * @param batches batches.
	 * @return this builder.
	 */
	public ClassifierBuilder setBatches(int batches) {
		this.batches = batches;
		return this;
	}
	
	
	/**
	 * Getting convolutional mode.
	 * @return convolutional mode.
	 */
	public boolean isConv() {
		return conv;
	}
	
	
	/**
	 * Setting flag to include convolutional neural network.
	 * @param conv flag to include convolutional neural network.
	 * @return this builder.
	 */
	public ClassifierBuilder setConv(boolean conv) {
		this.conv = conv;
		return this;
	}
	
	
	/**
	 * Getting vectorization mode.
	 * @return vectorization mode.
	 */
	public boolean isVectorized() {
		return vectorized;
	}
	
	
	/**
	 * Setting vectorization mode.
	 * @param vectorized vectorization mode.
	 * @return this builder.
	 */
	public ClassifierBuilder setVectorized(boolean vectorized) {
		this.vectorized = vectorized;
		return this;
	}
	
	
	/**
	 * Getting adjustment mode.
	 * @return adjustment mode.
	 */
	public boolean isAdjust() {
		return adjust;
	}
	
	
	/**
	 * Setting adjustment mode.
	 * @param adjust adjustment mode.
	 * @return this builder.
	 */
	public ClassifierBuilder setAdjust(boolean adjust) {
		this.adjust = adjust;
		return this;
	}
	
	
	/**
	 * Getting dual mode.
	 * @return dual mode.
	 */
	public boolean isDual() {
		return dual;
	}
	
	
	/**
	 * Setting baseline mode.
	 * @param baseline baseline mode.
	 * @return this builder.
	 */
	public ClassifierBuilder setBaseline(boolean baseline) {
		this.baseline = baseline;
		return this;
	}
	
	
	/**
	 * Getting dual mode.
	 * @return dual mode.
	 */
	public boolean isBaseline() {
		return baseline;
	}
	
	
	/**
	 * Setting dual mode.
	 * @param dual dual mode.
	 * @return this builder.
	 */
	public ClassifierBuilder setDual(boolean dual) {
		this.dual = dual;
		return this;
	}
	
	
	/**
	 * Build classifier.
	 * @return classifier.
	 */
	public Classifier build() {
		Classifier classifier = null;
		switch (model) {
		case mac:
			classifier = MatrixClassifier.create(neuronChannel, true); 
			break;
		case stack:
			classifier = StackClassifier.create(neuronChannel, true);
			break;
		default:
			break;
		}
		
		if (classifier instanceof MatrixClassifier) {
			MatrixClassifier mac = (MatrixClassifier)classifier;
			mac.paramSetLearningRate(learningRate);
			mac.paramSetBatches(batches);
			mac.paramSetConv(conv);
			mac.setVectorized(vectorized);
			mac.paramSetAdjust(adjust);
			mac.paramSetDual(dual);
			mac.paramSetBaseline(baseline);
		}
		else if (classifier instanceof StackClassifier) {
			
		}
		
		return classifier;
	}
	
	
	/**
	 * Creating builder by user entering.
	 * @param in input stream.
	 * @param out output stream.
	 * @return builder.
	 */
	public static ClassifierBuilder enter(InputStream in, OutputStream out) {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);
		
		int defaultModel = 0;
		int model = defaultModel;
		printer.print("Model (0-mac, 1-stack) (default " + defaultModel + " is mac):");
		try {
			model = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(model)) model = defaultModel;
		if (model <= 0) model = defaultModel;
		printer.println("Model is " + model + "\n");
		
		int defaultRasterChannel = 3;
		int rasterChannel = defaultRasterChannel;
		printer.print("Raster channel (1, 2, 3, 4) (default " + defaultRasterChannel + "):");
		try {
			rasterChannel = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(rasterChannel)) rasterChannel = defaultRasterChannel;
		if (rasterChannel < defaultRasterChannel) rasterChannel = defaultRasterChannel;
		printer.println("Raster channel is " + rasterChannel + "\n");
	
		double defaultlr = Network.LEARN_RATE_DEFAULT;
		double lr = defaultlr;
		printer.print("Enter starting learning rate (default " + defaultlr + "):");
		try {
			lr = Double.parseDouble(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(lr)) lr = defaultlr;
		if (lr <= 0 || lr > 1) lr = defaultlr;
		printer.println("Starting learning rate is " + lr + "\n");
	
		int defaultBatches = 100;
		int batches = defaultBatches;
		printer.print("Batches (default " + batches + "):");
		try {
			batches = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(batches)) batches = defaultBatches;
		if (batches <= 0) batches = defaultBatches;
		printer.println("Batches are " + batches + "\n");
	
		boolean conv = false;
		printer.print("Including convolutional network (" + conv + " is default):");
		try {
			conv = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println("Including convolutional network is " + conv + "\n");
	
		boolean vectorized = false;
		printer.print("Vectorization (" + vectorized + " is default):");
		try {
			vectorized = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println("Vectorization is " + vectorized + "\n");
	
		boolean adjust = true;
		printer.print("Adjustment (" + adjust + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (line.isBlank() || line.isEmpty())
				adjust = true;
			else
				adjust = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Adjustment is " + adjust + "\n");
	
		boolean dual = false;
		printer.print("Dual mode (" + dual + " is default):");
		try {
			dual = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println("Dual mode is " + dual + "\n");
		
		ClassifierBuilder builder = new ClassifierBuilder(rasterChannel);
		switch (model) {
		case 0:
			builder.setModel(ClassifierModel.mac);
			break;
		case 1:
			builder.setModel(ClassifierModel.stack);
			break;
		default:
			builder.setModel(ClassifierModel.mac);
			break;
		}
		builder.setLearningRate(lr);
		builder.setBatches(batches);
		builder.setConv(conv);
		builder.setVectorized(vectorized);
		builder.setAdjust(adjust);
		builder.setDual(dual);
		return builder;
	}
	
	
}
