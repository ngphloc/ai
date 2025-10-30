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
import net.ea.ann.transformer.TransformerBasic;

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
		 * Transformer classifier.
		 */
		tramac,

		
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
	protected boolean conv = ClassifierAbstract.CONV_DEFAULT;
	
	
	/**
	 * Vectorization mode.
	 */
	protected boolean vectorized = MatrixNetworkAbstract.VECTORIZED_DEFAULT;
	
	
	/**
	 * Adjustment mode.
	 */
	protected boolean adjust = ClassifierAbstract.ADJUST_DEFAULT;
	
	
	/**
	 * Dual mode.
	 */
	protected boolean dual = ClassifierAbstract.DUAL_DEFAULT;
	
	
	/**
	 * Baseline mode.
	 */
	protected boolean baseline = ClassifierAbstract.BASELINE_DEFAULT;

	
	/**
	 * Number of blocks.
	 */
	protected int blocks = TransformerBasic.BLOCKS_NUMBER_DEFAULT;
	
	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ClassifierBuilder(int neuronChannel) {
		this.neuronChannel = neuronChannel;
	}

	
	/**
	 * Default constructor.
	 */
	public ClassifierBuilder() {
		
	}
	
	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	public int getNeuronChannel() {
		return neuronChannel;
	}
	
	
	/**
	 * Setting neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return this builder.
	 */
	public ClassifierBuilder setNeuronChannel(int neuronChannel) {
		this.neuronChannel = neuronChannel;
		return this;
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
	 * Getting blocks.
	 * @return blocks.
	 */
	public int getBlocks() {
		return blocks;
	}
	
	
	/**
	 * Setting blocks.
	 * @param batches blocks.
	 * @return this builder.
	 */
	public ClassifierBuilder setBlocks(int blocks) {
		this.blocks = blocks;
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
		case tramac:
			classifier = TransformerClassifier.create(neuronChannel, true); 
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
			mac.paramSetVectorized(vectorized);
			mac.paramSetAdjust(adjust);
			mac.paramSetDual(dual);
			mac.paramSetBaseline(baseline);
		}
		else if (classifier instanceof TransformerClassifier) {
			TransformerClassifier tramac = (TransformerClassifier)classifier;
			tramac.paramSetLearningRate(learningRate);
			tramac.paramSetBatches(batches);
			tramac.paramSetConv(conv);
			tramac.paramSetVectorized(vectorized);
			tramac.paramSetAdjust(adjust);
			tramac.paramSetDual(dual);
			tramac.paramSetBaseline(baseline);
			tramac.paramSetBlocksNumber(blocks);
		}
		else if (classifier instanceof StackClassifier) {
			
		}
		
		return classifier;
	}
	
	
	/**
	 * Build classifier.
	 * @param model classifier model.
	 * @return classifier model.
	 */
	public Classifier build(ClassifierModel model) {
		setModel(model);
		return build();
	}
	
	
	/**
	 * Creating builder by user entering.
	 * @param in input stream.
	 * @param out output stream.
	 * @return builder.
	 */
	static ClassifierBuilder enter(InputStream in, OutputStream out) {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);
		
		int defaultModel = 0;
		int model = defaultModel;
		printer.print("Model (0-mac, 1-tramac, 2-stack) (default " + defaultModel + " is mac):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) model = Integer.parseInt(line);
		} catch (Throwable e) {}
		if (Double.isNaN(model)) model = defaultModel;
		if (model <= 0) model = defaultModel;
		printer.println("Model is " + model + "\n");
		
		int defaultRasterChannel = 3;
		int rasterChannel = defaultRasterChannel;
		printer.print("Raster channel (1, 2, 3, 4) (default " + defaultRasterChannel + "):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) rasterChannel = Integer.parseInt(line);
		} catch (Throwable e) {}
		if (Double.isNaN(rasterChannel)) rasterChannel = defaultRasterChannel;
		if (rasterChannel < defaultRasterChannel) rasterChannel = defaultRasterChannel;
		printer.println("Raster channel is " + rasterChannel + "\n");
	
		double defaultlr = Network.LEARN_RATE_DEFAULT;
		double lr = defaultlr;
		printer.print("Enter starting learning rate (default " + defaultlr + "):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) lr = Double.parseDouble(line);
		} catch (Throwable e) {}
		if (Double.isNaN(lr)) lr = defaultlr;
		if (lr <= 0 || lr > 1) lr = defaultlr;
		printer.println("Starting learning rate is " + lr + "\n");
	
		int defaultBatches = 100;
		int batches = defaultBatches;
		printer.print("Batches (default " + batches + "):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) batches = Integer.parseInt(line);
		} catch (Throwable e) {}
		if (Double.isNaN(batches)) batches = defaultBatches;
		if (batches <= 0) batches = defaultBatches;
		printer.println("Batches are " + batches + "\n");
	
		boolean conv = ClassifierAbstract.CONV_DEFAULT;
		printer.print("Including convolutional network (" + conv + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) conv = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Including convolutional network is " + conv + "\n");
	
		boolean vectorized = MatrixNetworkAbstract.VECTORIZED_DEFAULT;
		printer.print("Vectorization (" + vectorized + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) vectorized = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Vectorization is " + vectorized + "\n");
	
		boolean adjust = ClassifierAbstract.ADJUST_DEFAULT;
		printer.print("Adjustment (" + adjust + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) adjust = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Adjustment is " + adjust + "\n");
	
		boolean dual = ClassifierAbstract.DUAL_DEFAULT;
		printer.print("Dual mode (" + dual + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) dual = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Dual mode is " + dual + "\n");
		
		int defaultBlocks = TransformerBasic.BLOCKS_NUMBER_DEFAULT;
		int blocks = defaultBlocks;
		printer.print("Blocks (default " + defaultBlocks + "):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) blocks = Integer.parseInt(line);
		} catch (Throwable e) {}
		if (Double.isNaN(blocks)) blocks = defaultBlocks;
		if (blocks <= 0) blocks = defaultBlocks;
		printer.println("Blocks are " + blocks + "\n");

		ClassifierBuilder builder = new ClassifierBuilder(rasterChannel);
		switch (model) {
		case 0:
			builder.setModel(ClassifierModel.mac);
			break;
		case 1:
			builder.setModel(ClassifierModel.tramac);
			break;
		case 2:
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
		builder.setBlocks(blocks);
		return builder;
	}
	
	
}
