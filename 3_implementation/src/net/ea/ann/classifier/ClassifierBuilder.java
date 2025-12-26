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

import net.ea.ann.classifier.ForestClassifier.TreeModel;
import net.ea.ann.core.Network;
import net.ea.ann.core.function.Function;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.WeightSpec;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.FilterSpec.PoolType;
import net.ea.ann.mane.WeightSpec.Type;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.Size;

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
	 * Default batches.
	 */
	public final static int DEFAULT_BATCHES = 100;
	
	
	/**
	 * This enum represents classifier model.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static enum ClassifierModel {
		
		/**
		 * VGG model.
		 */
		vgg,

		/**
		 * Matrix neural network classifier.
		 */
		mac,
		
		/**
		 * Transformer classifier.
		 */
		tramac,

		/**
		 * Forest classifier.
		 */
		forest,

		/**
		 * Stack network classifier.
		 */
		stack,
		
	}
	
	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = RasterAbstract.NEURON_CHANNEL_DEFAULT;
	
	
	/**
	 * Neuron channel.
	 */
	protected int rasterChannel = RasterAbstract.RASTER_CHANNEL_DEFAULT;

	
	/**
	 * Classifier model.
	 */
	ClassifierModel model = ClassifierModel.vgg;
	
	
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
	 * Filter size.
	 */
	protected int filterSize = ClassifierAbstract.FILTER_SIZE_DEFAULT;
	
	
	/**
	 * Vectorization mode.
	 */
	protected boolean vectorized = MatrixNetworkAbstract.VECTORIZED_DEFAULT;
	
	
	/**
	 * Baseline mode.
	 */
	protected boolean baseline = ClassifierAbstract.BASELINE_DEFAULT;

	
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
	protected int depth = ClassifierAbstract.DEPTH_DEFAULT;

	
	/**
	 * Cross-entropy trainer mode.
	 */
	protected boolean entropyTrainer = ClassifierAbstract.ENTROPY_TRAINER_DEFAULT;

	
	/**
	 * Number of blocks.
	 */
	protected int blocks = TransformerClassifierAbstract.BLOCKS_NUMBER_DEFAULT;
	
	
	/**
	 * Pooling type.
	 */
	protected PoolType poolType = ClassifierAbstract.POOL_TYPE_DEFAULT;

	
	/**
	 * Weight type.
	 */
	protected Type weightType = ClassifierAbstract.WEIGHT_TYPE_DEFAULT;
	
	
	/**
	 * Number of filters,
	 */
	protected int filtersNumber = net.ea.ann.mane.beans.VGG.FILTERS_NUMBER_DEFAULT;
	
	
	/**
	 * Middle size.
	 */
	protected Size middleSize = net.ea.ann.mane.beans.VGG.MIDDLE_SIZE_DEFAULT;
	
	
	/**
	 * Length of feed-forward network.,
	 */
	protected int ffnLength = net.ea.ann.mane.beans.VGG.FFN_LENGTH_DEFAULT;

	
	/**
	 * Tree model.
	 */
	protected TreeModel treeModel = TreeModel.mac;
	
	
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
	public int getNeuronChannel() {return neuronChannel;}
	
	
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
	 * Getting raster channel.
	 * @return raster channel.
	 */
	public int getRasterChannel() {return rasterChannel;}
	
	
	/**
	 * Setting raster channel.
	 * @param rasterChannel raster channel.
	 * @return this builder.
	 */
	public ClassifierBuilder setRasterChannel(int rasterChannel) {
		this.rasterChannel = rasterChannel;
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
	 * Converting an integer into classifier model.
	 * @param modelIndex model index.
	 * @return classifier model.
	 */
	static ClassifierModel toModel(int modelIndex) {
		ClassifierModel model = ClassifierModel.vgg;
		switch (modelIndex) {
		case 0:
			model = ClassifierModel.vgg;
			break;
		case 1:
			model = ClassifierModel.mac;
			break;
		case 2:
			model = ClassifierModel.tramac;
			break;
		case 3:
			model = ClassifierModel.forest;
			break;
		case 4:
			model = ClassifierModel.stack;
			break;
		default:
			model = ClassifierModel.vgg;
			break;
		}
		return model;
	}
	
	
	/**
	 * Converting an integer into classifier model.
	 * @param modelIndex model index.
	 * @return classifier model.
	 */
	public static ClassifierModel toModel(String modelText) {
		ClassifierModel model = ClassifierModel.vgg;
		switch (modelText.toLowerCase()) {
		case "vgg":
			model = ClassifierModel.vgg;
			break;
		case "mac":
			model = ClassifierModel.mac;
			break;
		case "tramac":
			model = ClassifierModel.tramac;
			break;
		case "forest":
			model = ClassifierModel.forest;
			break;
		case "stack":
			model = ClassifierModel.stack;
			break;
		default:
			model = ClassifierModel.vgg;
			break;
		}
		return model;
	}

	
	/**
	 * Converting classifier model into integer.
	 * @param model classifier model.
	 * @return integer.
	 */
	static int toModelIndex(ClassifierModel model) {
		int modelIndex = 0;
		switch (model) {
		case vgg:
			modelIndex = 0; 
			break;
		case mac:
			modelIndex = 1; 
			break;
		case tramac:
			modelIndex = 2; 
			break;
		case forest:
			modelIndex = 3; 
			break;
		case stack:
			modelIndex = 4; 
			break;
		default:
			break;
		}
		return modelIndex;
	}
	
	
	/**
	 * Converting classifier model into text.
	 * @param model classifier model.
	 * @return text.
	 */
	static String toModelText(ClassifierModel model) {
		String modelText = "vgg";
		switch (model) {
		case vgg:
			modelText = "vgg";
			break;
		case mac:
			modelText = "mac";
			break;
		case tramac:
			modelText = "tramac";
			break;
		case forest:
			modelText = "forest";
			break;
		case stack:
			modelText = "stack";
			break;
		default:
			break;
		}
		return modelText;
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
	 * Setting classifier model.
	 * @param modelIndex
	 * @return this builder.
	 */
	public ClassifierBuilder setModel(int modelIndex) {
		setModel(toModel(modelIndex));
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
	 * Getting filter size.
	 * @return filter size.
	 */
	public int getFilterSize() {
		return filterSize;
	}
	
	
	/**
	 * Setting filter size.
	 * @param filterSize filter size.
	 * @return this builder.
	 */
	public ClassifierBuilder setFilterSize(int filterSize) {
		this.filterSize = filterSize;
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
	 * Setting dual mode.
	 * @param dual dual mode.
	 * @return this builder.
	 */
	public ClassifierBuilder setDual(boolean dual) {
		this.dual = dual;
		return this;
	}
	
	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	public int getDepth() {
		return depth;
	}
	
	
	/**
	 * Setting depth.
	 * @param depth depth.
	 * @return this builder.
	 */
	public ClassifierBuilder setDepth(int depth) {
		this.depth = depth;
		return this;
	}

	
	/**
	 * Getting cross-entropy mode.
	 * @return cross-entropy mode.
	 */
	public boolean isEntropyTrainer() {
		return entropyTrainer;
	}
	
	
	/**
	 * Setting cross-entropy trainer mode.
	 * @param entropyTrainer cross-entropy trainer mode.
	 * @return this builder.
	 */
	public ClassifierBuilder setEntropyTrainer(boolean entropyTrainer) {
		this.entropyTrainer = entropyTrainer;
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
	 * Getting pooling type.
	 * @return pooling type.
	 */
	public PoolType getPoolType() {
		return poolType;
	}
	
	
	/**
	 * Setting pooling type.
	 * @param poolType pooling type.
	 * @return this builder.
	 */
	public ClassifierBuilder setPoolType(PoolType poolType) {
		this.poolType = poolType;
		return this;
	}

	
	/**
	 * Getting weight type.
	 * @return weight type.
	 */
	public Type getWeightType() {
		return weightType;
	}
	
	
	/**
	 * Setting weight type.
	 * @param weightType weight type.
	 * @return this builder.
	 */
	public ClassifierBuilder setWeightType(Type weightType) {
		this.weightType = weightType;
		return this;
	}

	
	/**
	 * Getting number of filters.
	 * @return number of filters.
	 */
	public int getFiltersNumber() {
		return filtersNumber;
	}
	
	
	/**
	 * Setting number of filters.
	 * @param filtersNumber number of filters.
	 * @return this builder.
	 */
	public ClassifierBuilder setFiltersNumber(int filtersNumber) {
		this.filtersNumber = filtersNumber;
		return this;
	}

	
	/**
	 * Getting middle size.
	 * @return middle size.
	 */
	public Size getMiddleSize() {
		return middleSize;
	}
	
	
	/**
	 * Setting middle size.
	 * @param middle size.
	 * @return this builder.
	 */
	public ClassifierBuilder setMiddleSize(Size middleSize) {
		this.middleSize = middleSize;
		return this;
	}

	
	/**
	 * Getting length of feed-forward network.
	 * @return length of feed-forward network.
	 */
	public int getFFNLength() {
		return ffnLength;
	}
	
	
	/**
	 * Setting length of feed-forward network.
	 * @param ffnLength length of feed-forward network.
	 * @return this builder.
	 */
	public ClassifierBuilder setFFNLength(int ffnLength) {
		this.ffnLength = ffnLength;
		return this;
	}

	
	/**
	 * Getting tree model.
	 * @return tree model.
	 */
	public TreeModel getTreeModel() {
		return treeModel;
	}
	
	
	/**
	 * Setting tree model.
	 * @param treeModel tree model.
	 * @return this builder.
	 */
	public ClassifierBuilder setTreeModel(TreeModel treeModel) {
		this.treeModel = treeModel;
		return this;
	}

	
	/**
	 * Setting tree model.
	 * @param treeModelIndex tree model index.
	 * @return this builder.
	 */
	public ClassifierBuilder setTreeModel(int treeModelIndex) {
		this.treeModel = ForestClassifier.toTreeModel(treeModelIndex);
		return this;
	}

	
	/**
	 * Build classifier.
	 * @return classifier.
	 */
	public Classifier build() {
		Classifier classifier = null;
		switch (model) {
		case vgg:
			classifier = VGG.create(neuronChannel, rasterChannel, true); 
			break;
		case mac:
			classifier = MatrixClassifier.create(neuronChannel, rasterChannel, true); 
			break;
		case tramac:
			classifier = TransformerClassifier.create(neuronChannel, rasterChannel, true);
			break;
		case forest:
			classifier = ForestClassifier.create(neuronChannel, rasterChannel, true);
			break;
		case stack:
			classifier = StackClassifier.create(neuronChannel, rasterChannel, true);
			break;
		default:
			break;
		}
		
		if (classifier instanceof ClassifierAbstract) {
			ClassifierAbstract ca = (ClassifierAbstract)classifier;
			ca.paramSetLearningRate(learningRate);
			ca.paramSetBatches(batches);
			ca.paramSetConv(conv);
			ca.paramSetFilterSize(filterSize);
			ca.paramSetPoolType(poolType);
			ca.paramSetWeightType(weightType);
			ca.paramSetVectorized(vectorized);
			ca.paramSetBaseline(baseline);
			ca.paramSetAdjust(adjust);
			ca.paramSetDual(dual);
			ca.paramSetDepth(depth);
			ca.paramSetEntropyTrainer(entropyTrainer);
		}
		
		if (classifier instanceof VGG) {
			VGG vgg = (VGG)classifier;
			vgg.paramSetBlocksNumber(blocks);
			vgg.paramSetLayersNumber(depth);
			vgg.paramSetFiltersNumber(filtersNumber);
			vgg.paramSetVGGMiddleSize(middleSize);
			vgg.paramSetFFNLength(ffnLength);
		}
		else if (classifier instanceof MatrixClassifier) {

		}
		else if (classifier instanceof TransformerClassifier) {
			TransformerClassifier tramac = (TransformerClassifier)classifier;
			tramac.paramSetBlocksNumber(blocks);
		}
		else if (classifier instanceof ForestClassifier) {
			ForestClassifier forest = (ForestClassifier)classifier;
			forest.paramSetBlocksNumber(blocks);
			forest.paramSetTreeModel(treeModel);
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
		
		int defaultModelIndex = toModelIndex(ClassifierModel.vgg);
		int modelIndex = defaultModelIndex;
		printer.print("Model (0-vgg, 1-mac, 2-tramac, 3-forest, 4-stack) (default " + defaultModelIndex + " is " + toModel(defaultModelIndex) + "):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) modelIndex = Integer.parseInt(line);
		} catch (Throwable e) {}
		if (Double.isNaN(modelIndex)) modelIndex = defaultModelIndex;
		if (modelIndex < 0) modelIndex = defaultModelIndex;
		printer.println("Model is " + toModel(modelIndex) + "\n");
		
		int defaultRasterChannel = RasterAbstract.RASTER_CHANNEL_DEFAULT;
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
	
		int defaultBatches = DEFAULT_BATCHES;
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
		printer.print("Including convolutional neural network (" + conv + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) conv = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Including convolutional neural network is " + conv + "\n");
	
		int defaultFilterSize = ClassifierAbstract.FILTER_SIZE_DEFAULT;
		int filterSize = defaultFilterSize;
		if (conv) {
			printer.print("Filter size (default " + filterSize + "):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) filterSize = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(filterSize)) filterSize = defaultFilterSize;
			if (filterSize <= 0) filterSize = defaultFilterSize;
			printer.println("Filter size is " + filterSize + "\n");
		}
		
		PoolType poolType = PoolType.max;
		if (conv) {
			printer.print("Pooling type (0-max, 1-average) (default pooling type" + " is " + poolType + "):");
			int poolTypeIndex = FilterSpec.poolTypeToInt(poolType);
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) poolTypeIndex = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (!Double.isNaN(poolTypeIndex) && poolTypeIndex >= 0) poolType = FilterSpec.intToPoolType(poolTypeIndex);
			printer.println("Pooling type is " + poolType + "\n");
		}

		Type weightType = Type.normal;
		if (!conv) {
			printer.print("Weight type (0-normal, 1-transformer) (default weight type" + " is " + weightType + "):");
			int weightTypeIndex = WeightSpec.typeToInt(weightType);
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) weightTypeIndex = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (!Double.isNaN(weightTypeIndex) && weightTypeIndex >= 0) weightType = WeightSpec.intToType(weightTypeIndex);
			printer.println("Weight type is " + weightType + "\n");
		}

		boolean vectorized = MatrixNetworkAbstract.VECTORIZED_DEFAULT;
		printer.print("Vectorization (" + vectorized + " is default):");
		try {
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) vectorized = Boolean.parseBoolean(line);
		} catch (Throwable e) {}
		printer.println("Vectorization is " + vectorized + "\n");
	
		boolean baseline = ClassifierAbstract.BASELINE_DEFAULT;
		if (!baseline) {
			printer.print("Baseline (" + baseline + " is default):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) baseline = Boolean.parseBoolean(line);
			} catch (Throwable e) {}
			printer.println("Baseline is " + baseline + "\n");
		}

		boolean adjust = baseline ? ClassifierAbstract.ADJUST_DEFAULT : false;
		if (adjust) {
			printer.print("Adjustment (" + adjust + " is default):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) adjust = Boolean.parseBoolean(line);
			} catch (Throwable e) {}
			printer.println("Adjustment is " + adjust + "\n");
		}
	
		boolean dual = conv ? ClassifierAbstract.DUAL_DEFAULT : false;
		if (conv && dual) {
			printer.print("Dual mode (" + dual + " is default):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) dual = Boolean.parseBoolean(line);
			} catch (Throwable e) {}
			printer.println("Dual mode is " + dual + "\n");
		}
		
		int defaultDepth = ClassifierAbstract.DEPTH_DEFAULT;
		int depth = defaultDepth;
		if (depth <= 2) {
			printer.print("Depth (default " + depth + "):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) depth = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(depth)) depth = defaultDepth;
			if (depth <= 0) depth = defaultDepth;
			printer.println("Depth is " + depth + "\n");
		}

		boolean entropyTrainer = ClassifierAbstract.ENTROPY_TRAINER_DEFAULT;
		if (!entropyTrainer) {
			printer.print("Cross-entropy trainer mode (" + entropyTrainer + " is default):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) entropyTrainer = Boolean.parseBoolean(line);
			} catch (Throwable e) {}
			printer.println("Cross-entropy trainer mode is " + entropyTrainer + "\n");
		}

		ClassifierBuilder builder = new ClassifierBuilder(rasterChannel);
		builder.setModel(modelIndex);
		builder.setLearningRate(lr);
		builder.setBatches(batches);
		builder.setConv(conv);
		builder.setFilterSize(filterSize);
		builder.setPoolType(poolType);
		builder.setWeightType(weightType);
		builder.setVectorized(vectorized);
		builder.setBaseline(baseline);
		builder.setAdjust(adjust);
		builder.setDual(dual);
		builder.setDepth(depth);
		builder.setEntropyTrainer(entropyTrainer);

		if (builder.getModel() == ClassifierModel.vgg) {
			int defaultMiddleSize = net.ea.ann.mane.beans.VGG.MIDDLE_SIZE_DEFAULT.width;
			int middleSize = defaultMiddleSize;
			printer.print("Middle size (default (" + defaultMiddleSize + ", " + defaultMiddleSize + ") ):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) middleSize = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(middleSize)) middleSize = defaultMiddleSize;
			if (middleSize <= 0) middleSize = defaultMiddleSize;
			printer.println("Middle size is (" + middleSize + ", " + middleSize + ") )\n");
			builder.setMiddleSize(new Size(middleSize, middleSize));

			int defaultFiltersNumber = net.ea.ann.mane.beans.VGG.FILTERS_NUMBER_DEFAULT;
			int filtersNumber = defaultFiltersNumber;
			printer.print("Number of " + (conv ? "filters" : "weights") + " is (default " + defaultFiltersNumber + "):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) filtersNumber = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(filtersNumber)) filtersNumber = defaultFiltersNumber;
			if (filtersNumber <= 0) filtersNumber = defaultFiltersNumber;
			printer.println("Number of " + (conv ? "filters" : "weights") + " is " + filtersNumber + "\n");
			builder.setFiltersNumber(filtersNumber);

			int defaultFFNLength = net.ea.ann.mane.beans.VGG.FFN_LENGTH_DEFAULT;
			int ffnLength = defaultFFNLength;
			printer.print("Feed-forward network length (default " + defaultFFNLength + "):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) ffnLength = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(ffnLength)) ffnLength = defaultFFNLength;
			if (ffnLength <= 0) ffnLength = defaultFFNLength;
			printer.println("Feed-forward network length is " + ffnLength + "\n");
			builder.setFFNLength(ffnLength);
		}
		
		if (builder.getModel() == ClassifierModel.vgg || builder.getModel() == ClassifierModel.tramac) {
			int defaultBlocks = TransformerClassifierAbstract.BLOCKS_NUMBER_DEFAULT;
			int blocks = defaultBlocks;
			printer.print("Blocks (default " + defaultBlocks + "):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) blocks = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(blocks)) blocks = defaultBlocks;
			if (blocks <= 0) blocks = defaultBlocks;
			printer.println("Blocks are " + blocks + "\n");
			builder.setBlocks(blocks);
		}
		
		if (builder.getModel() == ClassifierModel.forest) {
			int defaultTreeModelIndex = ForestClassifier.toTreeModelIndex(TreeModel.mac);
			int treeModelIndex = defaultTreeModelIndex;
			printer.print("Tree model (0-mac, 1-tramac) (default " + defaultTreeModelIndex + " is " + ForestClassifier.toTreeModel(defaultTreeModelIndex) + "):");
			try {
				String line = scanner.nextLine().trim();
				if (!line.isBlank() && !line.isEmpty()) treeModelIndex = Integer.parseInt(line);
			} catch (Throwable e) {}
			if (Double.isNaN(treeModelIndex)) treeModelIndex = defaultTreeModelIndex;
			if (treeModelIndex < 0) treeModelIndex = defaultTreeModelIndex;
			printer.println("Tree model is " + ForestClassifier.toTreeModel(treeModelIndex) + "\n");
			builder.setTreeModel(treeModelIndex);
		}
		
		return builder;
	}
	
	
}
