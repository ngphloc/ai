package net.ea.ann.classifier;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.Record;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class implement forest classifier.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ForestClassifier extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for tree model.
	 */
	public final static String TREE_MODEL_FIELD = "forestclass_tree_model";
	
	
	/**
	 * Default value for tree model.
	 */
	public final static int TREE_MODEL_DEFAULT = 0;

	
	/**
	 * This enum represents tree model.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static enum TreeModel {
		
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

	}

	
	/**
	 * Internal classifiers.
	 */
	protected ClassifierAbstract[] trees = null;
	
	
	/**
	 * Final output.
	 */
	protected Matrix output = null;
	
	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	public ForestClassifier(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
		this.config.put(TransformerClassifierAbstract.BLOCKS_NUMBER_FIELD, TransformerClassifierAbstract.BLOCKS_NUMBER_DEFAULT);
		this.config.put(TREE_MODEL_FIELD, TREE_MODEL_DEFAULT);
		
		try {
			ClassifierAbstract classifier = new VGG(neuronChannel);
			this.config.putAll(classifier.getConfig());
			classifier.close();

			classifier = new MatrixClassifier(neuronChannel);
			this.config.putAll(classifier.getConfig());
			classifier.close();
			
			classifier = new TransformerClassifier(neuronChannel);
			this.config.putAll(classifier.getConfig());
			classifier.close();
		} catch (Throwable e) {Util.trace(e);}
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ForestClassifier(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	/**
	 * Creating individual classifier.
	 * @return individual classifier.
	 */
	protected ClassifierAbstract createTree() {
		TreeModel model = paramGetTreeModel();
		ClassifierAbstract classifier = null;
		switch (model) {
		case vgg:
			classifier = VGG.create(this.neuronChannel, paramGetRasterChannel(), this.paramIsNorm());
			break;
		case mac:
			classifier = MatrixClassifier.create(this.neuronChannel, paramGetRasterChannel(), this.paramIsNorm());
			break;
		case tramac:
			classifier = TransformerClassifier.create(this.neuronChannel, paramGetRasterChannel(), this.paramIsNorm());
			break;
		default:
			classifier = MatrixClassifier.create(this.neuronChannel, paramGetRasterChannel(), this.paramIsNorm());
			break;
		}
		classifier.updateConfig(this.config);
		return classifier;
	}
	
	
	@Override
	public void reset() {
		super.reset();
		this.trees = null;
		this.output = null;
	}


	@Override
	void updateConfig() {
		super.updateConfig();
		if (this.trees == null) return;
		for (ClassifierAbstract tree : this.trees) tree.updateConfig(this.config);
	}


	@Override
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2)) return false;
		this.trees = null;
		this.output = null;
		
		int classCount = getNumberOfClasses(0);
		this.trees = new ClassifierAbstract[classCount];
		for (int i = 0; i < classCount; i++) {
			this.trees[i] = createTree();
			if (!this.trees[i].initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2)) return false;
		}
		
		Matrix output = this.trees[0].getOutput();
		this.output = output.create(new Size(output.columns(), output.rows()));
		return true;
	}
	
	
	/**
	 * Validating this forest.
	 * @return true if this forest is valid.
	 */
	public boolean validate() {
		return trees != null && trees.length > 0 && output != null;
	}


	@Override
	protected Matrix getOutput() {
		return output;
	}
	
	
	@Override
	protected Matrix toMatrix(Raster raster) {
		return validate() ? trees[0].toMatrix(raster) : null;
	}


	/**
	 * Evaluating tree at specified index.
	 * @param treeIndex specified index.
	 * @param input input.
	 * @param params additional parameters.
	 * @return output.
	 */
	private Matrix evaluate(int treeIndex, Matrix input, Object...params) {
		if (!validate() || treeIndex >= this.trees.length) return null;
		return this.trees[treeIndex].evaluate(input, params);
	}
	
	
	@Override
	protected Matrix evaluate(Matrix input, Object...params) {
		if (!validate()) return null;
		updateConfig();
		
		Matrix finalOutput = null;
		for (int treeIndex = 0; treeIndex < trees.length; treeIndex++) {
			Matrix output = evaluate(treeIndex, input, params);
			finalOutput = finalOutput != null ? finalOutput.add(output) : output;
		}
		finalOutput = finalOutput.divide0(trees.length);
		return (this.output = finalOutput);
	}


	@Override
	protected Error[] learn(Iterable<Record> sample) {
		if (!validate()) return null;
		updateConfig();
		
		Error[] accum = null;
		for (ClassifierAbstract tree : this.trees) {
			try {
				Error[] errors = tree.learn(sample);
				accum = accum != null ? Error.accum(accum, errors) : errors;
			} catch (Throwable e) {Util.trace(e);}
		}
		return accum;
	}


//	/**
//	 * Getting the number of blocks.
//	 * @return the number of blocks.
//	 */
//	int paramGetBlocksNumber() {
//		int blocksNumber = config.getAsInt(TransformerClassifierAbstract.BLOCKS_NUMBER_FIELD);
//		return blocksNumber < 1 ? TransformerClassifierAbstract.BLOCKS_NUMBER_DEFAULT : blocksNumber;
//	}
	
	
	/**
	 * Converting an integer into tree model.
	 * @param treeModelIndex tree model index.
	 * @return tree model.
	 */
	public static TreeModel toTreeModel(int treeModelIndex) {
		TreeModel treeModel = TreeModel.vgg;
		switch (treeModelIndex) {
		case 0:
			treeModel = TreeModel.vgg;
			break;
		case 1:
			treeModel = TreeModel.mac;
			break;
		case 2:
			treeModel = TreeModel.tramac;
			break;
		default:
			treeModel = TreeModel.vgg;
			break;
		}
		return treeModel;
	}
	
	
	/**
	 * Converting tree model into index.
	 * @param treeModel tree model.
	 * @return index of tree model.
	 */
	static int toTreeModelIndex(TreeModel treeModel) {
		int treeModelIndex = TREE_MODEL_DEFAULT;
		switch (treeModel) {
		case vgg:
			treeModelIndex = 0;
			break;
		case mac:
			treeModelIndex = 1;
			break;
		case tramac:
			treeModelIndex = 2;
			break;
		default:
			treeModelIndex = TREE_MODEL_DEFAULT;
			break;
		}
		return treeModelIndex;
	}
	
	
	/**
	 * Converting tree model into text.
	 * @param treeModel tree model.
	 * @return text of tree model.
	 */
	static String toTreeModelText(TreeModel treeModel) {
		String treeModelText = "vgg";
		switch (treeModel) {
		case vgg:
			treeModelText = "vgg";
			break;
		case mac:
			treeModelText = "mac";
			break;
		case tramac:
			treeModelText = "tramac";
			break;
		default:
			treeModelText = "vgg";
			break;
		}
		return treeModelText;
	}

	
	/**
	 * Getting tree model.
	 * @return tree model.
	 */
	TreeModel paramGetTreeModel() {
		int treeModelIndex = config.getAsInt(TREE_MODEL_FIELD);
		treeModelIndex = treeModelIndex < 0 ? TREE_MODEL_DEFAULT : treeModelIndex;
		return toTreeModel(treeModelIndex);
	}
	
	
	/**
	 * Setting tree model.
	 * @param treeModel tree model.
	 * @return this classifier.
	 */
	ForestClassifier paramSetTreeModel(TreeModel treeModel) {
		int treeModelIndex = toTreeModelIndex(treeModel);
		config.put(TREE_MODEL_FIELD, treeModelIndex);
		return this;
	}

	
	/**
	 * Setting tree model.
	 * @param treeModelIndex index of tree model.
	 * @return this classifier.
	 */
	ForestClassifier paramSetTreeModel(int treeModelIndex) {
		TreeModel treeModel = toTreeModel(treeModelIndex);
		return paramSetTreeModel(treeModel);
	}

	
	/**
	 * Setting the number of blocks.
	 * @param blockNumber the number of blocks.
	 * @return this classifier.
	 */
	ForestClassifier paramSetBlocksNumber(int blockNumber) {
		blockNumber = blockNumber < 1 ? TransformerClassifierAbstract.BLOCKS_NUMBER_DEFAULT : blockNumber;
		config.put(TransformerClassifierAbstract.BLOCKS_NUMBER_FIELD, blockNumber);
		return this;
	}

	
	/**
	 * Creating matrix neural classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param rasterChannel raster channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static ForestClassifier create(int neuronChannel, int rasterChannel, boolean isNorm) {
		ForestClassifier forest = new ForestClassifier(neuronChannel);
		forest.paramSetRasterChannel(rasterChannel);
		forest.paramSetNorm(isNorm);
		return forest;
	}


}
