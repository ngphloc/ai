/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorWeighted;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterProperty.Label;
import net.ea.ann.raster.Size;

/**
 * This class is an extension of VGG model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class VGGExt extends VGG {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default value for by-column flag.
	 */
	public final static String BYCOLUMN_FIELD = "vgg_bycolumn";

	
	/**
	 * Default value for by-column flag.
	 */
	public final static boolean BYCOLUMN_DEFAULT = TaskTrainerLossEntropy.BYCOLUMN;

	
	/**
	 * Field of the number elements of a combination.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	public static final String COMB_NUMBER_FIELD = "vgg_class_comb_number";
	
	
	/**
	 * Default value for the field of the number elements of a combination.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	public static final int COMB_NUMBER_DEFAULT = GeneratorWeighted.COMB_NUMBER_DEFAULT;
	
	
	/**
	 * Field of class number.
	 */
	public static final String CLASS_NUMBER_FIELD = "vgg_class_number";
	
	
	/**
	 * Default value for field of class number.
	 */
	public static final int CLASS_NUMBER_DEFAULT = 10;
	
	
	/**
	 * Field for label smoothing.
	 * If true, label smoothing is applied so that one-hot class vector become class probability in which the probability of certain class is large enough but smaller than 1, for example {@link #MAX_CLASS_PROB_DEFAULT}.
	 */
	public final static String LABEL_SMOOTH_FIELD = "vgg_class_label_smooth";
	
	
	/**
	 * Default value for field of label smoothing.
	 * If true, label smoothing is applied so that one-hot class vector become class probability in which the probability of certain class is large enough but smaller than 1, for example {@link #MAX_CLASS_PROB_DEFAULT}.
	 */
	public final static boolean LABEL_SMOOTH_DEFAULT = true;

	
	/**
	 * Field of maximum class probability.
	 */
	public final static String MAX_CLASS_PROB_FIELD = "vgg_class_max_prob";

	
	/**
	 * Default value for field of maximum class probability.
	 */
	private final static double MAX_CLASS_PROB_DEFAULT = 0.9;
	
	
	/**
	 * This interface specifies trainer applying into VGG for specific task with sample of raster.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	@FunctionalInterface
	public static interface RasterTaskTrainer {
		
		/**
		 * Learning layer as VGG.
		 * @param layer layer as matrix neural network.
		 * @param sample sample.
		 * @param propagate propagation flag.
		 * @param learningRate learning rate.
		 * @param params additional parameters.
		 * @return learning biases.
		 */
		Error[] train(MatrixLayer layer, Iterable<Raster> sample, boolean propagate, double learningRate, Object...params);

		/**
		 * Learning layer as VGG.
		 * @param layer layer as matrix neural network.
		 * @param sample sample.
		 * @param learningRate learning rate.
		 * @param params additional parameters.
		 * @return learning biases.
		 */
		default Error[] train(MatrixLayer layer, Iterable<Raster> sample, double learningRate, Object...params) {
			return train(layer, sample, true, learningRate, params);
		}

	}

		
	/**
	 * List of outputs-classes maps. For an outputs-classes map whose each element is a subtask which is a combination given classes.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	protected List<Map<Integer, int[]>> outputClassMaps = Util.newList(0);
	
	
	/**
	 * List of classes-outputs maps. For a classes-outputs map whose each element is a class pointer to the subtask which is a combination given classes.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	protected List<Map<Integer, int[]>> classOutputMaps = Util.newList(0);

	
	/**
	 * List of class-label maps.
	 */
	protected List<Map<Integer, Label>> classMaps = Util.newList(0);

	
	/**
	 * List of trainers.
	 */
	protected List<RasterTaskTrainer> trainers = Util.newList(0);

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGGExt(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(BYCOLUMN_FIELD, BYCOLUMN_DEFAULT);
		config.put(COMB_NUMBER_FIELD, COMB_NUMBER_DEFAULT);
		config.put(CLASS_NUMBER_FIELD, CLASS_NUMBER_DEFAULT);
		config.put(LABEL_SMOOTH_FIELD, LABEL_SMOOTH_DEFAULT);
		config.put(MAX_CLASS_PROB_FIELD, MAX_CLASS_PROB_DEFAULT);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public VGGExt(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public VGGExt(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public VGGExt(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	/**
	 * Resetting some properties.
	 */
	private void reset0() {
		outputClassMaps.clear();
		classOutputMaps.clear();
		classMaps.clear();
	}
	
	
	@Override
	public void reset() {
		super.reset();
		reset0();
	}

	
	/**
	 * Checking whether the model is labeled.
	 * @return whether the model is labeled.
	 */
	public boolean isLabeled() {return classMaps.size() > 0;}
	
	
	/**
	 * Checking whether the sample is labeled.
	 * @param sample sample.
	 * @return true if the sample is labeled.
	 */
	public static boolean isLabeled(Iterable<Raster> sample) {
		for (Raster raster : sample) {
			RasterProperty rp = raster != null ? raster.getProperty() : null;
			if (rp == null || rp.getLabelId() < 0) continue;
			if (rp.getLabelCount() > 0) return true;
		}
		return false;
	}
	
	
	/**
	 * Configure class information.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 * @param nClass number of classes.
	 * @param outputClassMap outputs-classes map whose each element is a subtask which is a combination given classes.
	 * @param classOutputMap classes-outputs map whose each element is a class pointer to the subtask which is a combination given classes.
	 * @return true if configuration is successful.
	 */
	private boolean configClassInfo(int nClass, Map<Integer, int[]> outputClassMap, Map<Integer, int[]> classOutputMap) {
		if (nClass < 1) return false;
		int comb = paramGetCombNumber();
		if (comb < 1 || comb >= nClass) {
			comb = 1;
			paramSetCombNumber(comb);
			return false;
		}
		
		outputClassMap.clear(); //outputs-classes map whose each element is a subtask which is a combination given classes.
		classOutputMap.clear(); //classes-outputs map whose each element is a class pointer to the subtask which is a combination given classes.

		CombinationGenerator cg = new CombinationGenerator(nClass, comb);
		int index = 0;
		while (cg.hasMore()) {
			int[] classIndices = cg.getNext();
			classIndices = Arrays.copyOf(classIndices, classIndices.length);
			Arrays.sort(classIndices);
			outputClassMap.put(index, classIndices);
			index++;
		}
		if (outputClassMap.size() == 0) return false;
		
		for (int classIndex = 0; classIndex < nClass; classIndex++) {
			Set<Integer> outputIndices = outputClassMap.keySet();
			List<Integer> foundOutputIndexList = Util.newList(0);
			for (int outputIndex : outputIndices) {
				int[] classIndices = outputClassMap.get(outputIndex);
				if (Arrays.binarySearch(classIndices, classIndex) >= 0) {
					foundOutputIndexList.add(outputIndex);
				}
			}
			if (foundOutputIndexList.size() == 0) continue;
			
			int[] foundOutputIndices = new int[foundOutputIndexList.size()];
			for (int i = 0; i < foundOutputIndices.length; i++) foundOutputIndices[i] = foundOutputIndexList.get(i);
			Arrays.sort(foundOutputIndices);
			classOutputMap.put(classIndex, foundOutputIndices);
		}
		return classOutputMap.size() > 0;
	}

	
	/**
	 * Configure class information.
	 * @param nCoreClasses the number of rows and columns of core classes.
	 * @return true if configuration is successful.
	 */
	private boolean configClassInfo(Size nCoreClasses) {
		Map<Integer, int[]> outputClassMap = Util.newMap(0);
		Map<Integer, int[]> classOutputMap = Util.newMap(0);
		int nClass = paramIsByColumn() ? nCoreClasses.height : nCoreClasses.width;
		int nClassCount = paramIsByColumn() ? nCoreClasses.width : nCoreClasses.height;
		if (!configClassInfo(nClass, outputClassMap, classOutputMap)) return false;
		
		this.outputClassMaps.clear();
		this.classOutputMaps.clear();
		for (int count = 0; count < nClassCount; count++) {
			this.outputClassMaps.add(outputClassMap);
			this.classOutputMaps.add(classOutputMap);
		}
		
		this.classMaps.clear();
		return this.classOutputMaps.size() > 0;
	}

	
	/**
	 * Getting the number of output groups.
	 * @return the number of output groups.
	 */
	int getNumberOfGroups() {return this.outputClassMaps.size();}
	
	
	/**
	 * Getting the number of outputs given group.
	 * @param groupIndex given group index.
	 * @return the number of outputs given group.
	 */
	int getNumberOfOutputs(int groupIndex) {return this.outputClassMaps.get(groupIndex).size();}
	

	/**
	 * Getting the number of classes given group.
	 * @param groupIndex given group index.
	 * @return the number of classes given group.
	 */
	int getNumberOfClasses(int groupIndex) {return this.classOutputMaps.get(groupIndex).size();}
	
	
	/**
	 * Getting output as result of classification, which is often the output when output of the last layer includes core classes.
	 * All remaining methods in this class {@link VGGExt} called this method when retrieving output, as a convention.
	 * This method can be improved later.
	 * @return output as result of classification.
	 */
	Matrix getCoreOutput() {return getOutput();}
	
	
	/**
	 * Getting output group.
	 * @param output output.
	 * @param groupIndex group index.
	 * @return output group.
	 */
	NeuronValue[] getOutput(Matrix output, int groupIndex) {
		Matrix group = paramIsByColumn() ? output.getColumn(groupIndex) : output.getRow(groupIndex);
		return MatrixUtil.extractValues(group);
	}
	
	
	/**
	 * Getting output.
	 * @param output output.
	 * @param groupIndex group index.
	 * @param outputIndex output index.
	 * @return output value.
	 */
	NeuronValue getOutput(Matrix output, int groupIndex, int outputIndex) {
		return paramIsByColumn() ? output.getColumn(groupIndex).get(outputIndex, 0) :
			output.getRow(groupIndex).get(0, outputIndex);
	}
	
	
	/**
	 * Creating output from core class indices.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.<br/>
	 * In current version, this method does not support recursion of matrix stack and so it support only until 3-dimension matrix stack.
	 * @param classIndices core class indices of group, whose each element is core class index of a group.
	 * @param training training flag.
	 * @return output created from core class indices.
	 */
	Matrix createOutputByClass(int[] classIndices, boolean training) {
		Matrix OUTPUT = this.getCoreOutput();
		int groupCount = getNumberOfGroups();
		if (groupCount <= 0) return null;
		int outputCount = getNumberOfOutputs(0);
		
		int rows = paramIsByColumn() ? outputCount : groupCount;
		int columns = paramIsByColumn() ? groupCount : outputCount;
		Matrix output = OUTPUT instanceof MatrixStack ? ((MatrixStack)OUTPUT).get().create(new Size(columns, rows)) : OUTPUT.create(new Size(columns, rows)); //Fixing date: 2026.06.10.
		if (classIndices == null || classIndices.length == 0) return output;
		
		NeuronValue minor = output.get(0, 0).zero();
		NeuronValue major = minor.unit();
		if (training && paramIsLabelSmooth()) {
			double maxProb = paramGetMaxClassProb();
			int classCount = getNumberOfClasses(0);
			if (maxProb >= 0.5 && maxProb < 1 && classCount > 1) {
				minor = minor.valueOf((1.0-maxProb) / (classCount-1));
				major = major.valueOf(maxProb);
			}
		}
		
		if (paramIsByColumn()) {
			assert (columns == classIndices.length);
			int groups = Math.min(columns, classIndices.length);
			for (int group = 0; group < groups; group++) {
				int classIndex = classIndices[group];
				Map<Integer, int[]> outputClassMap = outputClassMaps.get(group);
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					int[] realClassIndices = outputClassMap.get(outputIndex);
					if (Arrays.binarySearch(realClassIndices, classIndex) >= 0)
						output.set(outputIndex, group, major);
					else
						output.set(outputIndex, group, minor);
				}
				
				//Normalization.
				NeuronValue sum = output.get(0, group).zero();
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					sum = sum.add(output.get(outputIndex, group));
				}
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					NeuronValue value = output.get(outputIndex, group).divide(sum);
					output.set(outputIndex, group, value);
				}
			}
		}
		else {
			assert (rows == classIndices.length);
			int groups = Math.min(rows, classIndices.length);
			for (int group = 0; group < groups; group++) {
				int classIndex = classIndices[group];
				Map<Integer, int[]> outputClassMap = outputClassMaps.get(group);
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					int[] realClassIndices = outputClassMap.get(outputIndex);
					if (Arrays.binarySearch(realClassIndices, classIndex) >= 0)
						output.set(group, outputIndex, major);
					else
						output.set(group, outputIndex, minor);
				}
				
				//Normalization.
				NeuronValue sum = output.get(group, 0).zero();
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					sum = sum.add(output.get(group, outputIndex));
				}
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					NeuronValue value = output.get(group, outputIndex).divide(sum);
					output.set(group, outputIndex, value);
				}
			}
		}

		return OUTPUT instanceof MatrixStack ? new MatrixStack(output, ((MatrixStack)OUTPUT).depth()) : output;
	}

	
	/**
	 * Creating output from class index.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.<br/>
	 * In current version, this method does not support recursion of matrix stack and so it support only until 3-dimension matrix stack.
	 * @param groupIndex group index.
	 * @param classIndex class index.
	 * @return output created from class index.
	 */
	NeuronValue[] createOutputByClass(int groupIndex, int classIndex) {
		Matrix OUTPUT = this.getCoreOutput();
		NeuronValue zero = OUTPUT instanceof MatrixStack ? ((MatrixStack)OUTPUT).get().get(0, 0).zero() : OUTPUT.get(0, 0).zero(); //Fixing date: 2026.06.10.
		NeuronValue unit = zero.unit();
		int outputCount = getNumberOfOutputs(groupIndex);
		NeuronValue[] output = new NeuronValue[outputCount];
		Map<Integer, int[]> outputClassMap = this.outputClassMaps.get(groupIndex);
		int unitCount = 0;
		for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
			int[] realClassIndices = outputClassMap.get(outputIndex);
			if (Arrays.binarySearch(realClassIndices, classIndex) >= 0) {
				output[outputIndex] = unit;
				unitCount++;
			}
			else
				output[outputIndex] = zero;
		}
		
		//Normalization.
		if (unitCount > 0) {
			for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
				output[outputIndex] = output[outputIndex].divide(unitCount);
			}
		}
		
		return output;
	}
	
	
	/**
	 * Extracting classes of given output.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 * @param output specified output.
	 * @return classes of given output.
	 */
	int[] extractClass(Matrix output) {
		int groups = getNumberOfGroups();
		int[] foundClasses = new int[groups]; 
		for (int group = 0; group < groups; group++) {
			double[] weights = weightsOfOutput(output, group);

			int classCount = getNumberOfClasses(group);
			int foundClass = 0;
			for (int classIndex = 1; classIndex < classCount; classIndex++) {
				if (weights[classIndex] > weights[foundClass]) foundClass = classIndex;
			}
			foundClasses[group] = foundClass;
		}
		
		return foundClasses;
	}

	
	/**
	 * Getting weights of specified output.
	 * @param output specified output.
	 * @return weights of specified output.
	 */
	double[] weightsOfOutput(Matrix output, int groupIndex) {
		NeuronValue[] values = getOutput(output, groupIndex);
		return weightsOfOutput(values);
	}

	
	/**
	 * Getting weights of specified output.
	 * @param output specified output.
	 * @return weights of specified output.
	 */
	static double[] weightsOfOutput(NeuronValue[] output) {
		double[] weights = new double[output.length];
		for (int i = 0; i < weights.length; i++) weights[i] = output[i].mean();
		return weights;
	}

	
	/**
	 * Getting class index of label.
	 * @param groupIndex group index.
	 * @param label specified label.
	 * @return class index of label.
	 */
	int classOf(int groupIndex, int label) {
		Map<Integer, Label> classMap = this.classMaps.get(groupIndex);
		Set<Integer> classIndices = classMap.keySet();
		for (int classIndex : classIndices) {
			Label labelObject = classMap.get(classIndex);
			if (labelObject.labelId == label) return classIndex;
		}
		return -1;
	}
	
	
	/**
	 * Getting class index of raster.
	 * @param groupIndex group index.
	 * @param raster raster.
	 * @return class index of raster.
	 */
	int classOf(int groupIndex, Raster raster) {
		RasterProperty rp = raster != null ? raster.getProperty() : null;
		if (rp == null) return -1;
		if (Math.min(getNumberOfGroups(), rp.getLabelCount()) <= 0) return -1;
		int labelId = rp.getLabelId(groupIndex);
		return classOf(groupIndex, labelId);
	}
	
	
	/**
	 * Getting label of class index.
	 * @param groupIndex group index.
	 * @param classIndex class index.
	 * @return label of class index.
	 */
	Label labelOf(int groupIndex, int classIndex) {
		Map<Integer, Label> classMap = this.classMaps.get(groupIndex);
		return classMap.containsKey(classIndex) ? classMap.get(classIndex) : null;
	}


	/**
	 * Create raster from matrix.
	 * @param matrix matrix.
	 * @return raster.
	 */
	Raster toRaster(Matrix matrix) {return getInputLayer().toRaster(matrix);}

	
	/**
	 * Converting raster to matrix.
	 * @param raster raster.
	 * @return matrix.
	 */
	Matrix toMatrix(Raster raster) {return getInputLayer().toMatrix(raster);}

	
	/**
	 * Converting raster to record.
	 * @param raster raster.
	 * @param training training flag.
	 * @return record.
	 */
	Record toRecord(Raster raster, boolean training) {
		if (raster == null) return null;
		if (!isLabeled()) return new Record(toMatrix(raster));
		
		RasterProperty rp = raster.getProperty();
		if (rp == null) return null;
		int groupCount = getNumberOfGroups();
		int n = Math.min(groupCount, rp.getLabelCount());
		if (n <= 0) return null;
		
		int[] classIndices = new int[groupCount];
		Arrays.fill(classIndices, 0);
		boolean valid = false;
		for (int group = 0; group < n; group++) {
			int labelId = rp.getLabelId(group);
			int classIndex = classOf(group, labelId);
			if (classIndex >= 0) {
				classIndices[group] = classIndex;
				valid = valid || true;
			}
		}
		if (!valid) return null;

		Matrix input = toMatrix(raster);
		Matrix output = createOutputByClass(classIndices, training);
		if (input == null || output == null) return null;
		return new Record(input, output);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @param nCoreClasses core classes size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size middleSize, Size outputSize, Size nCoreClasses) {
		reset0();
		if (nCoreClasses != null) {
			if (!configClassInfo(nCoreClasses)) return false;
		}
		
		if (middleSize != null) {
			if (!initialize(inputSize, middleSize, outputSize)) return false;
		}
		else {
			if (!initialize(inputSize, outputSize)) return false;
		}
		return true;
	}


	/**
	 * Initializing VGG.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @param labelGroups list of label groups which can be null.
	 * @return true if initialization is true.
	 */
	protected boolean initialize(Size inputSize, Size middleSize, Size outputSize, List<List<Label>> labelGroups) {
		if (labelGroups == null || labelGroups.size() == 0) return initialize(inputSize, middleSize, outputSize, (Size)null);
		
		//Removing empty labels and sorting labels.
		List<List<Label>> tempLabelGroups = Util.newList(labelGroups.size());
		tempLabelGroups.addAll(labelGroups);
		labelGroups.clear();
		for (List<Label> labels : tempLabelGroups) {
			if (labels.size() == 0) continue;
			Label.sort(labels, true);
			labelGroups.add(labels);
		}
		if (labelGroups.size() == 0) return false;

		//Adjusting the group label list so that every its element has the same class count.
		int minClassCount = labelGroups.get(0).size();
		for (List<Label> labels : labelGroups) {
			if (minClassCount > labels.size()) minClassCount = labels.size();
		}
		for (List<Label> labels : labelGroups) {
			if (labels.size() > minClassCount) {
				List<Label> temp = Util.newList(0);
				temp.addAll(labels);
				List<Label> sub = temp.subList(0, minClassCount);
				labels.clear();
				labels.addAll(sub);
			}
		}

		//Initializing matrix network.
		int groupCount = labelGroups.size();
		Size nCoreClasses = paramIsByColumn() ? new Size(groupCount, minClassCount, 1, 1) : new Size(minClassCount, groupCount, 1, 1);
		boolean initialized = middleSize != null ? initialize(inputSize, middleSize, outputSize, nCoreClasses) : initialize(inputSize, outputSize, (Size)null, nCoreClasses);
		if (!initialized) return false;
		
		//Main task: setting up class maps.
		this.classMaps.clear();
		for (int group = 0; group < groupCount; group++) {
			Map<Integer, Label> classMap = Util.newMap(0);
			int classCount = getNumberOfClasses(group);
			List<Label> labels = labelGroups.get(group);
			for (int classNumber = 0; classNumber < classCount; classNumber++) {
				Label label = classNumber < labels.size() ? labels.get(classNumber) : labels.get(labels.size()-1);
				if (label != null) classMap.put(classNumber, label);
			}
			if (classMap.size() > 0) this.classMaps.add(classMap);
		}
		return this.classMaps.size() > 0;
	}

	
	/**
	 * Initializing VGG.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @param labels array of labels which can be null.
	 * @return true if initialization is true.
	 */
	public boolean initialize(Size inputSize, Size middleSize, Size outputSize, Label...labels) {
		if (labels == null || labels.length == 0) return initialize(inputSize, middleSize, outputSize, (Size)null);
		List<List<Label>> labelGroups = Util.newList(1);
		labelGroups.add(Arrays.asList(labels));
		return initialize(inputSize, middleSize, outputSize, labelGroups);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @param nCoreClasses core classes.
	 * @return true if initialization is true.
	 */
	public boolean initialize(Iterable<Raster> sample, Size middleSize, Size outputSize, Size nCoreClasses) {
		//Getting minimum count of labels.
		int labelCount = -1;
		List<Raster> train = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			if (rp == null || rp.getLabelId() < 0) continue;
			int lc = rp.getLabelCount();
			if (lc <= 0) continue;
			
			train.add(raster);
			if (labelCount == -1)
				labelCount = lc;
			else
				labelCount = labelCount > lc ? lc : labelCount; //Label count is the minimum.
		}
		if (train.size() == 0 || labelCount <= 0) {
			Size averageSize = RasterAssoc.getAverageSize(train);
			return initialize(averageSize, middleSize, outputSize, nCoreClasses != null ? nCoreClasses : paramGetDefaultClassSize());
		}

		//Initializing list of label groups.
		if (train.size() == 0 || labelCount <= 0) return false;
		List<List<Label>> labelGroups = Util.newList(labelCount);
		for (int label = 0; label < labelCount; label++) labelGroups.add(Util.newList(0));
		
		//Initializing labels.
		for (Raster raster : train) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			if (rp == null) continue;
			for (int i = 0; i < rp.getLabelCount(); i++) {
				Label label = rp.getLabel(i);
				if (label == null || label.labelId < 0 || i >= labelGroups.size()) continue;
				List<Label> labels = labelGroups.get(i);
				boolean found = false;
				for (Label lb : labels) {
					if (lb.labelId == label.labelId) {
						found = true;
						break;
					}
				}
				if (!found) labels.add(label);
			}
		}

		Size averageSize = RasterAssoc.getAverageSize(train);
		return initialize(averageSize, middleSize, outputSize, labelGroups);
	}

	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is true.
	 */
	public boolean initialize(Iterable<Raster> sample, Size middleSize, Size outputSize) {return initialize(sample, middleSize, outputSize, null);}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, Size outputSize) {return initialize(sample, outputSize, null);}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param outputSize output size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Iterable<Raster> sample, Size outputSize) {return initialize(sample, paramGetVGGMiddleSize(), outputSize);}

	
	/**
	 * Learning VGG from raster sample.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @param nCoreClasses core classes.
	 * @return true if initialization is true.
	 */
	public Error[] learnRaster(Iterable<Raster> sample, Size middleSize, Size outputSize, Size nCoreClasses) {
		reset();
		if (!initialize(sample, middleSize, outputSize, nCoreClasses)) return null;
		return learnRaster(sample);
	}

	
	/**
	 * Learning VGG from raster sample.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is true.
	 */
	public Error[] learnRaster(Iterable<Raster> sample, Size middleSize, Size outputSize) {
		reset();
		if (!initialize(sample, middleSize, outputSize)) return null;
		return learnRaster(sample);
	}
	
	
	/**
	 * Learning VGG from raster sample.
	 * @param sample sample.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is true.
	 */
	public Error[] learnRaster(Iterable<Raster> sample, Size outputSize) {
		reset();
		if (!initialize(sample, outputSize)) return null;
		return learnRaster(sample);
	}

	
	/**
	 * Learning VGG from raster sample.
	 * @param sample sample.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is true.
	 */
	public Error[] learnRasterWithImplicitMiddleSize(Iterable<Raster> sample, Size outputSize) {
		reset();
		if (!initializeWithImplicitMiddleSize(sample, outputSize)) return null;
		return learnRaster(sample);
	}

	
	/**
	 * Learning VGG from raster sample.
	 * @param sample raster sample.
	 * @return learning errors.
	 */
	public Error[] learnRaster(Iterable<Raster> sample) {
		int maxIteration = paramGetMaxIteration();
		double terminatedThreshold = paramGetTerminatedThreshold();
		double learningRate = paramGetLearningRate();
		int epochs = paramGetPseudoEpochs();

		Error[] outputErrors = null;
		Iterable<Raster> newsample = sample;
		for (int epoch = 0; epoch < epochs; epoch++) {
			double lr = calcLearningRate(learningRate, epoch+1);
			if (epoch > 0) {
				if (!(newsample instanceof List<?>)) newsample = net.ea.ann.core.Record.listOf(newsample);
				Collections.shuffle((List<?>)newsample);
			}
			outputErrors = learnRaster(newsample, lr, terminatedThreshold, maxIteration);
		}
		return outputErrors;
	}

	
	/**
	 * Learning matrix neural network.
	 * @param sample sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learning errors.
	 */
	private Error[] learnRaster(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		resetBackwardInfo(); //Fixing date: 2026.06.19.
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Error[] outputErrors = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Raster> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			outputErrors = learnRaster(subsample, lr);
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "vggext_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (outputErrors == null || outputErrors.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = Matrix.normMean(Error.errors(outputErrors));
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}//End while
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "vggext_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return outputErrors;
	}

	
	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param sample sample of rasters.
	 * @param learningRate learning rate.
	 * @return backward error.
	 */
	protected Error[] learnRaster(Iterable<Raster> sample, double learningRate) {
		Error[] outputErrors = null;
		if (trainers.size() == 0) {
			List<Error> outputErrorList = Util.newList(0);
			for (Raster raster : sample) {
				Record record = toRecord(raster, true);
				Matrix input = record.input(), realOutput = record.output();
				Error error = new Error((Matrix)null);
				Object[] params = defineOutputErrorParams(error, new TrainingFlag() {});
				Matrix output = evaluate0(input, params); //Please pay attention to this code line because of tracking errors.
				Matrix err = calcOutputError(output, realOutput, getOutputLayer(), params);
				if (err == null) continue;
				
				error.errorSet(err);
				Error[] errors = backward(new Error[] {error}, false, learningRate);
				assert (errors != null && errors.length == 1 && errors[0] != null);
				if (errors != null) outputErrorList.add(errors[0]);
//				outputErrorList.add(error);
			}
			outputErrors = outputErrorList.toArray(new Error[] {});
			if (outputErrors.length > 0) {
				updateParametersFromBackwardInfo(outputErrors.length, learningRate);
				outputErrors = backwardAgain(outputErrors, this, true, learningRate);
			}
//			outputErrors = backward(outputErrorList.toArray(new Error[] {}), this, true, learningRate);
			assert (outputErrors != null && outputErrors.length > 0);
		}
		else {
			Object[] params = defineOutputErrorParams(new TrainingFlag() {});
			for (RasterTaskTrainer trainer : trainers) {
				outputErrors = trainer.train(this, sample, false, learningRate, params);
			}
		}
		
		if (outputErrors != null) learnRasterVerify(sample);
		return outputErrors;
	}
	
	
	/**
	 * Verifying learning rasters.
	 * @param sample raster sample.
	 */
	protected void learnRasterVerify(Iterable<Raster> sample) {}
	
	
	/**
	 * Checking by-column flag.
	 * @return by-column flag.
	 */
	boolean paramIsByColumn() {
		if (config.containsKey(BYCOLUMN_FIELD))
			return config.getAsBoolean(BYCOLUMN_FIELD);
		else
			return BYCOLUMN_DEFAULT;
	}
	
	
	/**
	 * Setting by-column flag.
	 * @param byColumn by-column flag.
	 * @return this model.
	 */
	VGGExt paramSetByColumn(boolean byColumn) {
		config.put(BYCOLUMN_FIELD, byColumn);
		return this;
	}
	
	
	/**
	 * Getting the number of elements of a combination.
	 * @return the number of elements of a combination.
	 */
	int paramGetCombNumber() {
		int combNumber = config.getAsInt(COMB_NUMBER_FIELD);
		return combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
	}
	
	
	/**
	 * Setting the number of elements of a combination.
	 * @param combNumber the number of elements of a combination.
	 * @return this model.
	 */
	VGGExt paramSetCombNumber(int combNumber) {
		combNumber = combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
		config.put(COMB_NUMBER_FIELD, combNumber);
		return this;
	}
	
	
	/**
	 * Getting the number of classes.
	 * @return the number of classes.
	 */
	int paramGetClassNumber() {
		int classNumber = config.getAsInt(CLASS_NUMBER_FIELD);
		return classNumber < 1 ? CLASS_NUMBER_DEFAULT : classNumber;
	}
	
	
	/**
	 * Setting the number of classes.
	 * @param classNumber the number of classes.
	 * @return this model.
	 */
	VGGExt paramSetClassNumber(int classNumber) {
		classNumber = classNumber < 1 ? CLASS_NUMBER_DEFAULT : classNumber;
		config.put(CLASS_NUMBER_FIELD, classNumber);
		return this;
	}

	
	/**
	 * Getting default class size as output size.
	 * @return default class size as output size.
	 */
	Size paramGetDefaultClassSize() {
		int classNumber = paramGetClassNumber();
		return paramIsByColumn() ? new Size(1, classNumber, 1, 1) : new Size(classNumber, 1, 1, 1);
	}

	
	/**
	 * Checking label smoothing mode.
	 * If true, label smoothing is applied so that one-hot class vector become class probability in which the probability of certain class is large enough but smaller than 1, for example {@link #MAX_CLASS_PROB_DEFAULT}.
	 * @return label smoothing mode.
	 */
	boolean paramIsLabelSmooth() {
		if (config.containsKey(LABEL_SMOOTH_FIELD))
			return config.getAsBoolean(LABEL_SMOOTH_FIELD);
		else
			return LABEL_SMOOTH_DEFAULT;
	}
	
	
	/**
	 * Setting label smoothing mode.
	 * @param labelSmooth label smoothing mode.
	 * If true, label smoothing is applied so that one-hot class vector become class probability in which the probability of certain class is large enough but smaller than 1, for example {@link #MAX_CLASS_PROB_DEFAULT}.
	 * @return this model.
	 */
	VGGExt paramSetLabelSmooth(boolean labelSmooth) {
		config.put(LABEL_SMOOTH_FIELD, labelSmooth);
		return this;
	}

	
	/**
	 * Getting maximum class probability.
	 * @return maximum class probability.
	 */
	double paramGetMaxClassProb() {
		if (config.containsKey(MAX_CLASS_PROB_FIELD)) {
			double maxProb = config.getAsReal(MAX_CLASS_PROB_FIELD);
			return Math.max(0.5, Math.min(1, maxProb));
		}
		else
			return MAX_CLASS_PROB_DEFAULT;
	}
	
	
	/**
	 * Setting maximum class probability.
	 * @param maxProb maximum class probability.
	 * @return this model.
	 */
	VGGExt paramSetMaxClassProb(double maxProb) {
		maxProb = Math.max(0.5, Math.min(1, maxProb));
		config.put(MAX_CLASS_PROB_FIELD, maxProb);
		return this;
	}

	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param nCoreClasses core classes size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByCoreClasses(Size inputSize, Size middleSize, Size nCoreClasses) {
		reset0();
		if (nCoreClasses != null) {
			if (!configClassInfo(nCoreClasses)) return false;
		}

		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = getNumberOfGroups();
		Size outputCombSize = paramIsByColumn() ? new Size(groupCount, outputCount, 1) : new Size(outputCount, groupCount, 1);
		if (middleSize != null) {
			if (!initialize(inputSize, middleSize, outputCombSize)) return false;
		}
		else {
			if (!initialize(inputSize, outputCombSize)) return false;
		}
		
		Matrix output = getCoreOutput();
		if (paramIsByColumn()) {
			if (output.rows() != this.outputClassMaps.get(0).size() ||
				output.columns() != this.outputClassMaps.size()) return false;
		}
		else {
			if (output.rows() != this.outputClassMaps.size() ||
				output.columns() != this.outputClassMaps.get(0).size()) return false;
		}
		
		return true;
	}


	/**
	 * Initializing VGG.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param labelGroups list of label groups which can be null.
	 * @return true if initialization is true.
	 */
	protected boolean initializeByCoreClasses(Size inputSize, Size middleSize, List<List<Label>> labelGroups) {
		if (labelGroups == null || labelGroups.size() == 0) throw new IllegalArgumentException();
		
		//Removing empty labels and sorting labels.
		List<List<Label>> tempLabelGroups = Util.newList(labelGroups.size());
		tempLabelGroups.addAll(labelGroups);
		labelGroups.clear();
		for (List<Label> labels : tempLabelGroups) {
			if (labels.size() == 0) continue;
			Label.sort(labels, true);
			labelGroups.add(labels);
		}
		if (labelGroups.size() == 0) return false;

		//Adjusting the group label list so that every its element has the same class count.
		int minClassCount = labelGroups.get(0).size();
		for (List<Label> labels : labelGroups) {
			if (minClassCount > labels.size()) minClassCount = labels.size();
		}
		for (List<Label> labels : labelGroups) {
			if (labels.size() > minClassCount) {
				List<Label> temp = Util.newList(0);
				temp.addAll(labels);
				List<Label> sub = temp.subList(0, minClassCount);
				labels.clear();
				labels.addAll(sub);
			}
		}

		//Initializing matrix network.
		int groupCount = labelGroups.size();
		Size nCoreClasses = paramIsByColumn() ? new Size(groupCount, minClassCount, 1, 1) : new Size(minClassCount, groupCount, 1, 1);
		if (!initializeByCoreClasses(inputSize, middleSize, nCoreClasses)) return false;
		
		//Main task: setting up class maps.
		this.classMaps.clear();
		for (int group = 0; group < groupCount; group++) {
			Map<Integer, Label> classMap = Util.newMap(0);
			int classCount = getNumberOfClasses(group);
			List<Label> labels = labelGroups.get(group);
			for (int classNumber = 0; classNumber < classCount; classNumber++) {
				Label label = classNumber < labels.size() ? labels.get(classNumber) : labels.get(labels.size()-1);
				if (label != null) classMap.put(classNumber, label);
			}
			if (classMap.size() > 0) this.classMaps.add(classMap);
		}
		return this.classMaps.size() > 0;
	}

	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @return true if initialization is true.
	 */
	public boolean initializeByCoreClasses(Iterable<Raster> sample, Size middleSize) {
		RasterSampleInfo rsi = RasterSampleInfo.extract(sample);
		if (rsi == null) return false;
		return initializeByCoreClasses(rsi.averageSize, middleSize, rsi.labelGroups);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param sample sample.
	 * @return true if initialization is successful.
	 */
	public boolean initializeByCoreClassesWithImplicitMiddleSize(Iterable<Raster> sample) {
		return initializeByCoreClasses(sample, paramGetVGGMiddleSize());
	}

	
	/**
	 * Learning VGG from raster sample.
	 * @param sample sample.
	 * @param middleSize middle size.
	 * @return true if initialization is true.
	 */
	public Error[] learnRasterByCoreClasses(Iterable<Raster> sample, Size middleSize) {
		reset();
		if (!initializeByCoreClasses(sample, middleSize)) return null;
		return learnRaster(sample);
	}

	
	/**
	 * Learning VGG from raster sample.
	 * @param sample sample.
	 * @return true if initialization is true.
	 */
	public Error[] learnRasterByCoreClassesWithImplicitMiddleSize(Iterable<Raster> sample) {
		reset();
		if (!initializeByCoreClassesWithImplicitMiddleSize(sample)) return null;
		return learnRaster(sample);
	}


}



/**
 * This class represents raster sample information.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class RasterSampleInfo implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Average size.
	 */
	public Size averageSize = new Size();
	
	
	/**
	 * List of label groups.
	 */
	List<List<Label>> labelGroups = Util.newList(0);
	
	
	/**
	 * Default constructor.
	 */
	public RasterSampleInfo() {}
	
	
	/**
	 * Extracting raster sample information.
	 * @param sample raster sample.
	 * @return raster sample information.
	 */
	public static RasterSampleInfo extract(Iterable<Raster> sample) {
		//Getting minimum count of labels.
		int labelCount = -1;
		List<Raster> train = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			if (rp == null || rp.getLabelId() < 0) continue;
			int lc = rp.getLabelCount();
			if (lc <= 0) continue;
			
			train.add(raster);
			if (labelCount == -1)
				labelCount = lc;
			else
				labelCount = labelCount > lc ? lc : labelCount; //Label count is the minimum.
		}
		if (train.size() == 0 || labelCount <= 0) return null;
		
		List<List<Label>> labelGroups = Util.newList(labelCount);
		for (int label = 0; label < labelCount; label++) labelGroups.add(Util.newList(0));
		
		//Initializing labels.
		for (Raster raster : train) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			if (rp == null) continue;
			for (int i = 0; i < rp.getLabelCount(); i++) {
				Label label = rp.getLabel(i);
				if (label == null || label.labelId < 0 || i >= labelGroups.size()) continue;
				List<Label> labels = labelGroups.get(i);
				boolean found = false;
				for (Label lb : labels) {
					if (lb.labelId == label.labelId) {
						found = true;
						break;
					}
				}
				if (!found) labels.add(label);
			}
		}

		//Removing empty labels and sorting labels.
		List<List<Label>> tempLabelGroups = Util.newList(labelGroups.size());
		tempLabelGroups.addAll(labelGroups);
		labelGroups.clear();
		for (List<Label> labels : tempLabelGroups) {
			if (labels.size() == 0) continue;
			Label.sort(labels, true);
			labelGroups.add(labels);
		}
		if (labelGroups.size() == 0) return null;

		//Adjusting the group label list so that every its element has the same class count.
		int minClassCount = labelGroups.get(0).size();
		for (List<Label> labels : labelGroups) {
			if (minClassCount > labels.size()) minClassCount = labels.size();
		}
		for (List<Label> labels : labelGroups) {
			if (labels.size() > minClassCount) {
				List<Label> temp = Util.newList(0);
				temp.addAll(labels);
				List<Label> sub = temp.subList(0, minClassCount);
				labels.clear();
				labels.addAll(sub);
			}
		}

		Size averageSize = RasterAssoc.getAverageSize(train);
		
		RasterSampleInfo rsi = new RasterSampleInfo();
		rsi.labelGroups = labelGroups;
		rsi.averageSize = averageSize;
		return rsi;
	}
	
	
}



/**
 * This class represents combination generator.
 * @author Someone on internet (http://www.merriampark.com/comb.htm)
 * @version 1.0
 *
 */
class CombinationGenerator implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Combination index array.
	 */
	private int[] a;
	
	/**
	 * Number of elements.
	 */
	private int n;
	
	
	/**
	 * Number of elements of a combination.
	 */
	private int r;
	
	
	/**
	 * Remaining number of combinations.
	 */
	private BigInteger numLeft;
	
	
	/**
	 * Total number of combinations.
	 */
	private BigInteger total;
	
	
	/**
	 * Constructor with number of elements and number of a combination.
	 * @param n number of elements.
	 * @param r number of a combination.
	 */
	public CombinationGenerator (int n, int r) {
		if (r > n) throw new IllegalArgumentException ();
		if (n < 1) throw new IllegalArgumentException ();

		this.n = n;
		this.r = r;
		a = new int[r];
		BigInteger nFact = getFactorial (n);
		BigInteger rFact = getFactorial (r);
		BigInteger nminusrFact = getFactorial (n - r);
		total = nFact.divide (rFact.multiply (nminusrFact));
		reset ();
	}
	
	
	/**
	 * Resetting.
	 */
	public void reset () {
		for (int i = 0; i < a.length; i++) {
			a[i] = i;
		}
		numLeft = new BigInteger (total.toString ());
	}

	
	/**
	 * Getting remaining number of combinations.
	 * @return remaining number of combinations.
	 */
	public BigInteger getNumLeft () {
		return numLeft;
	}


	/**
	 * Checking whether there more combinations.
	 * @return whether there more combinations.
	 */
	public boolean hasMore () {
		return numLeft.compareTo (BigInteger.ZERO) == 1;
	}

	
	/**
	 * Getting total number of combinations.
	 * @return total number of combinations.
	 */
	public BigInteger getTotal () {
		return total;
	}

	
	/**
	 * Computing factorial.
	 * @param n specified number.
	 * @return factorial of specified number.
	 */
	private static BigInteger getFactorial (int n) {
		BigInteger fact = BigInteger.ONE;
		for (int i = n; i > 1; i--) {
			fact = fact.multiply (new BigInteger (Integer.toString (i)));
		}
		return fact;
	}

	
	/**
	 * Generate next combination (algorithm from Rosen p. 286).
	 * @return next combination
	 */
	public int[] getNext () {
		if (numLeft.equals (total)) {
			numLeft = numLeft.subtract (BigInteger.ONE);
			return a;
		}

		int i = r - 1;
		while (a[i] == n - r + i) {
			i--;
		}
		a[i] = a[i] + 1;
		for (int j = i + 1; j < r; j++) {
			a[j] = a[i] + j - i;
		}

		numLeft = numLeft.subtract (BigInteger.ONE);
		return a;
	}
	
	
}


