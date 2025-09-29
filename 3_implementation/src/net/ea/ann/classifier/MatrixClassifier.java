/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.awt.Dimension;
import java.io.Serializable;
import java.math.BigInteger;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorWeighted;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterProperty.Label;
import net.ea.ann.raster.RasterWrapperProperty;
import net.ea.ann.raster.Size;

/**
 * This class is default implementation of classifier within context of matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixClassifier extends MatrixNetworkImpl implements Classifier {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default value for by-column flag.
	 */
	public final static String BYCOLUMN_FIELD = "mac_bycolumn";

	
	/**
	 * Default value for by-column flag.
	 */
	public final static boolean BYCOLUMN_DEFAULT = TaskTrainerLossEntropy.BYCOLUMN;

	
	/**
	 * Field of the number elements of a combination.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	public static final String COMB_NUMBER_FIELD = "mac_comb_number";
	
	
	/**
	 * Default value for the field of the number elements of a combination.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	public static final int COMB_NUMBER_DEFAULT = GeneratorWeighted.COMB_NUMBER_DEFAULT;
	
	
	/**
	 * Field for filter ignorance.
	 */
	public static final String FILTER_IGNORE_FIELD = "mac_filter_ignore";
	
	
	/**
	 * Default value for filter ignorance.
	 */
	public static final boolean FILTER_IGNORE_DEFAULT = true;

	
	
	
	/**
	 * Field for filter stride.
	 */
	public static final String FILTER_STRIDE_FIELD = "mac_filter_stride";
	
	
	/**
	 * Default value for filter stride.
	 */
	public static final int FILTER_STRIDE_DEFAULT = MatrixNetworkImpl.BASE_DEFAULT;

	
	/**
	 * Field for depth.
	 */
	public static final String DEPTH_FIELD = "mac_depth";
	
	
	/**
	 * Default value for depth.
	 */
	public static final int DEPTH_DEFAULT = MatrixNetworkImpl.DEPTH_DEFAULT;

	
	/**
	 * Field for dual mode.
	 */
	public static final String DUAL_FIELD = "mac_dual";
	
	
	/**
	 * Default value for dual mode.
	 */
	public static final boolean DUAL_DEFAULT = false;

	
	/**
	 * List of outputs-classes maps. For an outputs-classes map whose each element is a subtask which is a combination given classes.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	List<Map<Integer, int[]>> outputClassMaps = Util.newList(0);
	
	
	/**
	 * List of classes-outputs maps. For a classes-outputs map whose each element is a class pointer to the subtask which is a combination given classes.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	List<Map<Integer, int[]>> classOutputMaps = Util.newList(0);

	
	/**
	 * List of class-label maps.
	 */
	List<Map<Integer, Label>> classMaps = Util.newList(0);

	
	/**
	 * Output mean.
	 */
	Matrix outputMean = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixClassifier(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(BYCOLUMN_FIELD, BYCOLUMN_DEFAULT);
		config.put(COMB_NUMBER_FIELD, COMB_NUMBER_DEFAULT);
		config.put(FILTER_IGNORE_FIELD, FILTER_IGNORE_DEFAULT);
		config.put(FILTER_STRIDE_FIELD, FILTER_STRIDE_DEFAULT);
		config.put(DEPTH_FIELD, DEPTH_DEFAULT);
		config.put(DUAL_FIELD, DUAL_DEFAULT);
		
		setTrainer(new TaskTrainerLossEntropy());
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixClassifier(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixClassifier(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixClassifier(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	@Override
	public void reset() {
		super.reset();
		outputClassMaps.clear();
		classOutputMaps.clear();
		classMaps.clear();
		outputMean = null;
	}


	/*
	 * Note, nCoreClasses2 the number of rows and columns of core classes.
	 */
	@Override
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension nCoreClasses2, int depth2) {
		if (!configClassInfo(nCoreClasses2)) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = paramIsByColumn() ? nCoreClasses2.width : nCoreClasses2.height;
		Dimension outputSize2 = paramIsByColumn() ? new Dimension(groupCount, outputCount) : new Dimension(outputCount, groupCount);
		boolean initialized = false;
		if (paramIsFilterIgnore())
			initialized = super.initialize(inputSize1, outputSize2, (Filter2D)null, depth1, false, null, 0);
		else
			initialized = super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2);
		if (!initialized) return false;
		if (this.outputClassMaps.size() != this.classOutputMaps.size()) return false;
		
		Matrix output = getOutput();
		if (paramIsByColumn()) {
			return output.rows() == this.outputClassMaps.get(0).size() &&
				output.columns() == this.outputClassMaps.size();
		}
		else {
			return output.rows() == this.outputClassMaps.size() &&
				output.columns() == this.outputClassMaps.get(0).size();
		}
	}

	
	/*
	 * Note, nCoreClasses2 the number of rows and columns of core classes.
	 */
	@Override
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Dimension filterStride1, int depth1, boolean dual1, Dimension nCoreClasses2, int depth2) {
		Filter2D filter1 = defaultFilter(filterStride1);
		return initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2);
	}
	
	
	/**
	 * Configure class information.
	 * @param nCoreClasses the number of rows and columns of core classes.
	 * @return true if configuration is successful.
	 */
	protected boolean configClassInfo(Dimension nCoreClasses) {
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

	
	@Override
	public NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException {
		return learnRaster(sample);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		List<Matrix[]> newsample = prelearn(sample);
		Matrix[] errors = learn(newsample);
		NeuronValue[] errorArray = null;
		for (Matrix error : errors) {
			NeuronValue[] values = Matrix.extractValues(error);
			errorArray = errorArray == null ? values : NeuronValue.concatArray(errorArray, values);
		}
		return errorArray;
	}


	/**
	 * Getting the number of output groups.
	 * @return the number of output groups.
	 */
	int getNumberOfGroups() {
		return this.outputClassMaps.size();
	}
	
	
	/**
	 * Getting the number of outputs given group.
	 * @param groupIndex given group index.
	 * @return the number of outputs given group.
	 */
	int getNumberOfOutputs(int groupIndex) {
		return this.outputClassMaps.get(groupIndex).size();
	}
	

	/**
	 * Getting the number of classes given group.
	 * @param groupIndex given group index.
	 * @return the number of classes given group.
	 */
	int getNumberOfClasses(int groupIndex) {
		return this.classOutputMaps.get(groupIndex).size();
	}
	
	
	/**
	 * Getting output group.
	 * @param output output.
	 * @param groupIndex group index.
	 * @return output group.
	 */
	NeuronValue[] getOutput(Matrix output, int groupIndex) {
		Matrix group = paramIsByColumn() ? output.getColumn(groupIndex) : output.getRow(groupIndex);
		return Matrix.extractValues(group);
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
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 * @param classIndices core class indices of group, whose each element is core class index of a group.
	 * @return output created from core class indices.
	 */
	public Matrix createOutputByClass(int[] classIndices) {
		int groupCount = getNumberOfGroups();
		if (groupCount <= 0) return null;
		int outputCount = getNumberOfOutputs(0);
		
		int rows = paramIsByColumn() ? outputCount : groupCount;
		int columns = paramIsByColumn() ? groupCount : outputCount;
		Matrix output = this.getOutput().create(rows, columns);
		if (classIndices == null || classIndices.length == 0) return output;
		
		NeuronValue zero = output.get(0, 0).zero();
		NeuronValue unit = zero.unit();
		if (paramIsByColumn()) {
			int groups = Math.min(columns, classIndices.length);
			for (int group = 0; group < groups; group++) {
				int classIndex = classIndices[group];
				Map<Integer, int[]> outputClassMap = outputClassMaps.get(group);
				int unitCount = 0;
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					int[] realClassIndices = outputClassMap.get(outputIndex);
					if (Arrays.binarySearch(realClassIndices, classIndex) >= 0) {
						output.set(outputIndex, group, unit);
						unitCount++;
					}
					else
						output.set(outputIndex, group, zero);
				}
				
				//Normalization.
				if (unitCount > 0) {
					for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
						NeuronValue value = output.get(outputIndex, group).divide(unitCount);
						output.set(outputIndex, group, value);
					}
				}
			}
		}
		else {
			int groups = Math.min(rows, classIndices.length);
			for (int group = 0; group < groups; group++) {
				int classIndex = classIndices[group];
				Map<Integer, int[]> outputClassMap = outputClassMaps.get(group);
				int unitCount = 0;
				for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
					int[] realClassIndices = outputClassMap.get(outputIndex);
					if (Arrays.binarySearch(realClassIndices, classIndex) >= 0) {
						output.set(group, outputIndex, unit);
						unitCount++;
					}
					else
						output.set(group, outputIndex, zero);
				}
				
				//Normalization.
				if (unitCount > 0) {
					for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
						NeuronValue value = output.get(group, outputIndex).divide(unitCount);
						output.set(group, outputIndex, value);
					}
				}
			}
		}

		return output;
	}

	
	/**
	 * Creating output from class index.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 * @param groupIndex group index.
	 * @param classIndex class index.
	 * @return output created from class index.
	 */
	public NeuronValue[] createOutputByClass(int groupIndex, int classIndex) {
		NeuronValue zero = getOutput().get(0, 0).zero();
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
	 * Extracting class of output.
	 * @return classes of output.
	 */
	public int[] extractClass() {
		return extractClass(getOutput());
	}

	
	/**
	 * Extracting class of output, given group.
	 * @param groupIndex specified group.
	 * @return classes of output.
	 */
	public int extractClass(int groupIndex) {
		return extractClass()[groupIndex];
	}
	
	
	/**
	 * Extracting classes of given output.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 * @param output specified output.
	 * @return classes of given output.
	 */
	private int[] extractClass(Matrix output) {
		int groups = getNumberOfGroups();
		int[] foundClasses = new int[groups]; 
		for (int group = 0; group < groups; group++) {
			double[] weights = weightsOfOutput(output, group);

			int classCount = getNumberOfClasses(group);
			if (this.outputMean != null) {
				for (int classIndex = 0; classIndex < classCount; classIndex++) {
					NeuronValue mean = paramIsByColumn() ? this.outputMean.get(classIndex, group) : this.outputMean.get(group, classIndex);
					weights[classIndex] = Math.abs(weights[classIndex] - mean.mean()); //This line is important due to set mean as baseline for determining class.
				}
			}
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
	private double[] weightsOfOutput(Matrix output, int groupIndex) {
		NeuronValue[] values = getOutput(output, groupIndex);
		return weightsOfOutput(values);
	}

	
	/**
	 * Getting weights of specified output.
	 * @param output specified output.
	 * @return weights of specified output.
	 */
	private static double[] weightsOfOutput(NeuronValue[] output) {
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
	 * Pre-processing for learning.
	 * @param sample
	 * @return new sample.
	 */
	List<Matrix[]> prelearn(Iterable<Raster> sample) {
		reset();
		
		//Getting minimum count of labels.
		int labelCount = -1;
		List<Raster> train = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			if (rp == null) continue;
			int labelId = rp.getLabelId();
			if (labelId < 0) continue;
			int lc = rp.getLabelCount();
			if (lc <= 0) continue;
			
			train.add(raster);
			if (labelCount == -1)
				labelCount = lc;
			else
				labelCount = labelCount < lc ? lc : labelCount;
		}
		
		//Initializing list of label groups.
		if (train.size() == 0 || labelCount <= 0) return Util.newList(0);
		List<List<Label>> labelGroups = Util.newList(labelCount);
		for (int label = 0; label < labelCount; label++) labelGroups.add(Util.newList(0));
		
		//Initializing labels.
		for (Raster raster : train) {
			RasterProperty rp = raster.getProperty();
			for (int i = 0; i < rp.getLabelCount(); i++) {
				Label label = rp.getLabel(i);
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
		if (labelGroups.size() == 0) return Util.newList(0);

		//Adjusting the group label list so that its size is equal to group count.
		int groupCount = labelGroups.size();
		if (labelGroups.size() > groupCount) labelGroups = labelGroups.subList(0, groupCount);
		if (labelGroups.size() < groupCount) {
			int n = groupCount - labelGroups.size();
			List<Label> labels = labelGroups.get(labelGroups.size()-1);
			for (int i = 0; i < n; i++) labelGroups.add(labels);
		}

		//Initializing matrix network.
		int maxClassCount = labelGroups.get(0).size();
		for (List<Label> labels : labelGroups) {
			if (maxClassCount < labels.size()) maxClassCount = labels.size();
		}
		Size size = RasterAssoc.getAverageSize(train);
		Dimension inputSize = new Dimension(size.width, size.height);
		Dimension filterStride = new Dimension(paramGetFilterStride(), paramGetFilterStride());
		int depth = paramGetDepth();
		boolean dual = paramIsDual();
		Dimension nCoreClasses = paramIsByColumn() ? new Dimension(groupCount, maxClassCount) : new Dimension(maxClassCount, groupCount);
		if (!initialize(inputSize, null, filterStride, depth, dual, nCoreClasses, depth))
			return Util.newList(0);

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
		if (this.classMaps.size() == 0) return Util.newList(0);
		
		//Initializing new sample.
		List<Matrix[]> newsample = Util.newList(0);
		MatrixLayerAbstract inputLayer = getInputLayer();
		for (Raster raster : train) {
			int[] classIndices = new int[getNumberOfGroups()];
			Arrays.fill(classIndices, 0);
			RasterProperty rp = raster.getProperty();
			for (int i = 0; i < rp.getLabelCount(); i++) {
				int labelId = rp.getLabelId(i);
				int classIndex = classOf(i, labelId);
				if (classIndex >= 0) classIndices[i] = classIndex;
			}

			Matrix input = inputLayer.toMatrix(raster);
			Matrix output = createOutputByClass(classIndices);
			if (input == null || output == null) continue;
			newsample.add(new Matrix[] {input, output});
		}
		return newsample;
	}
	
	
	@Override
	public Matrix[] learn(Iterable<Matrix[]> inouts) throws RemoteException {
		Matrix[] errors = super.learn(inouts);
		learnVerify(inouts);
		return errors;
	}

	
	/**
	 * Verifying learning.
	 * @param inouts learning sample.
	 */
	public void learnVerify(Iterable<Matrix[]> inouts) {
		this.outputMean = null;
		List<Matrix> outputList = Util.newList(0);
		for (Matrix[] inout : inouts) {
			Matrix output = evaluate(inout[0], new Object[] {});
			if (output != null) outputList.add(output);
		}
		if (outputList.size() > 0) this.outputMean = Matrix.mean(outputList.toArray(new Matrix[] {}));
	}

	
	@Override
	public List<Raster> classify(Iterable<Raster> sample) throws RemoteException {
		List<Raster> results = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			try {
				evaluate(raster);
			} catch (Throwable e) {Util.trace(e);}
			
			int groupCount = getNumberOfGroups();
			if (groupCount <= 0) continue;
			Label[] labels = new Label[groupCount];
			int[] classIndices = extractClass();
			for (int group = 0; group < groupCount; group++) {
				Label label = labelOf(group, classIndices[group]);
				labels[group] = label != null ? label : new Label();
			}
			
			RasterProperty rp = raster.getProperty().shallowDuplicate();
			rp.setLabels(labels);
			RasterWrapperProperty rw = new RasterWrapperProperty(raster);
			rw.setProperty(rp);
			results.add(rw);
		}
		return results;
	}

	
	@Override
	public int getNeuronChannel() throws RemoteException {
		return neuronChannel;
	}

	
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
	 * @return this matrix classifier.
	 */
	MatrixClassifier paramSetByColumn(boolean byColumn) {
		config.put(BYCOLUMN_FIELD, byColumn);
		return this;
	}
	
	
	/**
	 * Getting the number elements of a combination.
	 * @return the number elements of a combination.
	 */
	int paramGetCombNumber() {
		int combNumber = config.getAsInt(COMB_NUMBER_FIELD);
		return combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
	}
	
	
	/**
	 * Setting the number elements of a combination.
	 * @param combNumber the number elements of a combination.
	 * @return this matrix classifier.
	 */
	MatrixClassifier paramSetCombNumber(int combNumber) {
		combNumber = combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
		config.put(COMB_NUMBER_FIELD, combNumber);
		return this;
	}


	/**
	 * Checking filter ignorance.
	 * @return filter ignorance.
	 */
	boolean paramIsFilterIgnore() {
		if (config.containsKey(FILTER_IGNORE_FIELD))
			return config.getAsBoolean(FILTER_IGNORE_FIELD);
		else
			return FILTER_IGNORE_DEFAULT;
	}
	
	
	/**
	 * Setting filter ignorance.
	 * @param filterIgnore filter ignorance.
	 * @return this matrix classifier.
	 */
	MatrixClassifier paramFilterIgnore(boolean filterIgnore) {
		config.put(FILTER_IGNORE_FIELD, filterIgnore);
		return this;
	}

	
	
	
	/**
	 * Getting filter stride.
	 * @return filter stride.
	 */
	int paramGetFilterStride() {
		int filterStride = config.getAsInt(FILTER_STRIDE_FIELD);
		return filterStride < 1 ? FILTER_STRIDE_DEFAULT : filterStride;
	}
	
	
	/**
	 * Setting filter stride.
	 * @param filterStride filter stride.
	 * @return this matrix classifier.
	 */
	MatrixClassifier paramSetFilterStride(int filterStride) {
		filterStride = filterStride < 1 ? FILTER_STRIDE_DEFAULT : filterStride;
		config.put(FILTER_STRIDE_FIELD, filterStride);
		return this;
	}

	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	int paramGetDepth() {
		int depth = config.getAsInt(DEPTH_FIELD);
		return depth < 1 ? DEPTH_DEFAULT : depth;
	}
	
	
	/**
	 * Setting depth.
	 * @param depth depth.
	 * @return this matrix classifier.
	 */
	MatrixClassifier paramSetDepth(int depth) {
		depth = depth < 1 ? DEPTH_DEFAULT : depth;
		config.put(DEPTH_FIELD, depth);
		return this;
	}

	
	/**
	 * Checking dual mode.
	 * @return dual mode.
	 */
	boolean paramIsDual() {
		if (config.containsKey(DUAL_FIELD))
			return config.getAsBoolean(DUAL_FIELD);
		else
			return DUAL_DEFAULT;
	}
	
	
	/**
	 * Setting dual mode.
	 * @param dual dual mode.
	 * @return this matrix classifier.
	 */
	MatrixClassifier paramSetDual(boolean dual) {
		config.put(DUAL_FIELD, dual);
		return this;
	}

	
	/**
	 * Creating classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static MatrixClassifier create(int neuronChannel, boolean isNorm) {
		Function activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		return new MatrixClassifier(neuronChannel, activateRef, contentActivateRef, null);
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