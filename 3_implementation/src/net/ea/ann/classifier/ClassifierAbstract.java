/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.io.Serializable;
import java.math.BigInteger;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Softmax;
import net.ea.ann.core.generator.GeneratorWeighted;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterProperty.Label;
import net.ea.ann.raster.RasterWrapperProperty;
import net.ea.ann.raster.Size;

/**
 * This class represents abstract implementation of classifier.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ClassifierAbstract extends NetworkAbstract implements Classifier {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default value for by-column flag.
	 */
	public final static String BYCOLUMN_FIELD = "classifier_bycolumn";

	
	/**
	 * Default value for by-column flag.
	 */
	public final static boolean BYCOLUMN_DEFAULT = TaskTrainerLossEntropy.BYCOLUMN;

	
	/**
	 * Field of the number elements of a combination.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	public static final String COMB_NUMBER_FIELD = "classifier_comb_number";
	
	
	/**
	 * Default value for the field of the number elements of a combination.
	 * Please see <a href="https://cusaas.com/blog/neural-classification">https://cusaas.com/blog/neural-classification</a> or /newtech-research/data-mining-analyzing/classification/neural-network/DataClassificationWithNeuralNetworks-Cusaas-2023.01.12.pdf.
	 */
	public static final int COMB_NUMBER_DEFAULT = GeneratorWeighted.COMB_NUMBER_DEFAULT;
	
	
	/**
	 * Field for convolutional network.
	 */
	public static final String CONV_FIELD = "classifier_conv";
	
	
	/**
	 * Default value for convolutional network.
	 */
	public static final boolean CONV_DEFAULT = false;

	
	/**
	 * Field for filter size.
	 */
	public static final String FILTER_SIZE_FIELD = "classifier_filter_size";
	
	
	/**
	 * Default value for filter size.
	 */
	public static final int FILTER_SIZE_DEFAULT = MatrixNetworkImpl.BASE_DEFAULT;

	
	/**
	 * Field for depth.
	 */
	public static final String DEPTH_FIELD = "classifier_depth";
	
	
	/**
	 * Default value for depth.
	 */
	public static final int DEPTH_DEFAULT = MatrixNetworkImpl.DEPTH_DEFAULT;

	
	/**
	 * Field for dual mode.
	 */
	public static final String DUAL_FIELD = "classifier_dual";
	
	
	/**
	 * Default value for dual mode.
	 */
	public static final boolean DUAL_DEFAULT = false;

	
	/**
	 * Field for base line field.
	 */
	public static final String BASELINE_FIELD = "classifier_baseline";
	
	
	/**
	 * Default value for base line field.
	 */
	public static final boolean BASELINE_DEFAULT = true;

	
	/**
	 * Field for adjusting.
	 */
	public static final String ADJUST_FIELD = "classifier_adjust";
	
	
	/**
	 * Default value for adjusting.
	 */
	public static final boolean ADJUST_DEFAULT = false;


	/**
	 * Field for sample weight.
	 */
	public static final String SAMPLE_WEIGHT_FIELD = "classifier_sample_weight";
	
	
	/**
	 * Default value for sample weight.
	 */
	public static final boolean SAMPLE_WEIGHT_DEFAULT = false;

	
	/**
	 * Field for cross-entropy trainer.
	 */
	public static final String ENTROPY_TRAINER_FIELD = "classifier_entropy_trainer";

	
	/**
	 * Default value for cross-entropy trainer.
	 */
	public static final boolean ENTROPY_TRAINER_DEFAULT = true;
	
	
	/**
	 * Field for creating adjuster mode.
	 */
	static final String CREATE_ADJUSTER_FIELD = "classifier_create_adjuster";
	
	
	/**
	 * Default value for creating adjuster mode.
	 */
	static final boolean CREATE_ADJUSTER_DEFAULT = false;
	
	
	/**
	 * Field for stack size.
	 */
	public static final String STACK_SIZE_FIELD = "classifier_stack_size";
	
	
	/**
	 * Default value for depth.
	 */
	public static final int STACK_SIZE_DEFAULT = 1; //MatrixNetworkImpl.DEPTH_DEFAULT;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
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
	 * Baseline.
	 */
	Matrix baseline = null;
	
	
	/**
	 * Adjusting baseline.
	 */
	Matrix adjustline = null;

	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	protected ClassifierAbstract(int neuronChannel, Id idRef) {
		super(idRef);
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);

		config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, MatrixNetworkAbstract.VECTORIZED_DEFAULT);
		config.put(BYCOLUMN_FIELD, BYCOLUMN_DEFAULT);
		config.put(COMB_NUMBER_FIELD, COMB_NUMBER_DEFAULT);
		config.put(CONV_FIELD, CONV_DEFAULT);
		config.put(FILTER_SIZE_FIELD, FILTER_SIZE_DEFAULT);
		config.put(DEPTH_FIELD, DEPTH_DEFAULT);
		config.put(DUAL_FIELD, DUAL_DEFAULT);
		config.put(BASELINE_FIELD, BASELINE_DEFAULT);
		config.put(ADJUST_FIELD, ADJUST_DEFAULT);
		config.put(MatrixNetworkImpl.MIDSIZE_FIELD, MatrixNetworkImpl.MIDSIZE_DEFAULT);
		config.put(SAMPLE_WEIGHT_FIELD, SAMPLE_WEIGHT_DEFAULT);
		config.put(ENTROPY_TRAINER_FIELD, ENTROPY_TRAINER_DEFAULT);
		config.put(CREATE_ADJUSTER_FIELD, CREATE_ADJUSTER_DEFAULT);
		config.put(STACK_SIZE_FIELD, STACK_SIZE_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected ClassifierAbstract(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	/**
	 * Resetting classifier.
	 */
	public void reset() {
		outputClassMaps.clear();
		classOutputMaps.clear();
		classMaps.clear();
		baseline = null;
		adjustline = null;
	}

	
	/**
	 * Updating configuration.
	 */
	void updateConfig() { }

	
	/**
	 * Updating configuration.
	 * @param config configuration.
	 */
	void updateConfig(NetworkConfig config) {
		try {
			this.config.putAll(config);
		} catch (Throwable e) {Util.trace(e);}
		updateConfig();
	}

	
	/**
	 * Initializing classifier.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 2.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		updateConfig();
		this.baseline = null;
		this.adjustline = null;
		
		Size nCoreClasses2 = outputSize2;
		if (nCoreClasses2 == null) nCoreClasses2 = outputSize1 != null ? outputSize1 : inputSize1; 
		return configClassInfo(nCoreClasses2);
	}

	
	/**
	 * Initializing matrix classifier (MAC).
	 * @param labelGroups list of label groups.
	 * @param averageSize average size.
	 * @return true if initialization is true.
	 */
	public boolean initialize(List<List<Label>> labelGroups, Size averageSize) {
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
		Size inputSize = new Size(averageSize.width, averageSize.height, 1); //Flattening later in matrix neural network.
		FilterSpec filter = new FilterSpec(paramGetFilterSize(), paramGetFilterSize());
		int depth = paramGetDepth();
		depth = depth > 0 ? depth : 0;
		Size nCoreClasses = paramIsByColumn() ? new Size(groupCount, minClassCount, 1) : new Size(minClassCount, groupCount, 1);
		if (paramIsConv()) {
			int halfDepth = depth > 1 ? depth/2 : depth;
			if (paramIsDual()) {
				if (!initialize(inputSize, nCoreClasses, filter, halfDepth, true, null, 0))
					return false;
			}
			else {
				int depth1 = depth - halfDepth, depth2 = halfDepth;
				depth1 = depth1 == 0 && depth2 > 0 ? 1 : depth1;
				if (paramGetMiddleSize() <= 0) {
					Size midSize = new Size(inputSize);
					midSize.depth = paramGetStackSize();
					if (!initialize(inputSize, midSize, filter, depth1, false, nCoreClasses, depth2)) return false;
				}
				else {
					int ratio = Math.min(inputSize.height, inputSize.width) / paramGetMiddleSize();
					if (ratio <= 1) {
						Size midSize = new Size(inputSize);
						midSize.depth = paramGetStackSize();
						if (!initialize(inputSize, midSize, filter, depth1, false, nCoreClasses, depth2)) return false;
					}
					else {
						int height = inputSize.height/ratio, width = inputSize.width/ratio;
						if (!initialize(inputSize, new Size(width, height, paramGetStackSize()), filter, depth1, false, nCoreClasses, depth2)) return false;
					}
				} //paramGetMiddleSize()
			} //paramIsDual()
		}
		else {
			if (!initialize(inputSize, nCoreClasses, (FilterSpec)null, depth, false, null, 0)) return false;
		} //paramIsConv()

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
	 * Initializing matrix classifier (MAC).
	 * @param sample sample.
	 * @return true if initialization is true.
	 */
	private boolean initialize(Iterable<Raster> sample) {
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
				labelCount = labelCount > lc ? lc : labelCount;
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
		return initialize(labelGroups, averageSize);
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
	 * Getting output.
	 * @return output.
	 */
	protected abstract Matrix getOutput();
	
	
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
	Matrix createOutputByClass(int[] classIndices) {
		int groupCount = getNumberOfGroups();
		if (groupCount <= 0) return null;
		int outputCount = getNumberOfOutputs(0);
		
		int rows = paramIsByColumn() ? outputCount : groupCount;
		int columns = paramIsByColumn() ? groupCount : outputCount;
		Matrix output = this.getOutput().create(new Size(columns, rows));
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
	NeuronValue[] createOutputByClass(int groupIndex, int classIndex) {
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
		values = paramIsEntropyTrainer() ? Softmax.softmax(values) : values;
		if (this.baseline == null) return weightsOfOutput(values);
		
		for (int classIndex = 0; classIndex < values.length; classIndex++) {
			NeuronValue base = paramIsByColumn() ? this.baseline.get(classIndex, groupIndex) : this.baseline.get(groupIndex, classIndex);
			//Following code lines are important due to apply baseline into determining class.
			NeuronValue sim = values[classIndex].subtract(base);
			if (this.adjustline != null) {
				NeuronValue adjust = paramIsByColumn() ? this.adjustline.get(classIndex, groupIndex) : this.adjustline.get(groupIndex, classIndex);
				sim = sim.multiply(adjust);
			}
			values[classIndex] = sim;
		}
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
	private int classOf(int groupIndex, int label) {
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
	private Label labelOf(int groupIndex, int classIndex) {
		Map<Integer, Label> classMap = this.classMaps.get(groupIndex);
		return classMap.containsKey(classIndex) ? classMap.get(classIndex) : null;
	}


	/**
	 * Converting raster to matrix.
	 * @param raster raster.
	 * @return matrix.
	 */
	protected abstract Matrix toMatrix(Raster raster);
	
	
	/**
	 * Evaluating input.
	 * @param input input.
	 * @param params additional parameters.
	 * @return evaluated output.
	 */
	protected abstract Matrix evaluate(Matrix input, Object...params);
	
	
	@Override
	public NeuronValue[] learnRasterOneByOne(Iterable<Raster> sample) throws RemoteException {
		return learnRaster(sample);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		List<Record> newsample = prelearn(sample);
		Error[] errors = learn(newsample);
		learnVerify(newsample);
		
		if (paramGetAdjuster() != null) {
			paramGetAdjuster().paramSetInclude(this);
			List<Record> adjustSample = Util.newList(0);
			for (Record record : newsample) {
				Matrix output = evaluate(record.input());
				if (output != null) adjustSample.add(new Record(output, record.output()));
			}
			errors = adjustSample.size() > 0 ? paramGetAdjuster().learn(adjustSample) : errors;
		}
		
		NeuronValue[] errorArray = null;
		for (Error error : errors) {
			NeuronValue[] values = Matrix.extractValues(error.error());
			errorArray = errorArray == null ? values : NeuronValue.concatArray(errorArray, values);
		}
		return errorArray;
	}

	
	/**
	 * Pre-processing for learning.
	 * @param sample
	 * @return new sample.
	 */
	List<Record> prelearn(Iterable<Raster> sample) {
		reset();
		
		//Initializing matrix classifier (MAC).
		if (!initialize(sample)) return Util.newList(0);
		
		//Initializing new sample.
		List<Record> newsample = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			if (rp == null) continue;
			int groupCount = getNumberOfGroups();
			int n = Math.min(groupCount, rp.getLabelCount());
			if (n <= 0) continue;
			
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
			if (!valid)
				continue;

			Matrix input = toMatrix(raster);
			Matrix output = createOutputByClass(classIndices);
			if (input == null || output == null) continue;
			newsample.add(new Record(input, output));
		}
		return newsample;
	}
	
	
	/**
	 * Learning classifier.
	 * @param sample sample.
	 * @return learning errors.
	 */
	protected abstract Error[] learn(Iterable<Record> sample);

	
	/**
	 * Verifying learning.
	 * @param sample learning sample.
	 */
	public void learnVerify(Iterable<Record> sample) {
		this.baseline = null;
		this.adjustline = null;
		if (!paramIsBaseline()) return;

		if (paramIsAdjust()) {
			Matrix[] lines = calcBaselineAdjust(sample, paramGetAdjuster());
			this.baseline = lines[0];
			this.adjustline = lines[1];
		}
		else
			this.baseline = calcBaseline(sample);
	}

	
	/**
	 * Calculating baseline.
	 * @param sample sample.
	 * @return baseline.
	 */
	Matrix calcBaseline(Iterable<Record> sample) {
		Matrix o = getOutput();
		Matrix baseline = o.create(new Size(o.columns(), o.rows()));
		Matrix.fill(baseline, 0);
		Matrix countBaseline = baseline.create(new Size(baseline.columns(), baseline.rows()));
		Matrix.fill(countBaseline, 0);
		
		int combNumber = paramGetCombNumber();
		int groups = getNumberOfGroups();
		for (Record record : sample) {
			Matrix output = evaluate(record.input());
			Matrix realOutput = record.output();
			for (int group = 0; group < groups; group++) {
				NeuronValue[] realOutputOne = getOutput(realOutput, group);
				double[] realOutputOneV = weightsOfOutput(realOutputOne);
				int maxIndex = 0;
				for (int i = 1; i < realOutputOneV.length; i++) {
					if (realOutputOneV[i] > realOutputOneV[maxIndex]) maxIndex = i;
				}
				
				boolean[] indicator = new boolean[realOutputOneV.length];
				Arrays.fill(indicator, false);
				indicator[maxIndex] = true;
				for (int i = 0; i < indicator.length; i++) {
					if (indicator[i] || combNumber == 1) continue;
					if (realOutputOneV[i] >= realOutputOneV[maxIndex] - Double.MIN_VALUE) indicator[i] = true;
				}
				
				NeuronValue[] outputOne = getOutput(output, group);
				outputOne = paramIsEntropyTrainer() ? Softmax.softmax(outputOne) : outputOne;
				for (int index = 0; index < indicator.length; index++) {
					if (!indicator[index]) continue;
					NeuronValue unit = countBaseline.get(0, 0).unit();
					if (paramIsByColumn()) {
						NeuronValue value = baseline.get(index, group).add(outputOne[index]);
						baseline.set(index, group, value);
						NeuronValue c = countBaseline.get(index, group).add(unit);
						countBaseline.set(index, group, c);
					}
					else {
						NeuronValue value = baseline.get(group, index).add(outputOne[index]);
						baseline.set(group, index, value);
						NeuronValue c = countBaseline.get(group, index).add(unit);
						countBaseline.set(group, index, c);
					}
				}
			}
		} //End sample.

		//Mean of base line.
		for (int row = 0; row < baseline.rows(); row++) {
			for (int column = 0; column < baseline.columns(); column++) {
				NeuronValue value = baseline.get(row, column);
				NeuronValue c = countBaseline.get(row, column);
				if (c.canInvertWise())
					value = value.divide(c);
				else if (paramIsNorm() || paramIsEntropyTrainer())
					value = value.unit();
				else
					value = value.valueOf(Float.MAX_VALUE); //Improving this code line later for non-normalized case.
				baseline.set(row, column, value);
			}
		}
		return baseline;
	}
	
	
//	/**
//	 * Calculating baseline.
//	 * @param matrices array of matrices.
//	 * @return baseline.
//	 */
//	@SuppressWarnings("unused")
//	@Deprecated
//	private Matrix calcBaselineByStat(Iterable<Record> sample) {
//		List<Matrix> outputList = Util.newList(0);
//		for (Record record : sample) {
//			Matrix output = evaluate(record.input());
//			if (output != null) outputList.add(output);
//		}
//		if (outputList.size() == 0) return null;
//
//		Matrix[] matrices = outputList.toArray(new Matrix[] {});
//		Matrix mean = Matrix.mean(matrices);
//		Matrix std = Matrix.std(matrices);
//		Matrix baseLineByMean = mean.subtract(std.multiply0(1.96));
//		Matrix baseLineByMin = Matrix.min(matrices);
//		return Matrix.max(baseLineByMean, baseLineByMin);
//	}
	
	
	/**
	 * Calculating adjusted line.
	 * @param sample sample.
	 * @param adjuster adjuster.
	 * @return adjusted line.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private Matrix calcAdjustline(Iterable<Record> sample, MatrixNetworkImpl adjuster) {
		Matrix o = getOutput();
		Matrix adjustline = o.create(new Size(o.columns(), o.rows()));
		Matrix.fill(adjustline, 0);
		Matrix countAdjustline = adjustline.create(new Size(adjustline.columns(), adjustline.rows()));
		Matrix.fill(countAdjustline, 0);

		for (Record record : sample) {
			Matrix output = evaluate(record.input());
			Matrix adjustOutput = null;
			try {
				adjustOutput = adjuster != null ? adjuster.evaluate(output) : output;
			} catch (Throwable e) {Util.trace(e);}

			int groups = getNumberOfGroups();
			for (int group = 0; group < groups; group++) {
				NeuronValue[] adjustOutputOne = getOutput(adjustOutput, group);
				adjustOutputOne = paramIsEntropyTrainer() ? Softmax.softmax(adjustOutputOne) : adjustOutputOne;
				for (int index = 0; index < adjustOutputOne.length; index++) {
					NeuronValue unit = countAdjustline.get(0, 0).unit();
					if (paramIsByColumn()) {
						NeuronValue value = adjustline.get(index, group).add(adjustOutputOne[index]);
						adjustline.set(index, group, value);
						NeuronValue c = countAdjustline.get(index, group).add(unit);
						countAdjustline.set(index, group, c);
					}
					else {
						NeuronValue value = adjustline.get(group, index).add(adjustOutputOne[index]);
						adjustline.set(group, index, value);
						NeuronValue c = countAdjustline.get(group, index).add(unit);
						countAdjustline.set(group, index, c);
					}
				}
			}
		} //End sample
		
		//Mean of adjusted line.
		for (int row = 0; row < adjustline.rows(); row++) {
			for (int column = 0; column < adjustline.columns(); column++) {
				NeuronValue value = adjustline.get(row, column);
				NeuronValue c = countAdjustline.get(row, column);
				if (c.canInvertWise())
					value = value.divide(c);
				else
					value = value.zero();
				adjustline.set(row, column, value);
			}
		}
		adjustline = paramIsByColumn() ? Softmax.softmaxByColumn(adjustline) : Softmax.softmaxByRow(adjustline);
		return adjustline;
	}
	

	/**
	 * Calculating baseline and adjusted line.
	 * @param sample sample.
	 * @param adjuster adjuster.
	 * @return baseline and adjusted line..
	 */
	Matrix[] calcBaselineAdjust(Iterable<Record> sample, MatrixNetworkImpl adjuster) {
		Matrix o = getOutput();
		Matrix baseline = o.create(new Size(o.columns(), o.rows()));
		Matrix.fill(baseline, 0);
		Matrix countBaseline = baseline.create(new Size(baseline.columns(), baseline.rows()));
		Matrix.fill(countBaseline, 0);
		
		Matrix adjustline = o.create(new Size(o.columns(), o.rows()));
		Matrix.fill(adjustline, 0);
		Matrix countAdjustline = adjustline.create(new Size(adjustline.columns(), adjustline.rows()));
		Matrix.fill(countAdjustline, 0);
		
		int combNumber = paramGetCombNumber();
		int groups = getNumberOfGroups();
		for (Record record : sample) {
			Matrix output = evaluate(record.input());
			Matrix realOutput = record.output();
			for (int group = 0; group < groups; group++) {
				NeuronValue[] realOutputOne = getOutput(realOutput, group);
				double[] realOutputOneV = weightsOfOutput(realOutputOne);
				int maxIndex = 0;
				for (int i = 1; i < realOutputOneV.length; i++) {
					if (realOutputOneV[i] > realOutputOneV[maxIndex]) maxIndex = i;
				}
				
				boolean[] indicator = new boolean[realOutputOneV.length];
				Arrays.fill(indicator, false);
				indicator[maxIndex] = true;
				for (int i = 0; i < indicator.length; i++) {
					if (indicator[i] || combNumber == 1) continue;
					if (realOutputOneV[i] >= realOutputOneV[maxIndex] - Double.MIN_VALUE) indicator[i] = true;
				}
				
				//Calculating base line.
				NeuronValue[] outputOne = getOutput(output, group);
				outputOne = paramIsEntropyTrainer() ? Softmax.softmax(outputOne) : outputOne;
				for (int index = 0; index < indicator.length; index++) {
					if (!indicator[index]) continue;
					NeuronValue unit = countBaseline.get(0, 0).unit();
					if (paramIsByColumn()) {
						NeuronValue value = baseline.get(index, group).add(outputOne[index]);
						baseline.set(index, group, value);
						NeuronValue c = countBaseline.get(index, group).add(unit);
						countBaseline.set(index, group, c);
					}
					else {
						NeuronValue value = baseline.get(group, index).add(outputOne[index]);
						baseline.set(group, index, value);
						NeuronValue c = countBaseline.get(group, index).add(unit);
						countBaseline.set(group, index, c);
					}
				}
				
				//Calculating adjusted line.
				Matrix adjustOutput = adjuster != null ? adjuster.evaluate0(output) : output;
				NeuronValue[] adjustOutputOne = null;
				if (adjustOutput == output)
					adjustOutputOne = outputOne;
				else {
					adjustOutputOne = getOutput(adjustOutput, group);
					adjustOutputOne = paramIsEntropyTrainer() ? Softmax.softmax(adjustOutputOne) : adjustOutputOne;
				}
				for (int index = 0; index < adjustOutputOne.length; index++) {
					NeuronValue unit = countAdjustline.get(0, 0).unit();
					if (paramIsByColumn()) {
						NeuronValue value = adjustline.get(index, group).add(adjustOutputOne[index]);
						adjustline.set(index, group, value);
						NeuronValue c = countAdjustline.get(index, group).add(unit);
						countAdjustline.set(index, group, c);
					}
					else {
						NeuronValue value = adjustline.get(group, index).add(adjustOutputOne[index]);
						adjustline.set(group, index, value);
						NeuronValue c = countAdjustline.get(group, index).add(unit);
						countAdjustline.set(group, index, c);
					}
				}
				
			}
		} //End sample.

		//Mean of base line.
		for (int row = 0; row < baseline.rows(); row++) {
			for (int column = 0; column < baseline.columns(); column++) {
				NeuronValue value = baseline.get(row, column);
				NeuronValue c = countBaseline.get(row, column);
				if (c.canInvertWise())
					value = value.divide(c);
				else if (paramIsNorm() || paramIsEntropyTrainer())
					value = value.unit();
				else {
					value = value.valueOf(Float.MAX_VALUE); //Improving this code line later for non-normalized case.
					System.out.println("Improving this code line later for non-normalized case.");
				}
				baseline.set(row, column, value);
			}
		}
		
		//Mean of adjusted line.
		for (int row = 0; row < adjustline.rows(); row++) {
			for (int column = 0; column < adjustline.columns(); column++) {
				NeuronValue value = adjustline.get(row, column);
				NeuronValue c = countAdjustline.get(row, column);
				if (c.canInvertWise())
					value = value.divide(c);
				else
					value = value.zero();
				adjustline.set(row, column, value);
			}
		}
		adjustline = paramIsByColumn() ? Softmax.softmaxByColumn(adjustline) : Softmax.softmaxByRow(adjustline);
		
		return new Matrix[] {baseline, adjustline};
	}

	
	@Override
	public List<Raster> classify(Iterable<Raster> sample) throws RemoteException {
		List<Raster> results = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			evaluate(toMatrix(raster));
			
			int groupCount = getNumberOfGroups();
			if (groupCount <= 0) continue;
			Label[] labels = new Label[groupCount];
			int[] classIndices = extractClass(getOutput());
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
	 * Getting adjuster.
	 * @return adjuster.
	 */
	MatrixNetworkImpl paramGetAdjuster() {return null;}
	
	
	/**
	 * Checking normalization mode.
	 * @return normalization mode in rang [0, 1].
	 */
	boolean paramIsNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Setting normalization mode.
	 * @param isNorm normalization mode in rang [0, 1]..
	 * @return this transformer.
	 */
	ClassifierAbstract paramSetNorm(boolean isNorm) {
		config.put(Raster.NORM_FIELD, isNorm);
		return this;
	}

	
	/**
	 * Checking whether the network data is vectorized.
	 * @return whether the network data is vectorized.
	 */
	boolean paramIsVectorized() {
		if (config.containsKey(MatrixNetworkAbstract.VECTORIZED_FIELD))
			return config.getAsBoolean(MatrixNetworkAbstract.VECTORIZED_FIELD);
		else
			return MatrixNetworkAbstract.VECTORIZED_DEFAULT;
	}


	/**
	 * Setting vectorization mode.
	 * @param vectorized vectorization mode.
	 * @return this network.
	 */
	ClassifierAbstract paramSetVectorized(boolean vectorized) {
		config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, vectorized);
		return this;
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
	 * @return this classifier.
	 */
	ClassifierAbstract paramSetByColumn(boolean byColumn) {
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
	 * @return this classifier.
	 */
	ClassifierAbstract paramSetCombNumber(int combNumber) {
		combNumber = combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
		config.put(COMB_NUMBER_FIELD, combNumber);
		return this;
	}


	/**
	 * Checking convolutional network mode.
	 * @return convolutional network mode.
	 */
	boolean paramIsConv() {
		if (config.containsKey(CONV_FIELD))
			return config.getAsBoolean(CONV_FIELD);
		else
			return CONV_DEFAULT;
	}
	
	
	/**
	 * Setting convolutional network mode.
	 * @param conv convolutional network mode.
	 * @return this classifier.
	 */
	ClassifierAbstract paramSetConv(boolean conv) {
		config.put(CONV_FIELD, conv);
		return this;
	}

	
	
	
	/**
	 * Getting filter size.
	 * @return filter size.
	 */
	int paramGetFilterSize() {
		int filterSize = config.getAsInt(FILTER_SIZE_FIELD);
		return filterSize < 1 ? FILTER_SIZE_DEFAULT : filterSize;
	}
	
	
	/**
	 * Setting filter size.
	 * @param filterSize filter size.
	 * @return this classifier.
	 */
	ClassifierAbstract paramSetFilterSize(int filterSize) {
		filterSize = filterSize < 1 ? FILTER_SIZE_DEFAULT : filterSize;
		config.put(FILTER_SIZE_FIELD, filterSize);
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
	 * @return this classifier.
	 */
	ClassifierAbstract paramSetDepth(int depth) {
		depth = depth < 1 ? DEPTH_DEFAULT : depth;
		config.put(DEPTH_FIELD, depth);
		return this;
	}

	
	/**
	 * Checking dual mode.
	 * @return dual mode.
	 */
	public boolean paramIsDual() {
		if (config.containsKey(DUAL_FIELD))
			return config.getAsBoolean(DUAL_FIELD);
		else
			return DUAL_DEFAULT;
	}
	
	
	/**
	 * Setting dual mode.
	 * @param dual dual mode.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetDual(boolean dual) {
		config.put(DUAL_FIELD, dual);
		return this;
	}

	
	/**
	 * Checking baseline mode.
	 * @return baseline mode.
	 */
	public boolean paramIsBaseline() {
		if (config.containsKey(BASELINE_FIELD))
			return config.getAsBoolean(BASELINE_FIELD);
		else
			return BASELINE_DEFAULT;
	}
	
	
	/**
	 * Setting baseline mode.
	 * @param baseline baseline mode.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetBaseline(boolean baseline) {
		config.put(BASELINE_FIELD, baseline);
		return this;
	}

	
	/**
	 * Checking adjust mode.
	 * @return baseline mode.
	 */
	public boolean paramIsAdjust() {
		if (config.containsKey(ADJUST_FIELD))
			return config.getAsBoolean(ADJUST_FIELD);
		else
			return ADJUST_DEFAULT;
	}
	
	
	/**
	 * Setting adjust mode.
	 * @param adjust adjust mode.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetAdjust(boolean adjust) {
		config.put(ADJUST_FIELD, adjust);
		return this;
	}


	/**
	 * Checking middle size.
	 * @return middle size.
	 */
	public int paramGetMiddleSize() {
		if (config.containsKey(MatrixNetworkImpl.MIDSIZE_FIELD))
			return config.getAsInt(MatrixNetworkImpl.MIDSIZE_FIELD);
		else
			return MatrixNetworkImpl.MIDSIZE_DEFAULT;
	}
	
	
	/**
	 * Setting middle size.
	 * @param minSize middle size.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetMiddleSize(int minSize) {
		config.put(MatrixNetworkImpl.MIDSIZE_FIELD, minSize);
		return this;
	}

	
	/**
	 * Checking sample weight mode.
	 * @return sample weight mode.
	 */
	public boolean paramIsSampleWeight() {
		if (config.containsKey(SAMPLE_WEIGHT_FIELD))
			return config.getAsBoolean(SAMPLE_WEIGHT_FIELD);
		else
			return SAMPLE_WEIGHT_DEFAULT;
	}
	
	
	/**
	 * Setting sample weight mode.
	 * @param sampleWeight sample weight mode.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetSampleWeight(boolean sampleWeight) {
		config.put(SAMPLE_WEIGHT_FIELD, sampleWeight );
		return this;
	}


	/**
	 * Checking cross-entropy trainer mode.
	 * @return cross-entropy trainer weight mode.
	 */
	public boolean paramIsEntropyTrainer() {
		if (config.containsKey(ENTROPY_TRAINER_FIELD))
			return config.getAsBoolean(ENTROPY_TRAINER_FIELD);
		else
			return ENTROPY_TRAINER_DEFAULT;
	}
	
	
	/**
	 * Setting cross-entropy trainer mode.
	 * @param entropyTrainer cross-entropy trainer.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetEntropyTrainer(boolean entropyTrainer) {
		config.put(ENTROPY_TRAINER_FIELD, entropyTrainer );
		return this;
	}


	/**
	 * Checking creating adjuster mode.
	 * @return creating adjuster mode.
	 */
	public boolean paramIsCreateAdjuster() {
		if (config.containsKey(CREATE_ADJUSTER_FIELD))
			return config.getAsBoolean(CREATE_ADJUSTER_FIELD);
		else
			return CREATE_ADJUSTER_DEFAULT;
	}
	
	
	/**
	 * Setting creating adjuster mode.
	 * @param createAdjuster creating adjuster mode.
	 * @return this classifier.
	 */
	public ClassifierAbstract paramSetCreateAdjuster(boolean createAdjuster) {
		config.put(CREATE_ADJUSTER_FIELD, createAdjuster);
		return this;
	}


	/**
	 * Getting stack size.
	 * @return stack size.
	 */
	int paramGetStackSize() {
		int stackSize = config.getAsInt(STACK_SIZE_FIELD);
		return stackSize < 1 ? STACK_SIZE_DEFAULT : stackSize;
	}
	
	
	/**
	 * Setting stack size.
	 * @param stackSize stack size.
	 * @return this classifier.
	 */
	ClassifierAbstract paramSetStackSize(int stackSize) {
		stackSize = stackSize < 1 ? STACK_SIZE_DEFAULT : stackSize;
		config.put(STACK_SIZE_FIELD, stackSize);
		return this;
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


