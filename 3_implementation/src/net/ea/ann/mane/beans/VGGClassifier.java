/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.util.Arrays;
import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.Softmax;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.LikelihoodGradient;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.Record;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterProperty.Label;
import net.ea.ann.raster.RasterWrapperProperty;
import net.ea.ann.raster.Size;
import net.hudup.core.logistic.NextUpdate;

/**
 * This class implements Proxy-NCA (Proxy-Neighborhood Component Analysis) algorithm for deep metric learning, supporting both classification and clustering.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class VGGClassifier extends VGGExt {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for base line field.
	 */
	public static final String BASELINE_FIELD = "classifier_baseline";
	
	
	/**
	 * Default value for base line field.
	 */
	public static final boolean BASELINE_DEFAULT = false;

	
	/**
	 * Softmax flag.
	 */
	private static boolean SOFTMAX = true;
	
	
	/**
	 * Baseline.
	 */
	Matrix baseline = null;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGGClassifier(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(BASELINE_FIELD, BASELINE_DEFAULT);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public VGGClassifier(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public VGGClassifier(int neuronChannel, Function activateRef) {this(neuronChannel, activateRef, null, null);}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public VGGClassifier(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	@Override
	public void reset() {
		super.reset();
		reset0();
	}

	
	/**
	 * Resetting some properties.
	 */
	private void reset0() {this.baseline = null;}
	

	@Override
	protected boolean initialize(Size inputSize, Size middleSize, Size outputSize) {
		reset0();
		return super.initialize(inputSize, middleSize, outputSize);
	}


	/**
	 * Classifying rasters.
	 * @param sample raster sample.
	 * @return classified rasters.
	 */
	public List<Raster> classifyRaster(Iterable<Raster> sample) {
		if (!isLabeled()) return Util.newList(0);
		
		List<Raster> results = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			Matrix output = evaluate(raster);
			if (output == null) continue;
			assert (output == getCoreOutput());
			
			int groupCount = getNumberOfGroups();
			if (groupCount <= 0) continue;
			Label[] labels = new Label[groupCount];
			int[] classIndices = extractClass(output);
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
	protected Matrix calcOutputError(Matrix output, Matrix realOutput, MatrixLayerAbstract outputLayer, Object... params) {
		if (SOFTMAX) {
			Matrix error = paramIsByColumn() ? LikelihoodGradient.entropyGradientByColumn(output, realOutput, params) : LikelihoodGradient.entropyGradientByRow(output, realOutput, params);
			if (error == null) return error;
			
			if (outputLayer == null) return error;
			Matrix input = outputLayer.getInput();
			Function activateRef = outputLayer.getOutputActivateRef();
			Matrix derivative = input != null && activateRef != null ? input.derivativeWise(activateRef) : null;
			return derivative != null ? derivative.multiplyWise(error) : error;
		}
		else
			return super.calcOutputError(output, realOutput, outputLayer, params);
	}


	@Override
	double[] weightsOfOutput(Matrix output, int groupIndex) {
		if (this.baseline == null) return super.weightsOfOutput(output, groupIndex);
		
		Matrix[] OUTPUT = MatrixUtil.split(output);
		Matrix[] BASELINE = MatrixUtil.split(this.baseline);
		if (OUTPUT.length != BASELINE.length) throw new IllegalArgumentException();
		NeuronValue[] means = new NeuronValue[paramIsByColumn() ? OUTPUT[0].rows() : OUTPUT[0].columns()];
		for (int outputIndex = 0; outputIndex < means.length; outputIndex++) means[outputIndex] = OUTPUT[0].get(0, 0).zero();
		
		for (int d = 0; d < OUTPUT.length; d++) {
			NeuronValue[] values = getOutput(OUTPUT[d], groupIndex);
			values = SOFTMAX ? Softmax.softmax(values) : values;
			if (means.length != values.length) throw new IllegalArgumentException();
			
			for (int outputIndex = 0; outputIndex < values.length; outputIndex++) {
				NeuronValue base = paramIsByColumn() ? BASELINE[d].get(outputIndex, groupIndex) : BASELINE[d].get(groupIndex, outputIndex);
				//Following code lines are important due to apply baseline into determining class.
				NeuronValue sim = values[outputIndex].subtract(base);
				means[outputIndex] = means[outputIndex].add(sim);
			}
		}
		
		for (int outputIndex = 0; outputIndex < means.length; outputIndex++) means[outputIndex] = means[outputIndex].divide(OUTPUT.length);
		return weightsOfOutput(means);
	}

	
	@Override
	protected void learnRasterVerify(Iterable<Raster> sample) {
		super.learnRasterVerify(sample);
		this.baseline = null;
		if (!paramIsBaseline()) return;
		
		this.baseline = calcBaseline(sample);
	}


	/**
	 * Calculating baseline.
	 * In current version, this method does not support recursion of matrix stack and so it supports only until 3-dimension matrix stack.
	 * @param sample sample.
	 * @return baseline.
	 */
	@NextUpdate
	Matrix calcBaseline(Iterable<Raster> sample) {
		Matrix OUTPUT0 = getOutput();
		Matrix[] baselines = OUTPUT0 instanceof MatrixStack ? new Matrix[((MatrixStack)OUTPUT0).depth()] : new Matrix[1];
		Matrix[] countBaselines = OUTPUT0 instanceof MatrixStack ? new Matrix[((MatrixStack)OUTPUT0).depth()] : new Matrix[1];
		Matrix output0 = OUTPUT0 instanceof MatrixStack ? ((MatrixStack)OUTPUT0).get() : OUTPUT0;
		for (int d = 0; d < baselines.length; d++) {
			baselines[d] = output0.create(new Size(output0.columns(), output0.rows()));
			MatrixUtil.fill(baselines[d], 0);
			countBaselines[d] = baselines[d].create(new Size(baselines[d].columns(), baselines[d].rows()));
			MatrixUtil.fill(countBaselines[d], 0);
		}
		
		int combNumber = paramGetCombNumber();
		int groups = getNumberOfGroups();
		for (Raster raster : sample) {
			if (raster == null) continue;
			Record record = toRecord(raster, false);
			if (record == null) continue;
			Matrix OUTPUT = evaluate(raster);
			if (OUTPUT == null) continue;
			Matrix REALOUTPUT = record.output();
			if (REALOUTPUT == null) continue;
			
			Matrix[] outputs = null, realOutputs = null;
			if (OUTPUT instanceof MatrixStack && REALOUTPUT instanceof MatrixStack) {
				outputs = ((MatrixStack)OUTPUT).matrices();
				realOutputs = ((MatrixStack)REALOUTPUT).matrices();
			}
			else if (!(OUTPUT instanceof MatrixStack) && !(REALOUTPUT instanceof MatrixStack)) {
				outputs = new Matrix[] {OUTPUT};
				realOutputs = new Matrix[] {REALOUTPUT};
			}
			else
				throw new IllegalArgumentException();
			if (outputs.length != realOutputs.length) throw new IllegalArgumentException();
			
			for (int d = 0; d < outputs.length; d++) {
				for (int group = 0; group < groups; group++) {
					Matrix output = outputs[d], realOutput = realOutputs[d];
					
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
					outputOne = SOFTMAX ? Softmax.softmax(outputOne) : outputOne;
					for (int index = 0; index < indicator.length; index++) {
						if (!indicator[index]) continue;
						NeuronValue unit = countBaselines[d].get(0, 0).unit();
						if (paramIsByColumn()) {
							NeuronValue value = baselines[d].get(index, group).add(outputOne[index]);
							baselines[d].set(index, group, value);
							NeuronValue c = countBaselines[d].get(index, group).add(unit);
							countBaselines[d].set(index, group, c);
						}
						else {
							NeuronValue value = baselines[d].get(group, index).add(outputOne[index]);
							baselines[d].set(group, index, value);
							NeuronValue c = countBaselines[d].get(group, index).add(unit);
							countBaselines[d].set(group, index, c);
						}
					}
				} //End group.
			} //End depth.
		} //End sample.

		//Mean of base lines.
		for (int d = 0; d < baselines.length; d++) {
			for (int row = 0; row < baselines[d].rows(); row++) {
				for (int column = 0; column < baselines[d].columns(); column++) {
					NeuronValue value = baselines[d].get(row, column);
					NeuronValue c = countBaselines[d].get(row, column);
					if (c.canInvertWise())
						value = value.divide(c);
					else if (paramIsNorm())
						value = value.unit();
					else {
						value = value.valueOf(Float.MAX_VALUE); //Improving this code line later for non-normalized case.
						System.out.println("Improving this code line later for non-normalized case.");
					}
					baselines[d].set(row, column, value);
				}
			}
		}
		
		return MatrixUtil.join(baselines);
	}

	
	/**
	 * Checking baseline mode.
	 * @return baseline mode.
	 */
	boolean paramIsBaseline() {
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
	VGGClassifier paramSetBaseline(boolean baseline) {
		config.put(BASELINE_FIELD, baseline);
		return this;
	}


}
