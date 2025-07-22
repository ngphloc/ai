/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.bp;

import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.core.Evaluator;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.Network;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.WeightedNeuron;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This class is abstract implementation of backpropagation algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class BackpropagatorAbstract extends BackpropagatorAbstract0 {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public BackpropagatorAbstract() {
		super();
	}


	/*
	 * Method can be overridden.
	 */
	@Override
	public NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		return super.updateWeightsBiases(bone, outputBatch, lastError, learningRate);
	}


	/*
	 * Method must be implemented.
	 */
	@Override
	protected abstract NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params);


	/*
	 * Method can be overridden.
	 */
	@Override
	public Map<Integer, NeuronValue[]> updateWeightsBiases(List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
		return super.updateWeightsBiases(bone, boneInput, boneOutput, learningRate);
	}

	
	/**
	 * Getting activation reference.
	 * @param neuron neuron.
	 * @param layer layer.
	 * @return activation reference.
	 */
	public static Function getActivateRef(NeuronStandard neuron, LayerStandard layer) {
		Function activateRef = neuron != null ? neuron.getActivateRef() : null;
		if (activateRef == null && layer != null) activateRef = layer.getActivateRef();
		if (activateRef == null && neuron != null) {
			LayerStandard neuronLayer = neuron.getLayer();
			if (neuronLayer != null) activateRef = neuronLayer.getActivateRef();
		}
		return activateRef;
	}
	
	
	/**
	 * Calculate error of output neuron.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * @param outputNeuron output neuron.
	 * @param realOutput real output. It can be null.
	 * @param outputLayer output layer. It can be null.
	 * @return error or loss of the output neuron.
	 */
	public static NeuronValue calcOutputErrorDefault(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer) {
		Function activateRef = getActivateRef(outputNeuron, outputLayer);
		NeuronValue neuronOutput = outputNeuron != null ? outputNeuron.getOutput() : null;
		NeuronValue neuronInput = NeuronStandard.getDerivativeInput(outputNeuron);
		return calcOutputErrorDefault(activateRef, realOutput, neuronOutput, neuronInput);
	}

	
	/**
	 * Calculate error of output neuron.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * @param outputNeuron output neuron.
	 * @param realOutput real output. It can be null.
	 * @return error or loss of the output neuron.
	 */
	public static NeuronValue calcOutputErrorDefault(NeuronStandard outputNeuron, NeuronValue realOutput) {
		return calcOutputErrorDefault(outputNeuron, realOutput, (LayerStandard)null);
	}

	
	/**
	 * Calculate error of output neuron
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * @param outputNeuron output neuron.
	 * @param realOutput real output. It can be null.
	 * @param neuronOutput neuron output. It can be null.
	 * @return error or loss of the output neuron.
	 */
	public static NeuronValue calcOutputErrorDefault(NeuronStandard outputNeuron, NeuronValue realOutput, NeuronValue neuronOutput) {
		Function activateRef = getActivateRef(outputNeuron, null);
		if (neuronOutput == null && outputNeuron != null) neuronOutput = outputNeuron.getOutput();
		NeuronValue neuronInput = NeuronStandard.getDerivativeInput(outputNeuron);
		return calcOutputErrorDefault(activateRef, realOutput, neuronOutput, neuronInput);
	}

	
	/**
	 * Calculate error of output neuron. This is utility method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * @param activateRef activation function. It can be null. If it is null, the derivative is 1.
	 * @param realOutput realistic output. It cannot be null.
	 * @param neuronOutput neuron output. It cannot be null.
	 * @param neuronInput neuron input. It can be null.
	 * @return error of output neuron.
	 */
	public static NeuronValue calcOutputErrorDefault(Function activateRef, NeuronValue realOutput, NeuronValue neuronOutput, NeuronValue neuronInput) {
		if (neuronOutput == null) return null;
		NeuronValue error = realOutput.subtract(neuronOutput);
		if (activateRef == null) return error;
		
		neuronInput = neuronInput != null ? neuronInput : neuronOutput;
		NeuronValue derivative = neuronInput.derivative(activateRef);
		return derivative != null ? error.multiplyDerivative(derivative) : error;
	}
	
	
}



/**
 * This class is abstract implementation of backpropagation algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class BackpropagatorAbstract0 implements Backpropagator {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public BackpropagatorAbstract0() {
		super();
	}

	
	/**
	 * Checking whether to learn bias.
	 * @return whether to learn bias.
	 */
	protected boolean isLearningBias() {
		return true;
	}
	
	
	@Override
	public NeuronValue[] updateWeightsBiases(Iterable<Record> sample, List<LayerStandard> bone, double learningRate, Evaluator evaluator) {
		if (bone.size() < 2) return null;
		LayerStandard outputLayer = bone.get(bone.size() - 1);
		NeuronValue[] meanOutputError = new NeuronValue[outputLayer.size()];
		NeuronValue zero = outputLayer.newNeuronValue().zero();
		for (int i = 0; i < meanOutputError.length; i++) meanOutputError[i] = zero;
		
		//Calculating mean output error.
		int n = 0;
		for (Record record : sample) {
			//Evaluating network.
			try {
				if (evaluator != null) evaluator.evaluate(record);
			} catch (Throwable e) {Util.trace(e);}

			//Calculating output error.
			NeuronValue[] outputError = calcOutputError(outputLayer, record.output);
			if (outputError == null) continue;
			
			for (int i = 0; i < meanOutputError.length; i++) meanOutputError[i] = meanOutputError[i].add(outputError[i]);
			n++;
		}
		
		if (n == 0) return null;
		
		//Updating weights and biases of bone.
		for (int i = 0; i < meanOutputError.length; i++) meanOutputError[i] = meanOutputError[i].divide(n);
		return updateWeightsBiases(bone, learningRate, meanOutputError);			
	}

	
	@Override
	public NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, NeuronValue[] realOutput, double learningRate) {
		if (realOutput == null) updateWeightsBiases(bone, (Iterable<NeuronValue[][]>)null, learningRate);

		List<NeuronValue[][]> outputBatch = Util.newList(1);
		outputBatch.add(new NeuronValue[][] {realOutput});
		return updateWeightsBiases(bone, outputBatch, learningRate);
	}
	
	
	/**
	 * Updating weights and biases.
	 * @param bone list of layers including input layer.
	 * @param outputBatch output batch of output layer.
	 * The first element is real output and the second element is neuron output. The second element may be null or removed.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	private NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, double learningRate) {
		return updateWeightsBiases(bone, outputBatch, null, learningRate);
	}
	
	
	/**
	 * Updating weights and biases.
	 * @param bone list of layers including input layer.
	 * @param learningRate learning rate.
	 * @param lastError output error which is optional parameter for batch learning.
	 * @return errors of output errors. Return null if errors occur.
	 */
	private NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, double learningRate, NeuronValue[] lastError) {
		return updateWeightsBiases(bone, null, lastError, learningRate);
	}
	
	
	@Override
	public NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		if (bone.size() < 2) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
		NeuronValue[] outputError = null;
		
		NeuronValue[] nextError = lastError;
		for (int i = bone.size()-1; i >= 1; i--) { //Browsing layers reversely from output layer down to first hidden layer.
			LayerStandard layer = bone.get(i);
			NeuronValue[] error = NeuronValue.makeArray(layer.size(), layer);
			
			for (int j = 0; j < layer.size(); j++) { //Browsing neurons of current layer.
				NeuronStandard neuron = layer.get(j);
				
				//Calculate error of current neuron at current layer.
				if (i == bone.size() - 1) {//Calculate error of last layer. This is most important for backpropagation algorithm.
					error[j] = nextError == null ? calcOutputError(layer, j, outputBatch) : nextError[j];
				}
				else {//Calculate error of of hidden layers.
					LayerStandard nextLayer = bone.get(i + 1);
					NeuronValue rsum = neuron.getOutput().zero();
					WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
					for (WeightedNeuron target : targets) {
						int index = nextLayer.indexOf(target.neuron);
						if (!checkIndex(index)) continue;
						rsum = rsum.add(nextError[index].multiply(target.weight.value));
					}
					
					NeuronValue derivative = calcDerivative(neuron);
					error[j] = derivative != null ? rsum.multiplyDerivative(derivative) : rsum;
				}
				
				//Update biases of current layer.
				if (isLearningBias()) {
					NeuronValue delta = error[j].multiply(learningRate);
					neuron.setBias(neuron.getBias().add(delta));
				}
			}
			
			//Update weights stored in previous layers.
			Set<LayerStandard> prevLayers = layer.getAllPrevLayers(); //Include virtual layer.
			if (!prevLayers.contains(bone.get(i-1))) prevLayers.add(bone.get(i-1));
			for (LayerStandard prevLayer : prevLayers) {
				if (prevLayer == null) continue;
				for (int j = 0; j < prevLayer.size(); j++) {
					NeuronStandard prevNeuron = prevLayer.get(j);
					NeuronValue prevOut = prevNeuron.getOutput();
					
					WeightedNeuron[] targets = prevNeuron.getNextNeurons(layer);
					if (targets.length == 0)
						targets = prevNeuron.getOutsideNextNeurons(layer).toArray(new WeightedNeuron[] {}); //Virtual layer.
					for (WeightedNeuron target : targets) {
						int index = layer.indexOf(target.neuron);
						if (!checkIndex(index)) continue;
						NeuronValue delta = error[index].multiply(prevOut).multiply(learningRate);
						Weight nw = target.weight;
						nw.value = nw.value.addValue(delta);
					}
				}
			}
			
			nextError = error;
			if (i == bone.size() - 1) outputError = error;
		}
		
		return outputError;
	}
	

	/**
	 * Calculating output error of output neuron at specified index over batch. Derived class can call or override this method.
	 * @param outputLayer output layer.
	 * @param outputNeuronIndex index of output neuron.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * @return output error of output neuron at specified index.
	 */
	protected NeuronValue calcOutputError(LayerStandard outputLayer, int outputNeuronIndex, Iterable<NeuronValue[][]> outputBatch) {
		NeuronStandard outputNeuron = outputLayer.get(outputNeuronIndex);
		//There are some cases that have no output (null output).
		if (outputBatch == null) return calcOutputError(outputNeuron, null, outputLayer, -1, null);

		int n = 0;
		NeuronValue errorMean = outputNeuron.getOutput().zero();
		for (NeuronValue[][] outputs : outputBatch) {
			NeuronValue[] realOutputs = (outputs != null && outputs.length > 0) ? outputs[0] : null;
			if (realOutputs != null) realOutputs = NeuronValue.adjustArray(realOutputs, outputLayer.size(), outputLayer);
			
			NeuronValue realOutput = realOutputs != null ? realOutputs[outputNeuronIndex] : null; //There are some cases that have no output (null output).
			NeuronValue error = null;
			if (outputs == null || outputs.length <= 1)
				error = calcOutputError(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs);
			else {
				NeuronValue[] neuronOutputs = outputs[1];
				if (neuronOutputs == null)
					error = calcOutputError(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs);
				else {
					neuronOutputs = NeuronValue.adjustArray(neuronOutputs, outputLayer.size(), outputLayer);
					error = BackpropagatorAbstract.calcOutputErrorDefault(outputNeuron, realOutput, (neuronOutputs != null ? neuronOutputs[outputNeuronIndex] : null));
				}
			}
			
			if (error == null) continue;
			
			errorMean = errorMean.add(error);
			n++;
		}
		
		if (n != 0 && n != 1) errorMean = errorMean.divide((double)n);
		return errorMean;
	}
	
	
	/**
	 * Calculating output error of output layer.
	 * @param outputLayer output layer.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * @return output error of output neuron at specified index.
	 */
	private NeuronValue[] calcOutputError(LayerStandard outputLayer, Iterable<NeuronValue[][]> outputBatch) {
		NeuronValue[] error = new NeuronValue[outputLayer.size()]; 
		for (int j = 0; j < outputLayer.size(); j++) {
			error[j] = calcOutputError(outputLayer, j, outputBatch);
		}
		return error;
	}
	
	
	/**
	 * Calculating output error of output layer.
	 * @param outputLayer output layer.
	 * @param realOutput realistic output of output layer.
	 * @return output error of output neuron at specified index.
	 */
	private NeuronValue[] calcOutputError(LayerStandard outputLayer, NeuronValue[] realOutput) {
		List<NeuronValue[][]> outputBatch = Util.newList(1);
		outputBatch.add(new NeuronValue[][] {realOutput});
		return calcOutputError(outputLayer, outputBatch);
	}
	

	/**
	 * Calculate error of output neuron. This method is the most important entrance of backpropagation algorithm.
	 * This error is the opposite of gradient of output error function for minimization or the gradient of output error function for maximization. 
	 * Similarly, this error is the gradient of output gain function for minimization or the opposite of gradient of output gain function for maximization.<br>
	 * <br>
	 * In general, this error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output.  
	 * @param outputNeuron output neuron. Output value of this neuron is retrieved by method {@link NeuronStandard#getOutput()}.
	 * @param realOutput real output. It can be null because this method is flexible.
	 * @param outputLayer output layer. It can be null because this method is flexible.
	 * @param outputNeuronIndex index of output neuron. It can be -1 because this method is flexible. This is optional parameter.
	 * @param realOutputs real outputs. It can be null because this method is flexible. This is optional parameter.
	 * @param params option parameter array which can be null. 
	 * @return error or loss of the output neuron.
	 */
	protected abstract NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params);

	
	/**
	 * Calculating derivative of neuron.
	 * @param neuron specified neuron.
	 * @return derivative of neuron.
	 */
	protected NeuronValue calcDerivative(NeuronStandard neuron) {
		return neuron.derivative();
	}
	
	
	@Override
	public Map<Integer, NeuronValue[]> updateWeightsBiases(List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
		if (bone.size() < 2) return Util.newMap(0);
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;

		//Setting input.
		Set<Integer> inputIndices = boneInput.keySet();
		for (int inputIndex : inputIndices) {
			if (inputIndex < 0 || inputIndex >= bone.size()) continue;
			NeuronValue[] input = boneInput.get(inputIndex);
			if (input == null) continue;
			
			LayerStandard layer = bone.get(inputIndex);
			layer.setInput(input);
			layer.setOutput(input);
		}
		
		//Evaluating bone.
		for (int i = 0; i < bone.size(); i++) {
			if (!inputIndices.contains(i)) bone.get(i).evaluate();
		}

		//Learning weights and biases.
		Map<Integer, NeuronValue[]> outputError = Util.newMap(boneOutput.size());
		Set<Integer> outputIndices = boneOutput.keySet();
		for (int outputIndex : outputIndices) {
			if (outputIndex <= 0 || outputIndex >= bone.size()) continue;
			NeuronValue[] output = boneOutput.get(outputIndex);
			if (output == null) continue;
			
			LayerStandard layer = bone.get(outputIndex);
			NeuronValue[] error = NeuronValue.makeArray(layer.size(), layer);
			for (int j = 0; j < layer.size(); j++) { //Browsing neurons of current layer.
				NeuronStandard neuron = layer.get(j);
				
				//Calculate error of current neuron at current layer.
				error[j] = calcOutputError(neuron, output[j], layer, -1, null);
				
				//Update biases of current layer.
				NeuronValue delta = error[j].multiply(learningRate);
				neuron.setBias(neuron.getBias().add(delta));
			}
			
			//Update weights of previous layer.
			Set<LayerStandard> prevLayers = layer.getAllPrevLayers(); //Include virtual layer.
			if (!prevLayers.contains(bone.get(outputIndex-1))) prevLayers.add(bone.get(outputIndex-1));
			for (LayerStandard prevLayer : prevLayers) {
				if (prevLayer == null) continue;
				for (int j = 0; j < prevLayer.size(); j++) {
					NeuronStandard prevNeuron = prevLayer.get(j);
					NeuronValue prevOut = prevNeuron.getOutput();
					
					WeightedNeuron[] targets = prevNeuron.getNextNeurons(layer);
					if (targets.length == 0) 
						targets = prevNeuron.getOutsideNextNeurons(layer).toArray(new WeightedNeuron[] {}); //Virtual layer.
					for (WeightedNeuron target : targets) {
						int index = layer.indexOf(target.neuron);
						if (!checkIndex(index)) continue;
						NeuronValue delta = error[index].multiply(prevOut).multiply(learningRate);
						Weight nw = target.weight;
						nw.value = nw.value.addValue(delta);
					}
				}
			}
			
			outputError.put(outputIndex, error);
		}
		
		return outputError;
	}
	
	
	/**
	 * Checking if index is valid.
	 * @param index specified index.
	 * @return if index is valid.
	 */
	protected boolean checkIndex(int index) {
		return index >= 0;
	}
	
	
}
