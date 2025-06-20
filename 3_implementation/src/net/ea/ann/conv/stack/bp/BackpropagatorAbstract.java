/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack.bp;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.RecordExt;
import net.ea.ann.conv.stack.ElementLayer;
import net.ea.ann.conv.stack.Stack;
import net.ea.ann.conv.stack.StackAbstract;
import net.ea.ann.conv.stack.WeightedElementLayer;
import net.ea.ann.core.Evaluator;
import net.ea.ann.core.Network;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Weight;

/**
 * This class is abstract implementation of backpropagation algorithm for stack network.
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
	 * Default constructor
	 */
	public BackpropagatorAbstract() {
		super();
	}


	/*
	 * Method can be overridden.
	 */
	@Override
	public Content[] updateWeightsBiases(List<Stack> stacks, Iterable<Content[][]> outputBatch, Content[] lastError, double learningRate) {
		return super.updateWeightsBiases(stacks, outputBatch, lastError, learningRate);
	}


	/*
	 * Method must be implemented.
	 */
	@Override
	protected abstract Content calcOutputError(ElementLayer outputLayer, Content realOutput, Stack outputStack);


	/**
	 * Default method to calculate output error.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output. 
	 * @param outputLayer output layer.
	 * @param realOutput real output. It can be null.
	 * @param outputStack output stack. It can be null.
	 * @return output error.
	 */
	public static Content calcOutputErrorDefault(ElementLayer outputLayer, Content realOutput, Stack outputStack) {
		Function activateRef = outputLayer != null ? outputLayer.getActivateRef() : null;
		if (activateRef == null && realOutput != null) activateRef = realOutput.getActivateRef();
		return calcOutputErrorDefault(activateRef, realOutput, outputLayer != null ? outputLayer.getContent() : null);
	}

	
	/**
	 * Calculate error of output neuron. This is utility method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output. 
	 * @param activateRef activation function. It can be null. If it is null, the derivative is 1.
	 * @param realOutput realistic output. It cannot be null.
	 * @param layerOutput layer output. It cannot be null.
	 * @return error of output neuron.
	 */
	public static Content calcOutputErrorDefault(Function activateRef, Content realOutput, Content layerOutput) {
		if (layerOutput == null)
			return null;
		else if (activateRef == null)
			return realOutput.subtract(layerOutput);
		else {
			Content derivative = layerOutput.derivative0(activateRef);
			return realOutput.subtract(layerOutput).multiplyDerivative(derivative);
		}
	}


}



/**
 * This class is basic abstract implementation of backpropagation algorithm for stack network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class BackpropagatorAbstract0 implements Backpropagator, Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor
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
	public Content[] updateWeightsBiases(Iterable<Record> sample, List<Stack> stacks, double learningRate, Evaluator evaluator) {
		if (stacks.size() < 2) return null;
		Stack outputStack = stacks.get(stacks.size() - 1);
		Content[] meanOutputError = new Content[outputStack.size()];
		Content zero =  StackAbstract.newOutputContent(outputStack);
		for (int i = 0; i < meanOutputError.length; i++) meanOutputError[i] = zero;
		
		//Calculating mean output error.
		int n = 0;
		for (Record record0 : sample) {
			if ((record0 == null) || !(record0 instanceof RecordExt)) continue;
			RecordExt record = (RecordExt)record0;

			//Evaluating network.
			try {
				if (evaluator != null) evaluator.evaluate(record);
			} catch (Throwable e) {Util.trace(e);}

			//Calculating output error.
			Content[] outputError = calcOutputError(outputStack, record.contentOutput); // The second output is content output for learning convolutional network.
			if (outputError == null) continue;
			
			for (int i = 0; i < meanOutputError.length; i++) meanOutputError[i] = meanOutputError[i].add(outputError[i]);
			n++;
		}
		
		if (n == 0) return null;
		
		//Updating weights and biases of bone.
		for (int i = 0; i < meanOutputError.length; i++) meanOutputError[i] = meanOutputError[i].divide0(n);
		return updateWeightsBiases(stacks, learningRate, meanOutputError);			
	}

	
	@Override
	public Content[] updateWeightsBiases(List<Stack> stacks, Content[] realOutput, double learningRate) {
		if (realOutput == null) updateWeightsBiases(stacks, (Iterable<Content[][]>)null, learningRate);

		List<Content[][]> outputBatch = Util.newList(1);
		outputBatch.add(new Content[][] {realOutput});
		return updateWeightsBiases(stacks, outputBatch, learningRate);
	}

	
	/**
	 * Updating weights and biases.
	 * @param stacks list of stacks including input stack.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the neuron output too. 
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	private Content[] updateWeightsBiases(List<Stack> stacks, Iterable<Content[][]> outputBatch, double learningRate) {
		return updateWeightsBiases(stacks, outputBatch, null, learningRate);
	}
	
	
	/**
	 * Updating weights and biases.
	 * @param stacks list of stacks including input stack.
	 * @param lastError last error which is optional parameter.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	private Content[] updateWeightsBiases(List<Stack> stacks, double learningRate, Content[] lastError) {
		return updateWeightsBiases(stacks, null, lastError, learningRate);
	}
	

	@Override
	public Content[] updateWeightsBiases(List<Stack> stacks, Iterable<Content[][]> outputBatch, Content[] lastError, double learningRate) {
		if (stacks.size() < 2) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
		Content[] outputError = null;
		
		Content[] nextError = lastError;
		for (int i = stacks.size() - 1; i >= 1; i--) { //Browsing stacks reversely from output stack down to first hidden stack.
			Stack stack = stacks.get(i);
			Content[] error = StackAbstract.makeArray(stack.size(), stack);
			
			for (int j = 0; j < stack.size(); j++) { //Browsing neurons of current layer.
				ElementLayer layer = stack.get(j);
				
				//Calculate error of current neuron at current stack.
				if (i == stacks.size() - 1) {//Calculate error of last stack. This is most important for backpropagation algorithm.
					error[j] = nextError == null ? calcOutputError(stack, j, outputBatch) : nextError[j];
				}
				else {//Calculate error of of hidden stacks.
					Stack nextStack = stacks.get(i + 1);
					Content rsum = StackAbstract.newContent(i, stack);
					List<WeightedElementLayer> targets = layer.getNextLayers(nextStack);
					for (WeightedElementLayer target : targets) {
						int index = nextStack.indexOf(target.layer);
						if (!checkIndex(index)) continue;
						rsum = rsum.add(nextError[index].multiply0(target.weight.value));
					}
					
					if (layer.getActivateRef() != null) {
						Content out = layer.getContent();
						Content derivative = out.derivative0(layer.getActivateRef());
						error[j] = rsum.multiplyDerivative(derivative);
					}
					else
						error[j] = rsum;
				}
				
				//Update biases of current stack.
				if (isLearningBias()) {
					Content delta = error[j].multiply0(learningRate);
					layer.setBias(layer.getBias().add(delta.mean0()));
				}
			}
			
			//Update weights stored in previous stacks.
			Set<Stack> prevStacks = stack.getAllPrevStacks();
			if (!prevStacks.contains(stacks.get(i-1))) prevStacks.add(stacks.get(i-1)); 
			for (Stack prevStack : prevStacks) {
				if (prevStack == null) continue;
				for (int j = 0; j < prevStack.size(); j++) {
					ElementLayer prevLayer = prevStack.get(j);
					Content prevOut = prevLayer.getContent();
					
					List<WeightedElementLayer> targets = prevLayer.getNextLayers(stack);
					for (WeightedElementLayer target : targets) {
						int index = stack.indexOf(target.layer);
						if (!checkIndex(index)) continue;
						Content delta = error[index].multiply(prevOut).multiply0(learningRate);
						Weight nw = target.weight;
						nw.value = nw.value.add(delta.mean0());
					}
				}
			}
			
			nextError = error;
			if (i == stacks.size() - 1) outputError = error;
		}
		
		return outputError;
	}

	
	/**
	 * Calculating output error of output layer at specified index. Derived class can call or override this method.
	 * @param outputStack output stack.
	 * @param outputLayerIndex index of output layer.
	 * @param outputBatch output batch of output stack.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is layer output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the layer output too. 
	 * @return output error of output layer at specified index.
	 */
	protected Content calcOutputError(Stack outputStack, int outputLayerIndex, Iterable<Content[][]> outputBatch) {
		ElementLayer outputNeuron = outputStack.get(outputLayerIndex);
		//There are some cases that have no output (null output).
		if (outputBatch == null) return calcOutputError(outputNeuron, null, outputStack);

		int n = 0;
		Content errorMean = StackAbstract.newOutputContent(outputStack);
		for (Content[][] outputs : outputBatch) {
			Content[] realOutputs = (outputs != null && outputs.length > 0) ? outputs[0] : null;
			if (realOutputs != null) realOutputs = StackAbstract.adjustArray(realOutputs, outputStack.size(), outputStack);
			
			Content realOutput = realOutputs != null ? realOutputs[outputLayerIndex] : null; //There are some cases that have no output (null output).
			Content error = null;
			if (outputs == null || outputs.length <= 1)
				error = calcOutputError(outputNeuron, realOutput, outputStack);
			else {
				Content[] layerOutputs = outputs[1];
				if (layerOutputs == null)
					error = calcOutputError(outputNeuron, realOutput, outputStack);
				else {
					layerOutputs = StackAbstract.adjustArray(layerOutputs, outputStack.size(), outputStack);
					error = BackpropagatorAbstract.calcOutputErrorDefault(outputNeuron.getActivateRef(), realOutput, (layerOutputs != null ? layerOutputs[outputLayerIndex] : null));
				}
			}
			
			if (error == null) continue;
			
			errorMean = errorMean.add(error);
			n++;
		}
		
		if (n != 0 && n != 1) errorMean = errorMean.divide0((double)n);
		return errorMean;
	}

	
	/**
	 * Calculating output error of output stack.
	 * @param outputStack output stack.
	 * @param outputBatch output batch of output stack.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is layer output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the layer output too. 
	 * @return output error of output layer at specified index.
	 */
	private Content[] calcOutputError(Stack outputStack, Iterable<Content[][]> outputBatch) {
		Content[] error = new Content[outputStack.size()]; 
		for (int j = 0; j < outputStack.size(); j++) {
			error[j] = calcOutputError(outputStack, j, outputBatch);
		}
		return error;
	}

	
	/**
	 * Calculating output error of output stack.
	 * @param outputStack output stack.
	 * @param realOutput realistic output of output layer.
	 * @return output error of output layer at specified index.
	 */
	private Content[] calcOutputError(Stack outputStack, Content[] realOutput) {
		List<Content[][]> outputBatch = Util.newList(1);
		outputBatch.add(new Content[][] {realOutput});
		return calcOutputError(outputStack, outputBatch);
	}

	
	/**
	 * Calculate output error of output layer. This method is the most important entrance of backpropagation algorithm.
	 * This error is the opposite of gradient of output error function for minimization or the gradient of output error function for maximization. 
	 * Similarly, this error is the gradient of output gain function for minimization or the opposite of gradient of output gain function for maximization.<br>
	 * <br>
	 * In general, this error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output.  
	 * @param outputLayer output layer.
	 * @param realOutput real output because this method is flexible.
	 * @param outputStack output stack because this method is flexible.
	 * @return output error.
	 */
	protected abstract Content calcOutputError(ElementLayer outputLayer, Content realOutput, Stack outputStack);

	
	/**
	 * Checking if index is valid.
	 * @param index specified index.
	 * @return if index is valid.
	 */
	private boolean checkIndex(int index) {
		return index >= 0;
	}


}

