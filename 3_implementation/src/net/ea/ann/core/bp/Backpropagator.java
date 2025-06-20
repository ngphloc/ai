/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.bp;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import net.ea.ann.core.Evaluator;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents backpropagation algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Backpropagator extends Serializable, Cloneable {

	
	/**
	 * Updating weights and biases.
	 * @param sample sample.
	 * @param bone list of layers including input layer.
	 * @param learningRate learning rate.
	 * @param evaluator specified evaluator.
	 * @return errors of output errors. Return null if errors occur.
	 */
	NeuronValue[] updateWeightsBiases(Iterable<Record> sample, List<LayerStandard> bone, double learningRate, Evaluator evaluator);

	
	/**
	 * Updating weights and biases.
	 * @param bone list of layers including input layer.
	 * @param realOutput realistic output of output layer. 
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, NeuronValue[] realOutput, double learningRate);

	
	/**
	 * Updating weights and biases. Derived class can override this method.
	 * @param bone list of layers including input layer.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the neuron output too. 
	 * @param lastError last error which is optional parameter for batch learning.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate);

		
	/**
	 * Updating weights and biases for every layer. Derived class can override this method.
	 * @param bone list of layers including input layer.
	 * @param boneInput bone input. Note, index of layer is ID.
	 * @param boneOutput bone output. Note, index of layer is ID.
	 * @param learningRate learning rate.
	 * @return errors of output errors.
	 */
	Map<Integer, NeuronValue[]> updateWeightsBiases(List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate);

	
}
