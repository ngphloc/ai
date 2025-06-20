/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.rnn;

import java.util.List;
import java.util.Map;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.Network;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This class represents a state (standard network) in recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class State extends GeneratorStandard<Trainer> {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public State(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public State(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	public State(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
	/**
	 * Getting recurrent neural network.
	 * @return recurrent neural network.
	 */
	protected RecurrentNetworkAbstract getNetwork() {
		Network parent = getParent();
		return parent != null && parent instanceof RecurrentNetworkAbstract ? (RecurrentNetworkAbstract)parent : null;
	}
	
	
	@Override
	protected NeuronValue newNeuronValue(LayerStandard layer) {
		RecurrentNetworkAbstract network = getNetwork();
		return network != null ? network.newNeuronValue(this, layer) : super.newNeuronValue(layer);
	}

	
	/**
	 * Creating an empty neuron value.
	 * @param layer specified layer.
	 * @return an empty neuron value.
	 */
	NeuronValue newNeuronValueCaller(LayerStandard layer) {
		return super.newNeuronValue(layer);
	}

	
	@Override
	protected Weight newWeight(LayerStandard layer) {
		RecurrentNetworkAbstract network = getNetwork();
		return network != null ? network.newWeight(this, layer) : super.newWeight(layer);
	}

	
	/**
	 * Creating a new weight.
	 * @param layer specified layer.
	 * @return new weight.
	 */
	Weight newWeightCaller(LayerStandard layer) {
		return super.newWeight(layer);
	}

	
	@Override
	protected NeuronValue newBias(LayerStandard layer) {
		RecurrentNetworkAbstract network = getNetwork();
		return network != null ? network.newBias(this, layer) : super.newBias(layer);
	}

	
	/**
	 * Create a new bias.
	 * @param layer specified layer.
	 * @return new bias.
	 */
	NeuronValue newBiasCaller(LayerStandard layer) {
		return super.newBias(layer);
	}

	
	@Override
	protected NeuronStandard newNeuron(LayerStandard layer) {
		RecurrentNetworkAbstract network = getNetwork();
		return network != null ? network.newNeuron(this, layer) : super.newNeuron(layer);
	}
	
	
	/**
	 * Create neuron.
	 * @param layer specified layer.
	 * @return created neuron.
	 */
	NeuronStandard newNeuronCaller(LayerStandard layer) {
		return super.newNeuron(layer);
	}

	
	@Override
	protected LayerStandard newLayer() {
		RecurrentNetworkAbstract network = getNetwork();
		return network != null ? network.newLayer(this) : super.newLayer();
	}


	/**
	 * Create layer.
	 * @return created layer.
	 */
	LayerStandard newLayerCaller() {
		return super.newLayer();
	}

	
	@Override
	protected NeuronValue[] updateWeightsBiases(Backpropagator bp, List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		RecurrentNetworkAbstract network = getNetwork();
		if (network != null)
			return network.updateWeightsBiases(this, bp, bone, outputBatch, lastError, learningRate);
		else
			return super.updateWeightsBiases(bp, bone, outputBatch, lastError, learningRate);
	}


	/**
	 * Updating weights and biases. Derived class can override this method.
	 * @param bp backpropagator agorithm.
	 * @param bone list of layers including input layer.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the neuron output too. 
	 * @param lastError last error which is optional parameter for batch learning.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	NeuronValue[] updateWeightsBiasesCaller(Backpropagator bp, List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		return super.updateWeightsBiases(bp, bone, outputBatch, lastError, learningRate);
	}
	
	
	@Override
	protected Map<Integer, NeuronValue[]> updateWeightsBiases(Backpropagator bp, List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
		RecurrentNetworkAbstract network = getNetwork();
		if (network != null)
			return network.updateWeightsBiases(this, bp, bone, boneInput, boneOutput, learningRate);
		else
			return super.updateWeightsBiases(bp, bone, boneInput, boneOutput, learningRate);
	}


	/**
	 * Updating weights and biases for every layer. Derived class can override this method.
	 * @param bp backpropagator agorithm.
	 * @param bone list of layers including input layer.
	 * @param boneInput bone input. Note, index of layer is ID.
	 * @param boneOutput bone output. Note, index of layer is ID.
	 * @param learningRate learning rate.
	 * @return errors of output errors.
	 */
	Map<Integer, NeuronValue[]> updateWeightsBiasesCaller(Backpropagator bp, List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
		return super.updateWeightsBiases(bp, bone, boneInput, boneOutput, learningRate);
	}


	@Override
	protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		RecurrentNetworkAbstract network = getNetwork();
		if (network != null)
			return network.calcOutputError2(this, outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		else
			return super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
	}


	/**
	 * Calculate error of output neuron for the second activation function. Derived classes should implement this method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output.  
	 * @param bp backpropagator agorithm.
	 * @param outputNeuron output neuron. Output value of this neuron is retrieved by method {@link NeuronStandard#getOutput()}.
	 * @param realOutput real output. It can be null because this method is flexible.
	 * @param outputLayer output layer. It can be null because this method is flexible.
	 * @param outputNeuronIndex index of output neuron. It can be -1 because this method is flexible. This is optional parameter.
	 * @param realOutputs real outputs. It can be null because this method is flexible. This is optional parameter. 
	 * @return error or loss of the output neuron.
	 */
	NeuronValue calcOutputError2Caller(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		return super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
	}


}

