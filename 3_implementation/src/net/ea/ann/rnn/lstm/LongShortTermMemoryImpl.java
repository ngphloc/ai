/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.rnn.lstm;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.bp.BackpropagatorAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.ReLU;
import net.ea.ann.core.generator.GeneratorStandard.Layer;
import net.ea.ann.core.generator.GeneratorStandard.Neuron;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.core.value.Weight;
import net.ea.ann.core.value.WeightValue;
import net.ea.ann.core.value.vector.NeuronValueVectorImpl;
import net.ea.ann.raster.Raster;
import net.ea.ann.rnn.RecurrentNetwork;
import net.ea.ann.rnn.RecurrentNetworkImpl;
import net.ea.ann.rnn.State;

/**
 * This class is default implementation of long short-term memory.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LongShortTermMemoryImpl extends RecurrentNetworkImpl implements LongShortTermMemory {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param auxActivateRef auxiliary activation function.
	 * @param idRef identifier reference.
	 */
	public LongShortTermMemoryImpl(int neuronChannel, Function activateRef, Function auxActivateRef, Id idRef) {
		super(neuronChannel, activateRef, auxActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param auxActivateRef auxiliary activation function.
	 */
	public LongShortTermMemoryImpl(int neuronChannel, Function activateRef, Function auxActivateRef) {
		this(neuronChannel, activateRef, auxActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public LongShortTermMemoryImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
		this.auxActivateRef = Raster.toReLUActivationRef(neuronChannel, isNorm());
	}


	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public LongShortTermMemoryImpl(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	protected NeuronValue newNeuronValue(State state, LayerStandard layer) {
		return new NeuronValueVectorImpl(Cell.GATE_NUMBERS, NeuronValueCreator.newNeuronValue(neuronChannel)).zero();
	}


	@Override
	protected Weight newWeight(State state, LayerStandard layer) {
		WeightValue weightValue = newNeuronValue(state, layer).newWeightValue().zero();
		return new Weight(weightValue);
	}


	@Override
	protected NeuronValue newBias(State state, LayerStandard layer) {
		return newNeuronValue(state, layer).zero();
	}


	@Override
	protected NeuronStandard newNeuron(State state, LayerStandard layer) {
		return new Cell(layer);
	}


	@Override
	protected NeuronValue calcOutputError2(State state, NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		realOutput = RecurrentNetwork.verify(realOutput, outputNeuron);
		
		if ((outputNeuron == null) || !(outputNeuron instanceof Neuron))
			return super.calcOutputError2(state, outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		Function auxActivateRef = ((Neuron)outputNeuron).getAuxActivateRef();
		if (auxActivateRef == null && outputLayer != null && outputLayer instanceof Layer)
			auxActivateRef = ((Layer)outputLayer).getAuxActivateRef();
		if ((auxActivateRef == null) || !(auxActivateRef instanceof ReLU))
			return super.calcOutputError2(state, outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		if (auxActivateRef == this.activateRef)
			return super.calcOutputError2(state, outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);

		NeuronValue neuronOutput = outputNeuron != null ? outputNeuron.getOutput() : null;
		NeuronValue neuronInput = NeuronStandard.getDerivativeInput(outputNeuron);
		return BackpropagatorAbstract.calcOutputErrorDefault(auxActivateRef, realOutput, neuronOutput, neuronInput);
	}


}


