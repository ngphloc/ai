/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.transformer;

import java.io.Serializable;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkStandardAbstract;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.GeneratorStandard.Backpropagator;
import net.ea.ann.core.generator.GeneratorStandard.Layer;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.Weight;

/**
 * This class represents standard attention.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Attention extends LinkedGenerator {


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
	public Attention(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public Attention(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public Attention(int neuronChannel) {
		this(neuronChannel, null, null);
	}


}



/**
 * This class represents linked generator.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class LinkedGenerator extends GeneratorStandard<Trainer> implements Cloneable, Serializable {


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
	public LinkedGenerator(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public LinkedGenerator(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public LinkedGenerator(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	@Override
	protected LayerStandard newLayer() {
		LinkedLayer layer = new LinkedLayer(neuronChannel, activateRef, idRef);
		layer.setNetwork(this);
		return layer;
	}


	@Override
	protected Backpropagator createBackpropagator() {
		LinkedBackpropagator bp = new LinkedBackpropagator ();
		bp.setNetwork(this);
		return bp;
	}


}



/**
 * This class represents linked backpropagation algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class LinkedBackpropagator extends Backpropagator {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public LinkedBackpropagator() {
		super();
	}


	/*
	 * Modifying this method for zero tokens.
	 */
	@Override
	protected boolean checkIndex(int index) {
		return super.checkIndex(index);
	}

	
}



/**
 * This class represents linked layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class LinkedLayer extends Layer {

	
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
	public LinkedLayer(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public LinkedLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}


	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public LinkedLayer(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	@Override
	protected void setNetwork(NetworkStandardAbstract network) {
		super.setNetwork(network);
	}


	/*
	 * Modifying this method for matrix weight.
	 */
	@Override
	protected Weight newWeightCaller() {
		return super.newWeightCaller();
	}


	/*
	 * Modifying this method for neurons with different dimensions.
	 */
	@Override
	protected NeuronStandard newNeuronCaller() {
		return super.newNeuronCaller();
	}


}


