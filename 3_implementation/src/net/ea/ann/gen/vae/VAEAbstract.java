/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.function.Function;
import net.ea.ann.gen.GenModelAbstract;

/**
 * This class is the abstract class of Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class VAEAbstract extends GenModelAbstract implements VAE {


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
	protected VAEAbstract(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	protected VAEAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	protected VAEAbstract(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	/**
	 * Creating encoder.
	 * @return created encoder.
	 */
	protected abstract NetworkStandardImpl createEncoder();

	
	/**
	 * Creating decoder.
	 * @return created decoder.
	 */
	protected abstract NetworkStandardImpl createDecoder();
	
	
}