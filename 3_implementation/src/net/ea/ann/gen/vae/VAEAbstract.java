/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import net.ea.ann.Id;
import net.ea.ann.NetworkAbstract;
import net.ea.ann.function.IdentityFunction1;
import net.ea.ann.function.Function;

/**
 * This class is the default implementation of Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class VAEAbstract extends NetworkAbstract implements VAE {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public VAEAbstract(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		
		if (neuronChannel <= 1 && activateRef == null) {
			this.neuronChannel = 1;
			this.activateRef = new IdentityFunction1();
		}
		else {
			this.neuronChannel = neuronChannel;
			this.activateRef = activateRef;
		}
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public VAEAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 */
	public VAEAbstract() {
		this(1, null, null);
	}

	
}