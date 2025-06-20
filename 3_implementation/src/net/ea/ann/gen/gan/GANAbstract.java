/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.gan;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.function.Function;
import net.ea.ann.gen.GenModelAbstract;

/**
 * This class is the abstract class of Generative Adversarial Network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class GANAbstract extends GenModelAbstract implements GAN {


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
	protected GANAbstract(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	protected GANAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected GANAbstract(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	/**
	 * Creating decoder.
	 * @return decoder.
	 */
	protected abstract NetworkStandardImpl createDecoder();


}