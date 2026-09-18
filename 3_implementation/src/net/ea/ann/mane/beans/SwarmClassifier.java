/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;

/**
 * This class implements Swarm classifier.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SwarmClassifier extends VGGClassifier {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default value for number of particles.
	 */
	public final static int PARTICLES_COUNT_SMALL = 2;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public SwarmClassifier(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		this.config.put(PARTICLES_COUNT_FIELD, PARTICLES_COUNT_SMALL);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public SwarmClassifier(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public SwarmClassifier(int neuronChannel, Function activateRef) {this(neuronChannel, activateRef, null, null);}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public SwarmClassifier(int neuronChannel) {this(neuronChannel, null, null, null);}
	
	
}



