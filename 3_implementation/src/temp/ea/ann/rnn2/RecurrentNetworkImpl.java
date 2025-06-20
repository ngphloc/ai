/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn2;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;

/**
 * This class is default implementation of recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class RecurrentNetworkImpl extends MultiWeightNetworkImpl implements RecurrentNetwork {


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
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}


	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	public RecurrentNetworkImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
}
