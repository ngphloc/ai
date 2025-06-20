/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn.lstm;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.function.Function;
import temp.ea.ann.rnn.RecurrentNetworkImpl;

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
	 * Index of output gate.
	 */
	protected final static int PARAMES_OUTPUTGATE_INDEX = 2;

	
	/**
	 * Index of cell input.
	 */
	protected final static int PARAMES_CELLINPUT_INDEX = 3;

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public LongShortTermMemoryImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public LongShortTermMemoryImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public LongShortTermMemoryImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	@Override
	protected NetworkStandardImpl newState() {
		return new State(neuronChannel, activateRef, idRef);
	}


}


