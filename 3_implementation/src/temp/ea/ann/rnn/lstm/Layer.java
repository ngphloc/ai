/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn.lstm;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandardImpl;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.indexed.IndexedNeuronValue1;
import net.ea.ann.core.value.indexed.IndexedNeuronValueV;

/**
 * This class is standard layer of a state (standard network) in recurrent neural network..
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class Layer extends LayerStandardImpl {


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
	public Layer(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public Layer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public Layer(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
	/**
	 * Getting the number of parameters.
	 * @return the number of parameters.
	 */
	protected abstract int getParamCount();

	
	/**
	 * Getting parameter index.
	 * @return parameter index.
	 */
	protected abstract int getParamIndex();
	
	
	/**
	 * Setting parameter index.
	 * @param paramIndex parameter index.
	 */
	protected abstract void setParamIndex(int paramIndex);

		
	/**
	 * Creating empty activation cell value.
	 * @return empty activation cell value.
	 */
	protected NeuronValue newCellValue( ) {
		return super.newNeuronValue();
	}
	
	
	@Override
	public NeuronValue newNeuronValue() {
		if (neuronChannel <= 0)
			return null;
		else if (neuronChannel == 1) {
			return new IndexedNeuronValue1(getParamCount(), 0) {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public int getIndex() {
					return getParamIndex();
				}
				
			};
		}
		else {
			return new IndexedNeuronValueV(getParamCount(), neuronChannel, 0) {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public int getIndex() {
					return getParamIndex();
				}
				
			};
		}
	}
	

}
