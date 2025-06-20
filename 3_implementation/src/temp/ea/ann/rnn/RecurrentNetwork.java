/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.Network;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RecurrentNetwork extends Network {


	/**
	 * Recurrent network layout.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	enum Layout {
		
		/**
		 * Output-input layout.
		 * In output-input layout, outputs of previous state become inputs of current state.
		 * In other words, output layer of previous state connects to the first hidden layer of current state.
		 * This implies the current state has two input layers, one input layer from itself and one input layer from previous state.
		 */
		outin,
		
		/**
		 * Parallel layout.
		 * In parallel layout, all layers of previous network connect parallel with all layers of previous network.
		 * Note, all connections are parallel because any pair of neurons (between two networks) have only one connection.
		 * It is implied that there is only one input layer for entire parallel layout recurrent network. This unique input layer is attached to the first layer. 
		 */
		parallel,
		
	}
	
	
	/**
	 * Evaluating network.
	 * @param input specified input.
	 * @throws RemoteException if any error raises.
	 */
	void evaluate(NeuronValue...input) throws RemoteException;
	
	
	/**
	 * Evaluating network.
	 * @param inputs list of inputs.
	 * @throws RemoteException if any error raises.
	 */
	void evaluate(List<NeuronValue[]> inputs) throws RemoteException;

	
}
