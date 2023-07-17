/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.rmi.RemoteException;

/**
 * This interface represents standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NetworkStandard extends Network {


	/**
	 * Layer type.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	enum LayerType {
		
		/**
		 * Input layer.
		 */
		input,
		
		/**
		 * Hidden layer.
		 */
		hidden,
		
		/**
		 * Output layer.
		 */
		output,
		
		/**
		 * Memory layer.
		 */
		memory,
		
		/**
		 * Input rib layer.
		 */
		ribin,
		
		/**
		 * Memory layer.
		 */
		ribout,
		
		/**
		 * Unknown layer.
		 */
		unknown,
		
	}
	
	
	/**
	 * Evaluating entire network.
	 * @param inputRecord input record for evaluating.
	 * @param refresh refresh mode.
	 * @return array as output of output layer.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] eval(Record inputRecord, boolean refresh) throws RemoteException;
	
	
	/**
	 * Learning the neural network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;
	
	
}
