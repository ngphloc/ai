/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.rmi.RemoteException;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvNetwork;
import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a convolutional network with stacks.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface StackNetwork extends ConvNetwork {


	/**
	 * Getting feature. 
	 * @return feature.
	 * @throws RemoteException if any error raises.
	 */
	Content getFeature() throws RemoteException;


	@Override
	NeuronValue[] evaluate(Record inputRecord) throws RemoteException;

	
	/**
	 * Learning the neural network one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnOne(Iterable<Record> sample) throws RemoteException;
	
	
	/**
	 * Learning the neural network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;


}
