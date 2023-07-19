/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.rmi.RemoteException;

import net.ea.ann.core.Network;
import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.Record;

/**
 * This interface represents a convolutional network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvNetwork extends Network {


	/**
	 * Evaluating the convolutional network by input image.
	 * @param inputImage input image for evaluating.
	 * @return array as output of output layer.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] evaluateByImage(ImageSpec inputImage) throws RemoteException;

	
	/**
	 * Learning the convolutional network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;

	
}
