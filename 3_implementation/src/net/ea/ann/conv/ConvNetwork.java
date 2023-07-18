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

/**
 * This interface represents a convolutional network.
 * 
 * @author Loc Nguyen
 *
 */
public interface ConvNetwork extends Network {


	/**
	 * Learning the convolutional network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnByImages(Iterable<SerializableImage> sample) throws RemoteException;

	
}
