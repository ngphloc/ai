/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.rmi.RemoteException;

import net.ea.ann.Network;
import net.ea.ann.NeuronValue;
import net.ea.ann.Record;

/**
 * This class represents an interface of Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface VAE extends Network {


	/**
	 * Learning the Variational Autoencoders.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;


	/**
	 * Generate values (X values).
	 * @return generated values.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] generate() throws RemoteException;
	
	
}
