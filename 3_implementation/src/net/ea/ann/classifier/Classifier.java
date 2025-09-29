/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This interface represents classifier.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Classifier extends Remote, Cloneable {

	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 * @throws RemoteException if any error raises.
	 */
	int getNeuronChannel() throws RemoteException;

	
	/**
	 * Learning classifier one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException;


	/**
	 * Learning classifier.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException;


	/**
	 * Classifying sample.
	 * @param sample specified sample.
	 * @return classified sample.
	 * @throws RemoteException if any error raises.
	 */
	List<Raster> classify(Iterable<Raster> sample) throws RemoteException;
	
	
}
