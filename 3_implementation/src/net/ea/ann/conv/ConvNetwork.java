/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.rmi.RemoteException;

import net.ea.ann.core.Evaluator;
import net.ea.ann.core.Network;
import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This interface represents a convolutional network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvNetwork extends Network, Evaluator {


	/**
	 * Getting feature. 
	 * @return feature.
	 * @throws RemoteException if any error raises.
	 */
	ConvLayerSingle getFeature() throws RemoteException;

	
	/**
	 * Evaluating the convolutional network by input raster.
	 * @param inputRaster input raster for evaluating.
	 * @return array as output of output layer.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] evaluateRaster(Raster inputRaster) throws RemoteException;


	@Override
	NeuronValue[] evaluate(Record inputRecord) throws RemoteException;

	
	/**
	 * Learning the convolutional network one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnOne(Iterable<Record> sample) throws RemoteException;

	
	/**
	 * Learning the convolutional network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;


}
