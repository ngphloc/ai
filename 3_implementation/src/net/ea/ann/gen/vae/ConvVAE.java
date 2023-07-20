/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.rmi.RemoteException;

import net.ea.ann.conv.Raster;
import net.ea.ann.core.NeuronValue;

/**
 * This interface represent a convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvVAE extends VAE {


	/**
	 * Learning the convolutional Variational Autoencoders by raster.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnByRaster(Iterable<Raster> sample) throws RemoteException;

	
	/**
	 * Generate raster.
	 * @return generated raster.
	 * @throws RemoteException if any error raises.
	 */
	Raster generateRaster() throws RemoteException;

	
}
