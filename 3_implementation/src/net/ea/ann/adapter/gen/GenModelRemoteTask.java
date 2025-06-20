/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.gen.GenModel.G;
import net.ea.ann.raster.Raster;
import net.hudup.core.alg.ExecuteAsLearnAlgRemoteTask;

/**
 * This interface declares methods for remote convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
public interface GenModelRemoteTask extends ExecuteAsLearnAlgRemoteTask {


	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 * @throws RemoteException if any error raises.
	 */
	int getNeuronChannel() throws RemoteException;

	
	/**
	 * Getting raster channel.
	 * @return raster channel.
	 * @throws RemoteException if any error raises.
	 */
	int getRasterChannel() throws RemoteException;

		
	/**
	 * Generating rasters from sample.
	 * @param sample specified sample.
	 * @param nGens number of generated rasters.
	 * @return generated rasters.
	 * @throws RemoteException if any error raises.
	 */
	List<Raster> genRasters(Iterable<Raster> sample, int nGens) throws RemoteException;
	
	
	/**
	 * Generating rasters.
	 * @param nGens number of generated rasters.
	 * @return generated rasters.
	 * @throws RemoteException if any error raises.
	 */
	List<Raster> genRasters(int nGens) throws RemoteException;

	
	/**
	 * Recovering rasters
	 * @param sample specified sample.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 * @throws RemoteException if any error raises.
	 */
	List<G> recoverRasters(Iterable<Raster> sample, Iterable<Raster> rasters, int nGens) throws RemoteException;

	
}
