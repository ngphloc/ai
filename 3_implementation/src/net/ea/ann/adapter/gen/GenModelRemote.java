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
import net.hudup.core.alg.ExecuteAsLearnAlgRemote;

/**
 * This interface represents remote convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface GenModelRemote extends GenModelRemoteTask, ExecuteAsLearnAlgRemote {


	@Override
	int getNeuronChannel() throws RemoteException;

	
	@Override
	int getRasterChannel() throws RemoteException;

	
	@Override
	List<Raster> genRasters(Iterable<Raster> trainRasters, int nGens) throws RemoteException;

	
	@Override
	List<Raster> genRasters(int nGens) throws RemoteException;

	
	@Override
	List<G> recoverRasters(Iterable<Raster> sample, Iterable<Raster> rasters, int nGens) throws RemoteException;

	
}
