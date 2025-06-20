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

import net.ea.ann.adapter.gen.ui.GenUI;
import net.ea.ann.gen.GenModel.G;
import net.ea.ann.raster.Raster;
import net.hudup.core.Util;
import net.hudup.core.alg.AllowNullTrainingSet;
import net.hudup.core.alg.ExecuteAsLearnAlgRemoteWrapper;
import net.hudup.core.logistic.BaseClass;
import net.hudup.core.logistic.Inspector;

/**
 * The class is a wrapper of remote generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@BaseClass //The annotation is very important which prevent Firer to instantiate the wrapper without referred remote object. This wrapper is not normal algorithm.
public class GenModelRemoteWrapper extends ExecuteAsLearnAlgRemoteWrapper implements GenModel, GenModelRemote, AllowNullTrainingSet {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with specified remote generative model.
	 * @param remoteGM remote generative model.
	 */
	public GenModelRemoteWrapper(GenModelRemote remoteGM) {
		super(remoteGM);
	}

	
	/**
	 * Constructor with specified remote generative model and exclusive mode.
	 * @param remoteGM remote generative model.
	 * @param exclusive exclusive mode.
	 */
	public GenModelRemoteWrapper(GenModelRemote remoteGM, boolean exclusive) {
		super(remoteGM, exclusive);
	}

	
	@Override
	public int getNeuronChannel() throws RemoteException {
		GenModelRemote remoteGM = (GenModelRemote)getRemoteAlg();
		return remoteGM.getNeuronChannel();
	}


	@Override
	public int getRasterChannel() throws RemoteException {
		GenModelRemote remoteGM = (GenModelRemote)getRemoteAlg();
		return remoteGM.getRasterChannel();
	}


	@Override
	public List<Raster> genRasters(Iterable<Raster> sample, int nGens) throws RemoteException {
		GenModelRemote remoteGM = (GenModelRemote)getRemoteAlg();
		return remoteGM != null ? remoteGM.genRasters(sample, nGens) : Util.newList();
	}


	@Override
	public List<Raster> genRasters(int nGens) throws RemoteException {
		GenModelRemote remoteGM = (GenModelRemote)getRemoteAlg();
		return remoteGM != null ? remoteGM.genRasters(nGens) : Util.newList();
	}


	@Override
	public List<G> recoverRasters(Iterable<Raster> sample, Iterable<Raster> rasters, int nGens) throws RemoteException {
		GenModelRemote remoteGM = (GenModelRemote)getRemoteAlg();
		return remoteGM != null ? remoteGM.recoverRasters(sample, rasters, nGens) : Util.newList();
	}


	@Override
	public Inspector getInspector() {
		return new GenUI(this, true);
	}


}
