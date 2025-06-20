/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.rmi.Remote;
import java.rmi.RemoteException;

import net.ea.ann.conv.Content;
import net.ea.ann.raster.Raster;

/**
 * This interface provides remote methods to retrieve feature from generative model.
 * In context of generative model, X is data (original as well as generated) and feature is feature of some other digital content like image.
 * @author USER
 *
 */
public interface FeatureGetter extends Cloneable, Remote {

	
	/**
	 * Getting feature.
	 * @return feature.
	 * @throws RemoteException if any error raises.
	 */
	Content getFeature() throws RemoteException;
	
	
	/**
	 * Getting feature of specified raster.
	 * @param raster specified raster.
	 * @return feature of specified raster.
	 * @throws RemoteException if any error raises.
	 */
	Content getFeature(Raster raster) throws RemoteException;
	
	
}
