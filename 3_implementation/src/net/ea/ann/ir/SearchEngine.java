/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.raster.Raster;

/**
 * This interface represents search engine.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface SearchEngine extends Remote {

	
	/**
	 * Searching for given query raster.
	 * @param query query raster.
	 * @return list of found rasters.
	 * @throws RemoteException if any error raises.
	 */
	List<Raster> search(Raster query) throws RemoteException;


	/**
	 * Starting search engine.
	 * @return true if starting is successful.
	 * @throws RemoteException if any error raises.
	 */
	boolean start() throws RemoteException;

	
	/**
	 * Stopping search engine.
	 * @return true if stopping is successful.
	 * @throws RemoteException if any error raises.
	 */
	boolean stop() throws RemoteException;
	
	
}
