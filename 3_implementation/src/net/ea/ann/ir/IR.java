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

/**
 * This interface represents information retrieval system.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface IR extends Remote {

	
	/**
	 * Searching for given query feature.
	 * @param query query feature.
	 * @return list of found features.
	 * @throws RemoteException if any error raises.
	 */
	List<Feature> search(Feature query) throws RemoteException;

	
}
