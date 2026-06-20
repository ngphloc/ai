/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter;

import java.rmi.RemoteException;
import java.util.List;

/**
 * This represents the evaluator for generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Evaluator extends net.hudup.core.evaluate.Evaluator {

	
	/**
	 * Register list of algorithms without evaluation.
	 * @param algNameList list of algorithm names.
	 * @return registering is successfully.
	 * @throws RemoteException if any error occurs.
	 */
	boolean remoteStartWithoutEvaluation(List<String> algNameList) throws RemoteException;

	
}
