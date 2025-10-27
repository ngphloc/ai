/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.rmi.RemoteException;

import net.ea.ann.core.Network;
import net.ea.ann.core.value.Matrix;

/**
 * This interface represents transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Transformer extends Network {

	
	/**
	 * Evaluating transformer.
	 * @param record record.
	 * @throws RemoteException if any error raises.
	 */
	Matrix evaluate(Record record) throws RemoteException;


	/**
	 * Learning transformer.
	 * @param sample sample.
	 * @return learning errors. The first element is main error and the second element is attached error, etc.
	 * @throws RemoteException if any error raises.
	 */
	Error[][] learn(Iterable<Record> sample) throws RemoteException;


}
