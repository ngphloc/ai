/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;
import java.rmi.Remote;
import java.rmi.RemoteException;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents an evaluator.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Evaluator extends Remote, Serializable, Cloneable {

	
	/**
	 * Evaluating task.
	 * @param inputRecord input record for evaluating.
	 * @return array as output.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] evaluate(Record inputRecord) throws RemoteException;


}
