/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.rmi.RemoteException;

import net.ea.ann.core.Network;
import net.ea.ann.core.value.Matrix;

/**
 * This interface represents matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface MatrixNetwork extends Network {

	
	/**
	 * Evaluating matrix neural network.
	 * @param input input matrix for evaluating.
	 * @param params additional parameters.
	 * @return matrix as output.
	 * @throws RemoteException if any error raises.
	 */
	Matrix evaluate(Matrix input, Object...params) throws RemoteException;


	/**
	 * Learning matrix neural network.
	 * @param sample sample.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	Error[] learn(Iterable<Record> sample) throws RemoteException;


}
