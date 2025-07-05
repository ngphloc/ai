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
	 * @return array as output.
	 * @throws RemoteException if any error raises.
	 */
	Matrix evaluate(Matrix input) throws RemoteException;


	/**
	 * Learning matrix neural network.
	 * @param input input matrix for learning.
	 * @param output output matrix for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	Matrix learn(Matrix input, Matrix output) throws RemoteException;


}
