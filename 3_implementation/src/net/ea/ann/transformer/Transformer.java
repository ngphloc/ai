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
	 * Evaluating matrix neural network.
	 * @param input1 the first input matrix.
	 * @param input2 the second input matrix.
	 * @return array as output.
	 * @throws RemoteException if any error raises.
	 */
	Matrix evaluate(Matrix input1, Matrix input2) throws RemoteException;


	/**
	 * Learning matrix neural network.
	 * @param input1 the first input matrix.
	 * @param input2 the second input matrix.
	 * @param output output matrix for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	Matrix learn(Matrix input1, Matrix input2, Matrix output) throws RemoteException;


}
