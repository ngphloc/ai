/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.rmi.RemoteException;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.value.Matrix;

/**
 * This class implements matrix neural network in default.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixNetworkImpl extends NetworkAbstract implements MatrixNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	public MatrixNetworkImpl(Id idRef) {
		super(idRef);
	}

	
	/**
	 * Default constructor.
	 */
	public MatrixNetworkImpl() {
		this(new Id());
	}

	
	@Override
	public Matrix evaluate(Matrix input) throws RemoteException {
		throw new RuntimeException("Method MatrixNetworkImpl.evaluate(Matrix) not implemented yet");
	}

	
	@Override
	public Matrix learn(Matrix input, Matrix output) throws RemoteException {
		throw new RuntimeException("Method MatrixNetworkImpl.learn(Matrix, Matrix) not implemented yet");
	}

	
}
