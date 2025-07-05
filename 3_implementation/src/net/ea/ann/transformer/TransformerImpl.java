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
 * This class implements simplest transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerImpl extends NetworkAbstract implements Transformer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal attention.
	 */
	protected Attention attention = null;
	
	
	/**
	 * Feed forward network.
	 */
	protected MatrixNetwork ffn = null;
	
	
	/**
	 * Add & norm component.
	 */
	protected AddNorm addNorm = null;
	
	
	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	protected TransformerImpl(Id idRef) {
		super(idRef);
	}

	
	/**
	 * Default constructor.
	 */
	protected TransformerImpl() {
		this(new Id());
	}

	
	@Override
	public Matrix evaluate(Matrix input1, Matrix input2) throws RemoteException {
		throw new RuntimeException("Method MatrixNetworkImpl.evaluate(Matrix) not implemented yet");
	}

	
	@Override
	public Matrix learn(Matrix input1, Matrix input2, Matrix output) throws RemoteException {
		throw new RuntimeException("Method MatrixNetworkImpl.learn(Matrix, Matrix) not implemented yet");
	}

	
}
