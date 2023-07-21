/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.vae;

import java.io.Serializable;
import java.rmi.RemoteException;

import net.hudup.core.alg.Alg;
import net.hudup.core.data.Profile;
import net.hudup.core.evaluate.execute.ExecuteAsLearnEvaluator;

/**
 * This class is the evaluator for convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvVAEEvaluator extends ExecuteAsLearnEvaluator {
	

	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public ConvVAEEvaluator() {

	}

	
	@Override
	public boolean acceptAlg(Alg alg) throws RemoteException {
		return (alg != null) && (alg instanceof ConvVAE);
	}


	@Override
	public String getName() throws RemoteException {
		return "convvae";
	}


	@Override
	protected Serializable extractTestValue(Alg alg, Profile testingProfile) {
		if (testingProfile == null) return null;
		if (!(alg instanceof ConvVAE)) return null;

		return null;
	}

	
}
