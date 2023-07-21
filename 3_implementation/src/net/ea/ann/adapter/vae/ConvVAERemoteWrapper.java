/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.vae;

import net.hudup.core.alg.AllowNullTrainingSet;
import net.hudup.core.alg.ExecuteAsLearnAlgRemoteWrapper;
import net.hudup.core.logistic.BaseClass;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.ui.DescriptionDlg;
import net.hudup.core.logistic.ui.UIUtil;

/**
 * The class is a wrapper of remote convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@BaseClass //The annotation is very important which prevent Firer to instantiate the wrapper without referred remote object. This wrapper is not normal algorithm.
public class ConvVAERemoteWrapper extends ExecuteAsLearnAlgRemoteWrapper implements ConvVAE, ConvVAERemote, AllowNullTrainingSet {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with specified remote VAE algorithm.
	 * @param remoteVAE remote VAE algorithm.
	 */
	public ConvVAERemoteWrapper(ConvVAERemote remoteVAE) {
		super(remoteVAE);
	}

	
	/**
	 * Constructor with specified remote VAE algorithm and exclusive mode.
	 * @param remoteVAE remote VAE algorithm.
	 * @param exclusive exclusive mode.
	 */
	public ConvVAERemoteWrapper(ConvVAERemote remoteVAE, boolean exclusive) {
		super(remoteVAE, exclusive);
	}

	
	@Override
	public Inspector getInspector() {
		String desc = "";
		try {
			desc = getDescription();
		} catch (Exception e) {LogUtil.trace(e);}
		
		return new DescriptionDlg(UIUtil.getDialogForComponent(null), "Inspector", desc);
	}


}
