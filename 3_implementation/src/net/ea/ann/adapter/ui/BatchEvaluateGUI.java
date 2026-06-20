/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.ui;

import java.rmi.RemoteException;

import net.ea.ann.core.Util;
import net.hudup.core.client.ConnectInfo;
import net.hudup.core.evaluate.Evaluator;
import net.hudup.core.evaluate.ui.EvaluateGUIData;

/**
 * This class is an extension of batch evaluation GUI for generative model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class BatchEvaluateGUI extends net.hudup.evaluate.ui.BatchEvaluateGUI {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with local evaluator.
	 * @param evaluator local evaluator.
	 */
	public BatchEvaluateGUI(Evaluator evaluator) {
		super(evaluator);
	}


	/**
	 * Constructor with specified evaluator and bound URI.
	 * @param evaluator specified evaluator.
	 * @param connectInfo connection information.
	 */
	public BatchEvaluateGUI(Evaluator evaluator, ConnectInfo connectInfo) {
		super(evaluator, connectInfo);
	}


	/**
	 * Constructor with local evaluator and referred GUI data.
	 * @param evaluator local evaluator.
	 * @param referredGUIData referred GUI data.
	 */
	public BatchEvaluateGUI(Evaluator evaluator, EvaluateGUIData referredGUIData) {
		super(evaluator, referredGUIData);
	}

	
	/**
	 * Constructor with specified evaluator, bound URI, and GUI data.
	 * @param evaluator specified evaluator.
	 * @param connectInfo connection information.
	 * @param referredGUIData referred GUI data.
	 */
	public BatchEvaluateGUI(Evaluator evaluator, ConnectInfo connectInfo, EvaluateGUIData referredGUIData) {
		super(evaluator, connectInfo, referredGUIData);
		remoteStartWithoutEvaluation();
	}


	/**
	 * Register list of algorithms without evaluation.
	 * @return registering is successfully.
	 * @throws RemoteException if any error occurs.
	 */
	protected boolean remoteStartWithoutEvaluation() {
		if (!(this.evaluator instanceof net.ea.ann.adapter.Evaluator)) return false;
		net.ea.ann.adapter.Evaluator ge = (net.ea.ann.adapter.Evaluator)this.evaluator;
		
		try {
			return ge.remoteStartWithoutEvaluation(lbAlgs.getAlgNameList());
		} catch (Throwable e) {Util.trace(e);}
		return false;
	}
	
	
}
