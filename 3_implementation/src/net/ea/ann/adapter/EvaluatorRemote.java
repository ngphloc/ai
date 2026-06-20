/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter;

import net.ea.ann.adapter.ui.EvalCompoundGUI;
import net.hudup.core.AccessPoint;
import net.hudup.core.Configuration;

/**
 * This class is a access point for generative remote evaluator.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class EvaluatorRemote implements AccessPoint {
	
	
	/**
	 * Default constructor.
	 */
	public EvaluatorRemote() {

	}

	
	/**
	 * The main method to start evaluator.
	 * @param args The argument parameter of main method. It contains command line arguments.
	 * @throws Exception if there is any error.
	 */
	public static void main(String[] args) throws Exception {
		new EvaluatorRemote().run(args);
	}

	
	@Override
	public void run(String[] args) {
		Configuration.RMI_MODE = true;
		EvalCompoundGUI.switchRemoteEvaluator(null, null);
	}

	
	@Override
	public String getName() {return "Genevaluator Remote";}

	
	@Override
	public String toString() {return getName();}
	
	
}
