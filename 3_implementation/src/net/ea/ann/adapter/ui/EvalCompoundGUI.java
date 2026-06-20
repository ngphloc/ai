/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.ui;

import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;

import javax.swing.AbstractAction;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.KeyStroke;

import net.hudup.core.client.ConnectInfo;
import net.hudup.core.evaluate.Evaluator;
import net.hudup.core.evaluate.ui.EvaluateGUIData;

/**
 * This class is an extension of compound evaluation GUI for generative model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class EvalCompoundGUI extends net.hudup.evaluate.ui.EvalCompoundGUI {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with specified evaluator.
	 * @param evaluator specified evaluator.
	 */
	public EvalCompoundGUI(Evaluator evaluator) {
		super(evaluator);
	}


	/**
	 * Constructor with specified evaluator and connection information.
	 * @param evaluator specified evaluator.
	 * @param connectInfo connection information.
	 */
	public EvalCompoundGUI(Evaluator evaluator, ConnectInfo connectInfo) {
		super(evaluator, connectInfo);
	}

	
	/**
	 * Constructor with specified evaluator.
	 * @param evaluator specified evaluator.
	 * @param connectInfo connection information.
	 * @param referredData evaluator GUI data.
	 */
	public EvalCompoundGUI(Evaluator evaluator, ConnectInfo connectInfo, EvaluateGUIData referredData) {
		super(evaluator, connectInfo, referredData);
	}


	@Override
	protected BatchEvaluateGUI createBatchEvaluateGUI(Evaluator evaluator, ConnectInfo connectInfo, EvaluateGUIData referredGUIData) {
		return new BatchEvaluateGUI(evaluator, connectInfo, referredGUIData);
	}

	
	@Override
	protected void addMoreTools(JMenu tools) {
		super.addMoreTools(tools);
		if (tools.getMenuComponentCount() > 0) tools.addSeparator();

		JMenuItem mniRegWithoutEv = new JMenuItem(
			new AbstractAction("Reg. delegators w/o eval.") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					registerAlgsWithoutEvaluation();
				}
			});
		mniRegWithoutEv.setMnemonic('d');
		mniRegWithoutEv.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_D, InputEvent.CTRL_DOWN_MASK));
		tools.add(mniRegWithoutEv);
	}


	/**
	 * Registering algorithms without evaluation.
	 */
	private void registerAlgsWithoutEvaluation() {
		if (!(this.batchEvaluateGUI instanceof BatchEvaluateGUI)) return;
		BatchEvaluateGUI evGUI = (BatchEvaluateGUI)this.batchEvaluateGUI;
		boolean registered = evGUI.remoteStartWithoutEvaluation();
		if (registered)
			JOptionPane.showMessageDialog(this, "Successful to register delegators without evaluation", "Success", JOptionPane.INFORMATION_MESSAGE);
		else
			JOptionPane.showMessageDialog(this, "Not register any delegators without evaluation. \nMaybe algs are not delegators. \nMaybe all delegators registered or not existing delegators.", "Notice", JOptionPane.WARNING_MESSAGE);
	}
	
	
	@Override
	protected EvalCompoundGUIShower newShower() {return shower();}


	/**
	 * Creating compound GUI shower.
	 * @return compound GUI shower.
	 */
	protected static EvalCompoundGUIShower shower() {
		return new EvalCompoundGUIShower() {
			@Override
			public void show(Evaluator evaluator, ConnectInfo connectInfo, EvaluateGUIData referredData) {
				new EvalCompoundGUI(evaluator, connectInfo, referredData);
			}
		};
	}

	
	/**
	 * Switching evaluator.
	 * @param selectedEvName selected evaluator name.
	 * @param oldGUI old GUI which can be null.
	 */
	public static void switchRemoteEvaluator(String selectedEvName, Window oldGUI) {
		net.hudup.evaluate.ui.EvalCompoundGUI.switchRemoteEvaluator(selectedEvName, oldGUI, shower());
	}

	
	/**
	 * Staring the particular evaluator selected by user.
	 * @param evaluator particular evaluator selected by user.
	 * @param connectInfo connection information.
	 * @param referredData evaluator GUI data.
	 * @param oldGUI old GUI.
	 */
	public static void run(Evaluator evaluator, ConnectInfo connectInfo, EvaluateGUIData referredData, Window oldGUI) {
		net.hudup.evaluate.ui.EvalCompoundGUI.run(evaluator, connectInfo, referredData, oldGUI, shower());
	}


}
