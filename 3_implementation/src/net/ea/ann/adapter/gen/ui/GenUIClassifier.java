/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.ui;

import javax.swing.JMenuBar;

import net.ea.ann.adapter.gen.GenModelRemote;
import net.ea.ann.core.Util;

/**
 * This class implements classifier user interface based on generative AI user interface.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenUIClassifier extends GenUI {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Imagic sub-project.
	 */
	protected static final String IMAGIC = "Imagic";

	
	/**
	 * Classified view label text.
	 */
	protected static final String CLASSIFIED_LABEL_TEXT = "Classified view";

	
	/**
	 * Default constructor.
	 */
	private GenUIClassifier() {
		super();
	}
	
	
	/**
	 * Constructor with generative model and exclusive mode.
	 * @param gm generative model.
	 * @param exclusive exclusive mode.
	 */
	public GenUIClassifier(GenModelRemote gm, boolean exclusive) {
		this();
		this.gm = gm;
		this.exclusive = exclusive;
	    
		JMenuBar mnuBar = createMenuBar();
	    if (mnuBar != null) setJMenuBar(mnuBar);
	    initGUI();
	    
	    try {
	    	if (gm != null && isLocalGenModel()) gm.addSetupListener(this);
	    } catch (Throwable e) {Util.trace(e);}
	}

	
	/**
	 * Constructor with generative model.
	 * @param gm generative model.
	 */
	public GenUIClassifier(GenModelRemote gm) {
		this(gm, false);
	}

	
	@Override
	void initGUI() {
		super.initGUI();
		this.setTitle("Classifier \"" + IMAGIC + "\" of project \"" + MAGICONIC + "\"");
	}


	@Override
	void reset() {
		super.reset();
		
		this.chkRecover.setSelected(true);
		this.chkAllowAdd.setSelected(true);
		this.chkRecoverToTest.setSelected(true);
		
		this.chkRecover.setEnabled(false);
		this.chkAllowAdd.setEnabled(false);
		this.chkRecoverToTest.setEnabled(false);
		
		this.lblGen.setText(CLASSIFIED_LABEL_TEXT);
		this.btnGen.setText("Classify");
	}


	@Override
	void updateControls() {
		super.updateControls();
		
		this.chkLoad3D.setEnabled(false);
		this.chkRecover.setEnabled(false);
		this.chkAllowAdd.setEnabled(false);
		this.txtGenNum.setEnabled(false);
		this.chkGenAutoSave.setEnabled(false);
		this.chkRecoverToTest.setEnabled(false);
	}


	@Override
	void recover() {
		super.recover();
	}


	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		GenUIClassifier classifier = new GenUIClassifier(null);
		classifier.setVisible(true);
	}


}
