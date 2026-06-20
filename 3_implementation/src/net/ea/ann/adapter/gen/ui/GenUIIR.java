/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.ui;

import javax.swing.JMenuBar;
import javax.swing.JOptionPane;

import net.ea.ann.adapter.gen.GenModelRemote;
import net.ea.ann.adapter.gen.IRRemote;
import net.ea.ann.adapter.gen.beans.IRProxyNCA;
import net.ea.ann.core.Util;
import net.hudup.core.logistic.NetUtil;

/**
 * This class implements information retrieval (IR) user interface based on generative AI user interface.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenUIIR extends GenUI {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Imagic sub-project.
	 */
	protected static final String IRAGIC = "Iragic";

	
	/**
	 * Classified view label text.
	 */
	protected static final String SEARCHED_LABEL_TEXT = "Searching view";

	
	/**
	 * Default constructor.
	 */
	private GenUIIR() {
		super();
	}
	
	
	/**
	 * Constructor with remote information retrieval (IR) system and exclusive mode.
	 * @param gm remote information retrieval (IR) system.
	 * @param exclusive exclusive mode.
	 */
	public GenUIIR(IRRemote gm, boolean exclusive) {
		this();
		this.gm = gm;
		this.exclusive = exclusive;
	    
		JMenuBar mnuBar = createMenuBar();
	    if (mnuBar != null) setJMenuBar(mnuBar);
	    initGUI();
	    
	    try {
	    	if (isRemoteGenModel()) {
	    		this.exportedStub = NetUtil.RegistryRemote.export(this); //Exporting at random port.
	    		if (this.exportedStub != null) this.gm.addSetupListener(this);
	    	}
	    	else
	    		this.gm.addSetupListener(this);
	    } catch (Throwable e) {Util.trace(e);}
	}

	
	/**
	 * Constructor with remote information retrieval (IR) system.
	 * @param gm classifier model.
	 */
	public GenUIIR(IRRemote gm) {
		this(gm, false);
	}

	
	@Override
	void initGUI() {
		super.initGUI();
		this.setTitle("Information retrieval (IR) system \"" + IRAGIC + "\" of project \"" + MAGICONIC + "\"");
	}


	@Override
	void updateControls() {
		super.updateControls();
		
		this.chkLoad3D.setEnabled(false);
		this.chkRecover.setSelected(true);
		this.chkRecover.setEnabled(false);
		this.chkAllowAdd.setSelected(true);
		this.chkAllowAdd.setEnabled(false);
		this.txtGenNum.setEnabled(false);
		this.chkGenAutoSave.setEnabled(false);
		this.chkRecoverToTest.setSelected(true);
		this.chkRecoverToTest.setEnabled(false);
		
		this.txtGenNum.setValue(1);
		this.lblGen.setText(SEARCHED_LABEL_TEXT);
		this.btnGen.setText("Search");
	}


	@Override
	void retrieve() {
		JOptionPane.showMessageDialog(this, "Not show retrieval GUI.\nBecause this GUI is retrieval GUI", "This GUI is retrieval GUI", JOptionPane.INFORMATION_MESSAGE);
	}


	@Override
	void recover() {
		super.recover();
	}

	
	@Override
	protected GenUI queryLocalGenModel(GenModelRemote initialGM, GenUI initialUI) {
		return queryLocalGenModel((IRRemote)initialGM, (GenUIIR)initialUI);
	}


	/**
	 * Querying local IR.
	 * @param initialGM initial IR.
	 * @param initialUI IR UI.
	 * @return local IR.
	 */
	private static GenUIIR queryLocalGenModel(IRRemote initialGM, GenUIIR initialUI) {
		return queryLocalGenModel(IRRemote.class, initialGM, initialUI);
	}

	
	/**
	 * Querying local IR.
	 * @param gmClass IR class.
	 * @param initialGM initial IR.
	 * @param initialUI IR UI.
	 * @param <T> IR type.
	 * @return local generative model.
	 */
	private static <T extends IRRemote> GenUIIR queryLocalGenModel(Class<T> gmClass, T initialGM, GenUIIR initialUI) {
		return (GenUIIR) GenUI.queryLocalGenModel(gmClass, initialGM, initialUI, new GenUI.GenUICreator<T>() {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public GenUI create(T gm) {return new GenUIIR(gm, false);}

			@SuppressWarnings("unchecked")
			@Override
			public T[] getDefaultGMs() {
				T[] defaultGMs = Util.newArray(gmClass, 1);
				defaultGMs[0] = (T)new IRProxyNCA();
				return defaultGMs;
			}
			
		});
		
	}
	
	
	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		String arg = null;
		if (args != null && args.length > 0 && args[0] != null && !args[0].isEmpty()) {
			arg = args[0].trim();
			arg = arg.replaceAll("-", "");
			if (arg.isBlank() || arg.isEmpty()) arg = null;
		}

		if (arg == null) {
			GenUIIR genUIIR = queryLocalGenModel(new IRProxyNCA(), null);
			if (genUIIR != null) genUIIR.setVisible(true);
		}
		else {
			GenUIIR genUIIR = queryLocalGenModel(new IRProxyNCA(), null);
			if (genUIIR != null) genUIIR.setVisible(true);
		}
	}


}
