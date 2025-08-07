/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.ui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.util.List;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.KeyStroke;
import javax.swing.WindowConstants;

import net.ea.ann.adapter.ui.ImagePathListExt;
import net.ea.ann.classifier.Classifier;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.raster.Raster;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.PropList;
import net.hudup.core.data.ui.PropDlg;
import net.hudup.core.logistic.ui.UIUtil;

/**
 * This class implements partially raster classifier user interface.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ClassifierUI extends JFrame {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Information dialog size.
	 */
	protected final static Dimension DIALOG_INFO_SIZE = new Dimension(300, 200);

	
	/**
	 * Information dialog size.
	 */
	protected final static Dimension FRAME_SIZE = new Dimension(800, 600);

	
	/**
	 * Splitter division location.
	 */
	protected final static int SPLITTER_DIVISION_LOCATION = 250;

	
	/**
	 * Base view.
	 */
	protected static final String BASE_VIEW = "base";

	
	/**
	 * Internal classifier.
	 */
	protected Classifier classifier = null;
	
	
	/**
	 * Configuration.
	 */
	protected DataConfig config = new DataConfig();
	
	
	/**
	 * View.
	 */
	protected String view = BASE_VIEW;
	
	
	/**
	 * Training view.
	 */
	protected JPanel trainView = null;
	
	
	/**
	 * Training rasters.
	 */
	protected ImagePathListExt trainRasters = new ImagePathListExt();
	
	
	/**
	 * Button to add training rasters.
	 */
	protected JButton btnTrainAddRasters = null;
	
	
	/**
	 * Button to clear training rasters.
	 */
	protected JButton btnTrainClearRasters = null;
	
	
	/**
	 * Classified view.
	 */
	protected JPanel classifyView = null;
	
	
	/**
	 * Classified view.
	 */
	protected ImagePathListExt classifyRasters = new ImagePathListExt();

	
	/**
	 * Saving classified rasters button.
	 */
	protected JButton btnClassifySave = null;
	
	
	/**
	 * Button to show information of  classified.
	 */
	protected JButton btnClassifyInfo = null;

	
	/**
	 * Tested view.
	 */
	protected JPanel testView = null;
	
	
	/**
	 * Tested rasters.
	 */
	protected ImagePathListExt testRasters = new ImagePathListExt();
	
	
	/**
	 * Button to add tested rasters.
	 */
	protected JButton btnTestAddRasters = null;
	
	
	/**
	 * Button to clear tested rasters.
	 */
	protected JButton btnTestClearRasters = null;
	
	
	/**
	 * Button to classify tested rasters.
	 */
	protected JButton btnTestClassifyRasters = null;

	
	/**
	 * Constructor with parent component and title.
	 * @param trainingRasters training rasters.
	 * @param view specified view.
	 */
	public ClassifierUI(List<Raster> trainingRasters, String view) {
		super("Rasters classifier");
		this.classifier = getClassifier();
		this.view = view != null ? view : BASE_VIEW;
		this.config.put(NetworkAbstract.LEARN_ONE_FIELD, NetworkAbstract.LEARN_ONE_DEFAULT);
		
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		setSize(FRAME_SIZE);
		setLocationRelativeTo(null);
		setLayout(new BorderLayout());
        
		JMenuBar mnuBar = createMenuBar();
	    if (mnuBar != null) setJMenuBar(mnuBar);

	    JPanel header = new JPanel(new BorderLayout());
        add(header, BorderLayout.NORTH);
        
        JPanel body = new JPanel(new BorderLayout());
        add(body, BorderLayout.CENTER);
        
        if (trainRasters != null) this.trainRasters.setRasters(view, trainingRasters);
        this.trainView = new JPanel(new BorderLayout());
        this.trainView.add(new JLabel("Training view"), BorderLayout.NORTH);
        this.trainView.add(new JScrollPane(this.trainRasters), BorderLayout.CENTER);
        //
		JPanel trainFooter = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		this.trainView.add(trainFooter, BorderLayout.SOUTH);
		//
		this.btnTrainAddRasters = UIUtil.makeIconButton(
			"add-16x16.png",
			"add_train_rasters", 
			"Add training rasters", 
			"Add training rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					addTrainRastersStarter();
				}
				
			});
		this.btnTrainAddRasters.setMargin(new Insets(0, 0 , 0, 0));
		trainFooter.add(this.btnTrainAddRasters);
		//
		this.btnTrainClearRasters = UIUtil.makeIconButton(
			"clear-16x16.png",
			"clear_train_rasters", 
			"Clear training rasters", 
			"Clear training rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					trainRasters.clearItems();
				}
				
			});
		this.btnTrainClearRasters.setMargin(new Insets(0, 0 , 0, 0));
		trainFooter.add(this.btnTrainClearRasters);
        
        this.classifyView = new JPanel(new BorderLayout());
        this.classifyView.add(new JLabel("Classified view"), BorderLayout.NORTH);
        this.classifyView.add(new JScrollPane(this.classifyRasters), BorderLayout.CENTER);
        //
		JPanel classifyFooter = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		this.classifyView.add(classifyFooter, BorderLayout.SOUTH);
        //
		this.btnClassifySave = UIUtil.makeIconButton(
			"save-16x16.png",
			"save_classify_rasters", 
			"Save classified rasters", 
			"Save classified rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					classifySave();
				}
				
			});
		this.btnClassifySave.setMargin(new Insets(0, 0 , 0, 0));
		classifyFooter.add(this.btnClassifySave);
		//
		this.btnClassifyInfo = UIUtil.makeIconButton(
			"info-16x16.png",
			"classify_info", 
			"Information about classification", 
			"Information about classification", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					classifyInfo();
				}
				
			});
		this.btnClassifyInfo.setMargin(new Insets(0, 0 , 0, 0));
		classifyFooter.add(this.btnClassifyInfo);

        this.testView = new JPanel(new BorderLayout());
        this.testView.add(new JLabel("Tested view"), BorderLayout.NORTH);
        this.testView.add(new JScrollPane(this.testRasters), BorderLayout.CENTER);
        //
		JPanel testFooter = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		this.testView.add(testFooter, BorderLayout.SOUTH);
		//
		this.btnTestAddRasters = UIUtil.makeIconButton(
			"add-16x16.png",
			"add_test_rasters", 
			"Add tested rasters", 
			"Add tested rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					addTestRastersStarter();
				}
				
			});
		this.btnTestAddRasters.setMargin(new Insets(0, 0 , 0, 0));
		testFooter.add(this.btnTestAddRasters);
		//
		this.btnTestClearRasters = UIUtil.makeIconButton(
			"clear-16x16.png",
			"clear_test_rasters", 
			"Clear tested rasters", 
			"Clear tested rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					testRasters.clearItems();
				}
				
			});
		this.btnTestClearRasters.setMargin(new Insets(0, 0 , 0, 0));
		testFooter.add(this.btnTestClearRasters);
		//
		this.btnTestClassifyRasters = net.ea.ann.core.UIUtil.makeIconButton(
			"classify-16x16.png",
			"classify_test_rasters", 
			"Classify tested rasters", 
			"Classify tested rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					classify();
				}
				
			});
		this.btnTestClassifyRasters.setMargin(new Insets(0, 0 , 0, 0));
		testFooter.add(this.btnTestClassifyRasters);

        JSplitPane trainSplitter = new JSplitPane(JSplitPane.VERTICAL_SPLIT, this.trainView, this.classifyView);
        trainSplitter.setOneTouchExpandable(true);
        trainSplitter.setDividerLocation(SPLITTER_DIVISION_LOCATION);
        
        JSplitPane mainSplitter = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, this.testView, trainSplitter);
        mainSplitter.setOneTouchExpandable(true);
        mainSplitter.setDividerLocation(SPLITTER_DIVISION_LOCATION);
        body.add(mainSplitter, BorderLayout.CENTER);

		JPanel footer = new JPanel();
		add(footer, BorderLayout.SOUTH);
		
		JButton btnClassify = new JButton("Classify");
		btnClassify.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				trainClassify();
			}
			
		});
		footer.add(btnClassify);

		JButton close = new JButton("Close");
		close.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				dispose();
			}
			
		});
		footer.add(close);
	}
	
	
	/**
	 * Creating main menu bar.
	 * @return main menu bar.
	 */
	protected JMenuBar createMenuBar() {
		JMenuBar mnBar = new JMenuBar();
		
		JMenu mnFile = new JMenu("File");
		mnFile.setMnemonic('f');

		JMenuItem mniLoadTrainRasters = new JMenuItem(
			new AbstractAction("Load training rasters") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					loadTrainDir();
				}
				
			});
		mniLoadTrainRasters.setMnemonic('l');
		mniLoadTrainRasters.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, InputEvent.CTRL_DOWN_MASK));
		mnFile.add(mniLoadTrainRasters);

		JMenuItem mniLoadTestRasters = new JMenuItem(
			new AbstractAction("Load tested rasters") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					loadTestDir();
				}
				
			});
		mniLoadTestRasters.setMnemonic('t');
		mniLoadTestRasters.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_T, InputEvent.CTRL_DOWN_MASK));
		mnFile.add(mniLoadTestRasters);
		
		if (mnFile.getMenuComponentCount() > 0) mnFile.addSeparator();

		JMenuItem mniSetting = new JMenuItem(
			new AbstractAction("Setting") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					setting();
				}
				
			});
		mniSetting.setMnemonic('s');
		mniSetting.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S, InputEvent.CTRL_DOWN_MASK));
		mnFile.add(mniSetting);

		if (mnFile.getMenuComponentCount() > 0) mnFile.addSeparator();

		JMenuItem mniExit = new JMenuItem(
			new AbstractAction("Exit") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					dispose();
				}
				
			});
		mniExit.setMnemonic('x');
		mniExit.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_X, InputEvent.ALT_DOWN_MASK));
		mnFile.add(mniExit);

		if (mnFile.getMenuComponentCount() > 0) mnBar.add(mnFile);
		
		return mnBar.getMenuCount() > 0 ? mnBar : null;
	}

	
	/**
	 * Getting classifier.
	 * @return specified classifier.
	 */
	protected abstract Classifier getClassifier();

	
	/**
	 * Loading training directory.
	 */
	protected abstract void loadTrainDir();

	
	/**
	 * Start to add training rasters.
	 */
	protected abstract void addTrainRastersStarter();

	
	/**
	 * Loading tested directory.
	 */
	protected abstract void loadTestDir();

	
	/**
	 * Start to add tested rasters.
	 */
	protected abstract void addTestRastersStarter();
	
	
	/**
	 * Training classifier and classifying.
	 */
	private void trainClassify() {
		if (classifier != null) classifier = getClassifier();
		if (classifier == null) return;
		
		List<Raster> sample = trainRasters.queryItemRasters();
		try {
			if (config.getAsBoolean(NetworkAbstract.LEARN_ONE_FIELD))
				classifier.learnRasterOne(sample);
			else
				classifier.learnRaster(sample);
			classify();
		} catch (Throwable e) {Util.trace(e);}
	}
	
	
	/**
	 * Classifying.
	 */
	private void classify() {
		if (classifier == null) {
			JOptionPane.showMessageDialog(this, "Null classifier", "Null classifier", JOptionPane.ERROR_MESSAGE);
			return;
		}
		List<Raster> test = testRasters.queryItemRasters();
		if (test.size() == 0) {
			JOptionPane.showMessageDialog(this, "Empty testing set", "Empty testing set", JOptionPane.ERROR_MESSAGE);
			return;
		}
		classifyRasters.clearItems();
		
		List<Raster> results = Util.newList(0);
		try {
			results = classifier.classify(test);
		} catch (Throwable e) {Util.trace(e);}
		classifyRasters.setRasters("classified", results);
	}
	
	
	/**
	 * Saving classified rasters.
	 */
	private void classifySave() {
		JOptionPane.showMessageDialog(this, "Not implemented yet", "Not implemented yet", JOptionPane.ERROR_MESSAGE);
	}
	
	
	/**
	 * Information of classification.
	 */
	private void classifyInfo() {
		JOptionPane.showMessageDialog(this, "Not implemented yet", "Not implemented yet", JOptionPane.ERROR_MESSAGE);
	}
	
	
	/**
	 * Setting some parameters.
	 */
	private void setting() {
		if (classifier == null) {
			JOptionPane.showMessageDialog(this, "Null classifier", "Null classifier", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (!(classifier instanceof Network)) {
			JOptionPane.showMessageDialog(this, "No configuration", "No configuration", JOptionPane.WARNING_MESSAGE);
			return;
		}
		
		NetworkConfig annConfig = null;
		try {
			annConfig = ((Network)classifier).getConfig();
			DataConfig classiferConfig = net.ea.ann.adapter.Util.toConfig(annConfig);
			this.config.putAll(classiferConfig);
		} catch (Throwable e) {Util.trace(e);}
		if (config == null) {
			JOptionPane.showMessageDialog(this, "Null configuration", "Null configuration", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		PropDlg cfg = new PropDlg(this, this.config, "classfier");
		PropList result = cfg.getResult();
		if (result == null) return;
		
		this.config.putAll(result);
		annConfig = net.ea.ann.adapter.Util.transferToANNConfig(config);
		try {
			((Network)classifier).setConfig(annConfig);
		} catch (Throwable e) {Util.trace(e);}
	}
	
	
}
