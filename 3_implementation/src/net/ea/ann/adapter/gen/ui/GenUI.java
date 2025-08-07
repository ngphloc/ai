/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.ui;

import java.awt.BorderLayout;
import java.awt.Dialog.ModalityType;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.rmi.RemoteException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

import javax.swing.AbstractAction;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextField;
import javax.swing.KeyStroke;
import javax.swing.WindowConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.text.NumberFormatter;

import net.ea.ann.adapter.gen.GenModel;
import net.ea.ann.adapter.gen.GenModelAbstract;
import net.ea.ann.adapter.gen.GenModelRemote;
import net.ea.ann.adapter.gen.GenModelRemoteAssoc;
import net.ea.ann.adapter.gen.GenModelRemoteWrapper;
import net.ea.ann.adapter.gen.beans.AVA;
import net.ea.ann.adapter.gen.beans.AVAExt;
import net.ea.ann.adapter.gen.beans.GAN;
import net.ea.ann.adapter.gen.beans.VAE;
import net.ea.ann.adapter.ui.ImagePathListExt;
import net.ea.ann.classifier.Classifier;
import net.ea.ann.classifier.StackClassifier;
import net.ea.ann.conv.Content;
import net.ea.ann.conv.ContentAssoc;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.gen.FeatureGetter;
import net.ea.ann.gen.GenModel.G;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.ImageAssoc;
import net.ea.ann.raster.ImageWrapper;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterWrapper;
import net.hudup.core.Constants;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgListener;
import net.hudup.core.alg.ui.AlgConfigDlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.PropList;
import net.hudup.core.data.Wrapper;
import net.hudup.core.logistic.AbstractRunner;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.ui.JImageList.ImageListItem;
import net.hudup.core.logistic.ui.JImageList.ImagePathList;
import net.hudup.core.logistic.ui.StartDlg;
import net.hudup.core.logistic.ui.TextArea;
import net.hudup.core.logistic.ui.TextField;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.core.logistic.ui.WaitDialog;
import net.hudup.core.logistic.ui.WaitDialog.Task;

/**
 * This class is an user interface for generative model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenUI extends JFrame implements Inspector, SetupAlgListener {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Magiconic project.
	 */
	protected static final String MAGICONIC = "Magiconic";
	
	
	/**
	 * AVA sub-project.
	 */
	protected static final String AVA = "AVA";

	
	/**
	 * Flag to indicate whether task is running in background.
	 */
	protected static final boolean TASK_BACKGROUND = false;
	
	
	/**
	 * Default debug mode.
	 */
	protected static final boolean DEBUG = true;
	
	
	/**
	 * Minimum icon size.
	 */
	public static final int ICON_MINSIZE = ImageListItem.ICON_MINSIZE;

	
	/**
	 * Maximum icon size.
	 */
	public static final int ICON_MAXSIZE = 8*ICON_MINSIZE;

	
	/**
	 * Flag to indicate store image in image list.
	 */
	public static final boolean IMAGELIST_STORE_IMAGE = false;

	
	/**
	 * Working base directory.
	 */
	public static final String WORKING_BASE = Constants.WORKING_DIRECTORY + "/base";
	
	
	/**
	 * Working generation directory.
	 */
	public static final String WORKING_GEN = Constants.WORKING_DIRECTORY + "/gen";

	
	/**
	 * Working recovering directory.
	 */
	public static final String WORKING_RECOVER = Constants.WORKING_DIRECTORY + "/recover";

	
	/**
	 * Test directory.
	 */
	public static final String WORKING_TEST = Constants.WORKING_DIRECTORY + "/test";

	
	/**
	 * Test result directory.
	 */
	public static final String WORKING_TESTRESULT = Constants.WORKING_DIRECTORY + "/testresult";

	
	/**
	 * Test file name.
	 */
	public static final String TEST_FILE_NAME = "TestResult.txt";

	
	/**
	 * Recovering view label text.
	 */
	protected static final String RECOVERING_LABEL_TEXT = "Recovering view";
	
	
	/**
	 * Test view label text.
	 */
	protected static final String TEST_LABEL_TEXT = "Test view";

	
	/**
	 * Base view label text.
	 */
	protected static final String BASE_LABEL_TEXT = "Base view";

	
	/**
	 * Generation view label text.
	 */
	protected static final String GEN_LABEL_TEXT = "Generated view";

	
	/**
	 * Recovered view label text.
	 */
	protected static final String RECOVERED_LABEL_TEXT = "Recovered view";

	
	/**
	 * Base view.
	 */
	protected static final String BASE_VIEW = "base";

	
	/**
	 * Generation view.
	 */
	protected static final String GEN_VIEW = "gen";

	
	/**
	 * Recovering view.
	 */
	protected static final String RECOVER_VIEW = "recover";

	
	/**
	 * Views.
	 */
	protected static String[] VIEWS = new String[] {BASE_VIEW, GEN_VIEW, RECOVER_VIEW};

	
	/**
	 * Flag to add rasters by normal way.
	 */
	protected final static String ADD_RASTERS_NORMAL = "normal";
	
	
	/**
	 * Flag to add rasters from CIFAR file.
	 */
	protected final static String ADD_RASTERS_CIFAR = "cifar";

	
	/**
	 * Flag to add rasters from folders.
	 */
	protected final static String ADD_RASTERS_FOLDERS = "folders";

	
	/**
	 * Flag array to add rasters.
	 */
	protected static String[] ADD_RASTERS = new String[] {ADD_RASTERS_CIFAR, ADD_RASTERS_FOLDERS, ADD_RASTERS_NORMAL};
	
	
	/**
	 * Loading rasters always.
	 */
	protected static boolean LOAD_RASTER_ALWAYS = false;

	
	/**
	 * Information dialog size.
	 */
	protected final static Dimension DIALOG_INFO_SIZE = new Dimension(300, 200);
	
	
	/**
	 * Frame size.
	 */
	protected final static Dimension FRAME_SIZE = new Dimension(800, 600);

	
	/**
	 * Splitter division location.
	 */
	protected final static int SPLITTER_DIVISION_LOCATION = 250;
	
	
	/**
	 * This class is the extended image list with image directory.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected class ImagePathListGen extends ImagePathListExt {
		
    	/**
    	 * Serial version UID for serializable class. 
    	 */
    	private static final long serialVersionUID = 1L;

    	/**
    	 * Default constructor.
    	 */
		public ImagePathListGen() {
			super();
		}

		/**
		 * Constructor from image directory.
		 * @param imageDir image directory.
		 * @param iconSize icon size.
		 * @param storeImage flag to indicate whether to store image.
		 */
		public ImagePathListGen(Path imageDir, int iconSize, boolean storeImage) {
			super(imageDir, iconSize, storeImage);
		}

		/**
		 * Constructor from image directory.
		 * @param imageDir image directory.
		 * @param iconSize icon size.
		 */
		public ImagePathListGen(Path imageDir, int iconSize) {
			super(imageDir, iconSize);
		}

		/**
		 * Constructor from image directory.
		 * @param imageDir image directory.
		 */
		public ImagePathListGen(Path imageDir) {
			super(imageDir);
		}

		@Override
		protected int getIconSize() {
			return (iconSize = imageListIconSize);
		}
		
		@Override
		protected List<Raster> loadRasters(Path dirOrFile) {
			return getGenUI().loadRasters(dirOrFile);
		}

		@Override
		protected void addToContextMenu(JPopupMenu contextMenu) {
			super.addToContextMenu(contextMenu);
			if (contextMenu == null) return;
			
			JMenuItem mniFeature = new JMenuItem(
				new AbstractAction("Feature") {
					
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public void actionPerformed(ActionEvent e) {
						viewUnifiedContent();
					}
					
				});
			if (gm != null && getSelectedIndex() >= 0) contextMenu.add(mniFeature);
		}

		/**
		 * View unified content.
		 */
		private void viewUnifiedContent() {
			if (gm == null) return;
			int selectedIndex = getSelectedIndex();
			if (selectedIndex < 0) return;
			Raster raster = queryItemRaster(selectedIndex);
			if (raster == null) {
				JOptionPane.showMessageDialog(this, "No raster", "No raster", JOptionPane.WARNING_MESSAGE);
				return;
			}
			
			Object parameter = null;
			try {
				parameter = gm.getParameter();
			} catch (Throwable e) {Util.trace(e);}
			if ((parameter == null) || !(parameter instanceof FeatureGetter)) {
				JOptionPane.showMessageDialog(this, "Cannot retrieve feature", "Cannot retrieve feature", JOptionPane.ERROR_MESSAGE);
				return;
			}
		
			FeatureGetter featureGetter = (FeatureGetter)parameter;
			Content feature = null;
			try {
				feature = featureGetter.getFeature(raster);
			} catch (Throwable e) {Util.trace(e);}
			if (feature == null) {
				JOptionPane.showMessageDialog(this, "Cannot retrieve feature", "Cannot retrieve feature", JOptionPane.ERROR_MESSAGE);
				return;
			}
			
			boolean isNorm = new GenModelRemoteAssoc(gm).isNorm();
			int defaultAlpha = new GenModelRemoteAssoc(gm).getDefaultAlpha();
			Raster featureRaster = new ContentAssoc(feature).getRaster(isNorm, defaultAlpha);
			if (featureRaster == null) {
				JOptionPane.showMessageDialog(this, "Cannot retrieve feature", "Cannot retrieve feature", JOptionPane.ERROR_MESSAGE);
				return;
			}
			UIUtil.viewImage(featureRaster.getRepImage(), false, this);
		}
		
	}
	
	
	/**
	 * Internal generative model.
	 */
	protected GenModelRemote gm = null;
	
	
	/**
	 * Task runner.
	 */
	protected AbstractRunner runner = null;
	
	
	/**
	 * Exclusive mode.
	 */
	protected boolean exclusive = false;
	
	
	/**
	 * Icon size in image list.
	 */
	protected int imageListIconSize = ICON_MINSIZE;
	
	
	/**
	 * Flag to indicate whether to store images in image list.
	 */
	protected boolean imageListStoreImage = IMAGELIST_STORE_IMAGE;
	
	
	/**
	 * Flag to indicate whether task is running in background.
	 */
	protected boolean taskBackground = TASK_BACKGROUND;
	
	
	/**
	 * Debug mode.
	 */
	protected boolean debug = DEBUG;
	
	
	/**
	 * Result path.
	 */
	protected Path resultDir = null;
	
	
	/**
	 * Testing error.
	 */
	protected double error = 0;
	
	
	/**
	 * Navigator panel.
	 */
	protected JPanel navigator;
	
	
	/**
	 * Label of model name.
	 */
	protected JLabel lblModelName;
	
	
	/**
	 * Text field of base directory.
	 */
	protected TextField txtBaseDir;
	
	
	/**
	 * Base directory browsing button.
	 */
	protected JButton btnBaseDirBrowse;
	
	
	/**
	 * Base directory loading button.
	 */
	protected JButton btnBaseDirLoad;
	
	
	/**
	 * Text field of generation directory.
	 */
	protected TextField txtGenDir;

	
	/**
	 * Generated directory browsing button.
	 */
	protected JButton btnGenDirBrowse;
	
	
	/**
	 * Generated directory loading button.
	 */
	protected JButton btnGenDirLoad;

	
	/**
	 * Loading 3D rasters check box.
	 */
	protected JCheckBox chkLoad3D;

	
	/**
	 * Recovering check box.
	 */
	protected JCheckBox chkRecover;

	
	/**
	 * Allowing to add rasters check box.
	 */
	protected JCheckBox chkAllowAdd;

	
	/**
	 * Text field to the number of generated rasters.
	 */
	protected JFormattedTextField txtGenNum = null;

	
	/**
	 * Auto-saving check box.
	 */
	protected JCheckBox chkGenAutoSave;
	
	
	/**
	 * Configuring button.
	 */
	protected JButton btnConfig;

	
	/**
	 * Applying parameters button.
	 */
	protected JButton btnApplyParams;
	
	
	/**
	 * Refreshing parameters button.
	 */
	protected JButton btnRefreshParams;

	
	/**
	 * Resetting parameters button.
	 */
	protected JButton btnResetParams;
	
	
	/**
	 * Base view.
	 */
	protected JPanel baseView;

	
	/**
	 * Base view label.
	 */
	protected JLabel lblBase;
	
	
	/**
	 * List box of base rasters.
	 */
	protected ImagePathListGen baseRasters = new ImagePathListGen();
	
	
	/**
	 * Adding base rasters button.
	 */
	protected JButton btnBaseAddRasters;

	
	/**
	 * Clearing base rasters button.
	 */
	protected JButton btnBaseClearRasters;

	
	/**
	 * Generation view.
	 */
	protected JPanel genView;

	
	/**
	 * Generation view label.
	 */
	protected JLabel lblGen;
	
	
	/**
	 * List box of generated rasters.
	 */
	protected ImagePathListGen genRasters = new ImagePathListGen();

	
	/**
	 * Saving generated rasters button.
	 */
	protected JButton btnGenSave;
	
	
	/**
	 * Clearing generated rasters button.
	 */
	protected JButton btnGenClearRasters;

	
	/**
	 * Generation information button.
	 */
	protected JButton btnGenInfo;

	
	/**
	 * Generation button.
	 */
	protected JButton btnGenGen;

	
	/**
	 * Recovery view.
	 */
	protected JPanel recoverView;
	
	
	/**
	 * Recovery view label.
	 */
	protected JLabel lblRecover;
	
	
	/**
	 * Text field of recovery directory.
	 */
	protected TextField txtRecoverDir;

	
	/**
	 * Recovery directory browsing button.
	 */
	protected JButton btnRecoverDirBrowse;
	
	
	/**
	 * Recovery directory loading button.
	 */
	protected JButton btnRecoverDirLoad;

	
	/**
	 * List box of recovery rasters.
	 */
	protected ImagePathListGen recoverRasters = new ImagePathListGen();

	
	/**
	 * Adding recovering rasters button.
	 */
	protected JButton btnRecoverAddRasters;
	
	
	/**
	 * Clearing recovering rasters button.
	 */
	protected JButton btnRecoverClearRasters;

	
	/**
	 * Check box to change recovering directory to testing directory.
	 */
	protected JCheckBox chkRecoverToTest;

	
	/**
	 * Resetting button.
	 */
	protected JButton btnReset;
	
	
	/**
	 * Generation button.
	 */
	protected JButton btnGen;
	
	
	/**
	 * Running progress bar.
	 */
	protected JProgressBar prgRunning;
	
	
	/**
	 * Default constructor.
	 */
	GenUI() {
		super("Generator \"" + AVA + "\" of project \"" + MAGICONIC + "\"");
	    this.resultDir = Paths.get(WORKING_TESTRESULT);
		
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		java.awt.Image image = UIUtil.getImage("generate-sonic-32x32.png");
        if (image != null) setIconImage(image);
        
		setSize(FRAME_SIZE);
		setLocationRelativeTo(null);
		setLayout(new BorderLayout());
	}
	
	
	/**
	 * Constructor with generative model and exclusive mode.
	 * @param gm generative model.
	 * @param exclusive exclusive mode.
	 */
	public GenUI(GenModelRemote gm, boolean exclusive) {
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
	public GenUI(GenModelRemote gm) {
		this(gm, false);
	}
	
	
	/**
	 * Getting this UI.
	 * @return this generation UI.
	 */
	private GenUI getGenUI() {
		return this;
	}
	
	
	/**
	 * Checking whether the internal generative model is local.
	 * @return whether the internal generative model is local.
	 */
	boolean isLocalGenModel() {
		return gm != null && gm instanceof GenModel;
	}
	
	
	/**
	 * Initialize user interface.
	 */
	void initGUI() {
		JPanel header = createHeader();
		if (header != null) add(header, BorderLayout.NORTH);
		
		this.navigator = createNavigator();

		if (this.navigator == null) {
			JPanel body = createBody();
			if (body != null) add(body, BorderLayout.CENTER);
		}
		else {
			JPanel body = createBody();
			if (body != null) {
				JSplitPane splitter = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, this.navigator, body);
				splitter.setOneTouchExpandable(true);
				splitter.setDividerLocation(SPLITTER_DIVISION_LOCATION);
				add(splitter, BorderLayout.CENTER);
			}
		}
		
		JPanel footer = createFooter();
		if (footer != null) add(footer, BorderLayout.SOUTH);
		
		reset();
	}
	
	
	/**
	 * Enabling controls
	 * @param enabled enabling flag.
	 */
	private void enableControls(boolean enabled) {
		lblModelName.setEnabled(enabled);
		txtBaseDir.setEnabled(enabled);
		txtGenDir.setEnabled(enabled);
		btnBaseDirBrowse.setEnabled(enabled);
		btnBaseDirLoad.setEnabled(enabled);
		btnGenDirBrowse.setEnabled(enabled);
		btnGenDirLoad.setEnabled(enabled);
		chkLoad3D.setEnabled(enabled);
		chkRecover.setEnabled(enabled);
		chkAllowAdd.setEnabled(enabled);
		txtGenNum.setEnabled(enabled);
		chkGenAutoSave.setEnabled(enabled);
		btnConfig.setEnabled(enabled);
		
		btnApplyParams.setEnabled(enabled);
		btnRefreshParams.setEnabled(enabled);
		btnResetParams.setEnabled(enabled);
		
		btnReset.setEnabled(enabled);
		btnGen.setEnabled(enabled);
		
		baseRasters.setEnabled(enabled);
		btnBaseAddRasters.setEnabled(enabled);
		btnBaseClearRasters.setEnabled(enabled);

		genRasters.setEnabled(enabled);
		btnGenSave.setEnabled(enabled);
		btnGenClearRasters.setEnabled(enabled);
		btnGenInfo.setEnabled(enabled);
		btnGenGen.setEnabled(enabled);
		
		recoverRasters.setEnabled(enabled);
		txtRecoverDir.setEnabled(enabled);
		btnRecoverDirBrowse.setEnabled(enabled);
		btnRecoverDirLoad.setEnabled(enabled);
		btnRecoverAddRasters.setEnabled(enabled);
		btnRecoverClearRasters.setEnabled(enabled);
		chkRecoverToTest.setEnabled(enabled);

		prgRunning.setEnabled(enabled);
	}
	
	
	/**
	 * Updating controls.
	 */
	void updateControls() {
		enableControls(!isRunning());
		
		baseRasters.setEnabled(true);
		btnBaseAddRasters.setEnabled(chkAllowAdd.isSelected() && !isRunning());
		
		genRasters.setEnabled(true);
		btnGenSave.setEnabled(!chkGenAutoSave.isSelected() && !isRunning());

		recoverRasters.setEnabled(chkRecover.isSelected());
		txtRecoverDir.setEnabled(chkRecover.isSelected() && !isRunning());
		btnRecoverDirBrowse.setEnabled(chkRecover.isSelected() && !isRunning());
		btnRecoverDirLoad.setEnabled(chkRecover.isSelected() && !isRunning());
		btnRecoverAddRasters.setEnabled(chkRecover.isSelected() && chkAllowAdd.isSelected() && !isRunning());
		btnRecoverClearRasters.setEnabled(chkRecover.isSelected() && !isRunning());
		chkRecoverToTest.setEnabled(chkRecover.isSelected() && !isRunning());
		
		prgRunning.setEnabled(taskBackground && isRunning() && isLocalGenModel());
	}
	
	
	/**
	 * Checking whether some task is running.
	 * @return whether some task is running.
	 */
	private boolean isRunning() {
		return runner != null && runner.isStarted();
	}
	
	
	/**
	 * Creating header panel.
	 * @return header panel.
	 */
	private JPanel createHeader() {
		return null;
	}
	
	
	/**
	 * Creating navigator.
	 * @return navigator.
	 */
	private JPanel createNavigator() {
		JPanel navigator = new JPanel(new BorderLayout());
		Dimension preferredSize = new JTextField(WORKING_BASE).getPreferredSize();
		preferredSize = new Dimension(preferredSize.width, preferredSize.height);
		NumberFormatter formatter = new NumberFormatter();
		formatter.setAllowsInvalid(false);
		
		JPanel paneParameters = new JPanel(new BorderLayout());
		navigator.add(paneParameters, BorderLayout.NORTH);
		
		JPanel left = new JPanel(new GridLayout(0, 1));
		paneParameters.add(left, BorderLayout.WEST);
		
		left.add(new JLabel("Name:"));
		left.add(new JLabel("Base: "));
		left.add(new JLabel("Gen.:"));
		left.add(new JLabel("Load 3D:"));
		left.add(new JLabel("Recover:"));
		left.add(new JLabel("Allow adding:"));
		left.add(new JLabel("Gen. number:"));
		left.add(new JLabel("Gen. auto-save:"));
		left.add(new JLabel("Configure:"));
		left.add(new JLabel(" "));

		JPanel right = new JPanel(new GridLayout(0, 1));
		paneParameters.add(right, BorderLayout.CENTER);

		JPanel paneModelName = new JPanel(new BorderLayout());
		right.add(paneModelName);
		this.lblModelName = new JLabel();
		try {
			if (gm != null) this.lblModelName.setText(gm.queryName());
		} catch (Throwable e) {Util.trace(e);}
		paneModelName.add(this.lblModelName, BorderLayout.CENTER);
		
		JPanel paneBaseDir = new JPanel(new BorderLayout());
		right.add(paneBaseDir);
		//
		this.txtBaseDir = new TextField(Paths.get(WORKING_BASE).toString());
		this.txtBaseDir.setPreferredSize(preferredSize);
		paneBaseDir.add(this.txtBaseDir, BorderLayout.CENTER);
		//
		JPanel baseDirButtons = new JPanel();
		baseDirButtons.setLayout(new BoxLayout(baseDirButtons, BoxLayout.X_AXIS));
		paneBaseDir.add(baseDirButtons, BorderLayout.EAST);
		this.btnBaseDirBrowse = UIUtil.makeIconButton(
			"open-16x16.png", 
			"browse_base_dir", 
			"Browse base directory", 
			"Browse base directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					browseBaseDir();
				}
				
			});
		this.btnBaseDirBrowse.setMargin(new Insets(0, 0 , 0, 0));
		baseDirButtons.add(this.btnBaseDirBrowse);
		this.btnBaseDirLoad = UIUtil.makeIconButton(
			"load-16x16.png", 
			"load_base_dir", 
			"Load base directory", 
			"Load base directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					loadBaseDir();
				}
				
			});
		this.btnBaseDirLoad.setMargin(new Insets(0, 0 , 0, 0));
		baseDirButtons.add(this.btnBaseDirLoad);
		
		JPanel paneGenDir = new JPanel(new BorderLayout());
		right.add(paneGenDir);
		this.txtGenDir = new TextField(Paths.get(WORKING_GEN).toString());
		this.txtGenDir.setPreferredSize(preferredSize);
		paneGenDir.add(txtGenDir, BorderLayout.CENTER);
		//
		JPanel genDirButtons = new JPanel();
		genDirButtons.setLayout(new BoxLayout(genDirButtons, BoxLayout.X_AXIS));
		paneGenDir.add(genDirButtons, BorderLayout.EAST);
		this.btnGenDirBrowse = UIUtil.makeIconButton(
			"open-16x16.png",
			"browse_gen_dir", 
			"Browse generation directory", 
			"Browse generation directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					browseGenDir();
				}
				
			});
		this.btnGenDirBrowse.setMargin(new Insets(0, 0 , 0, 0));
		genDirButtons.add(this.btnGenDirBrowse);
		this.btnGenDirLoad = UIUtil.makeIconButton(
			"load-16x16.png", 
			"load_gen_dir", 
			"Load generated directory", 
			"Load generated directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					loadGenDir();
				}
				
			});
		this.btnGenDirLoad.setMargin(new Insets(0, 0 , 0, 0));
		genDirButtons.add(this.btnGenDirLoad);
		
		JPanel paneLoad3D = new JPanel(new BorderLayout());
		right.add(paneLoad3D);
		this.chkLoad3D = new JCheckBox();
		this.chkLoad3D.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {

			}
			
		});
		this.chkLoad3D.setSelected(false);
		paneLoad3D.add(this.chkLoad3D, BorderLayout.CENTER);

		JPanel paneRecover = new JPanel(new BorderLayout());
		right.add(paneRecover);
		this.chkRecover = new JCheckBox();
		this.chkRecover.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {
				if (recoverView != null) recoverView.setVisible(chkRecover.isSelected());
				if (recoverRasters != null) recoverRasters.setEnabled(chkRecover.isSelected());
				if (txtRecoverDir != null) txtRecoverDir.setEnabled(chkRecover.isSelected());
				if (btnRecoverDirBrowse != null) btnRecoverDirBrowse.setEnabled(chkRecover.isSelected());
				if (btnRecoverDirLoad != null) btnRecoverDirLoad.setEnabled(chkRecover.isSelected());
				if (btnRecoverAddRasters != null) btnRecoverAddRasters.setEnabled(chkRecover.isSelected() && chkAllowAdd.isSelected());
				if (btnRecoverClearRasters != null) btnRecoverClearRasters.setEnabled(chkRecover.isSelected());
				if (chkRecoverToTest != null) chkRecoverToTest.setEnabled(chkRecover.isSelected());
			}
			
		});
		this.chkRecover.setSelected(false);
		paneRecover.add(this.chkRecover, BorderLayout.CENTER);
		
		JPanel paneBaseAllowAdd = new JPanel(new BorderLayout());
		right.add(paneBaseAllowAdd);
		this.chkAllowAdd = new JCheckBox();
		this.chkAllowAdd.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {
				if (btnBaseAddRasters != null) btnBaseAddRasters.setEnabled(chkAllowAdd.isSelected());
				if (btnRecoverAddRasters != null) btnRecoverAddRasters.setEnabled(chkRecover.isSelected() && chkAllowAdd.isSelected());
			}
			
		});
		this.chkAllowAdd.setSelected(false);
		paneBaseAllowAdd.add(this.chkAllowAdd, BorderLayout.CENTER);

		JPanel paneGenNum = new JPanel(new BorderLayout());
		right.add(paneGenNum);
		this.txtGenNum = new JFormattedTextField(formatter);
		this.txtGenNum.setValue(GenModelAbstract.GENS_DEFAULT);
		paneGenNum.add(this.txtGenNum, BorderLayout.CENTER);

		JPanel paneGenAutoSave = new JPanel(new BorderLayout());
		right.add(paneGenAutoSave);
		this.chkGenAutoSave = new JCheckBox();
		this.chkGenAutoSave.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {
				if (btnGenSave != null) btnGenSave.setEnabled(!chkGenAutoSave.isSelected());
			}
			
		});
		this.chkGenAutoSave.setSelected(false);
		paneGenAutoSave.add(this.chkGenAutoSave, BorderLayout.CENTER);

		JPanel paneConfig = new JPanel(new BorderLayout());
		paneConfig.setLayout(new BoxLayout(paneConfig, BoxLayout.X_AXIS));
		right.add(paneConfig);
		this.btnConfig = UIUtil.makeIconButton(
			"config-16x16.png",
			"config", 
			"Configure generative model", 
			"Configure generative model", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					configGM();
				}
				
			});
		this.btnConfig.setMargin(new Insets(0, 0 , 0, 0));
		paneConfig.add(this.btnConfig);

		JPanel parametersButtons = new JPanel();
		parametersButtons.setLayout(new BoxLayout(parametersButtons, BoxLayout.X_AXIS));
		right.add(parametersButtons);
		//
		this.btnApplyParams = UIUtil.makeIconButton(
			"apply-16x16.png",
			"apply_params", 
			"Apply parameters", 
			"Apply parameters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					applyParams();
				}
				
			});
		this.btnApplyParams.setMargin(new Insets(0, 0 , 0, 0));
		this.btnApplyParams.setVisible(false);
		parametersButtons.add(this.btnApplyParams);
		//
		this.btnRefreshParams = UIUtil.makeIconButton(
			"refresh-16x16.png",
			"refresh_params", 
			"Refresh parameters", 
			"Refresh parameters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					refreshParams();
				}
				
			});
		this.btnRefreshParams.setMargin(new Insets(0, 0 , 0, 0));
		this.btnRefreshParams.setVisible(false);
		parametersButtons.add(this.btnRefreshParams);
		//
		this.btnResetParams = UIUtil.makeIconButton(
			"reset-16x16.png",
			"reset_params", 
			"Reset parameters", 
			"Reset parameters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					resetParams();
				}
				
			});
		this.btnResetParams.setMargin(new Insets(0, 0 , 0, 0));
		this.btnResetParams.setVisible(false);
		parametersButtons.add(this.btnResetParams);

		JPanel body = new JPanel(new BorderLayout());
		navigator.add(body, BorderLayout.CENTER);
		
		this.recoverView = createRecoverView();
		if (recoverView != null) {
			this.recoverView.setVisible(false);
			body.add(this.recoverView, BorderLayout.CENTER);
		}
		
		JPanel footer = new JPanel(new BorderLayout());
		navigator.add(footer, BorderLayout.SOUTH);

		JPanel toolbar = new JPanel();
		footer.add(toolbar, BorderLayout.SOUTH);
		this.btnReset = new JButton("Reset");
		this.btnReset.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				reset();
			}
			
		});
		toolbar.add(this.btnReset);

		this.btnGen = new JButton("Generate");
		this.btnGen.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				doTask();
			}
			
		});
		toolbar.add(this.btnGen);

		return navigator;
	}

	
	/**
	 * Creating recovery view.
	 * @return recovery view.
	 */
	private JPanel createRecoverView() {
		Dimension preferredSize = new JTextField(WORKING_BASE).getPreferredSize();
		JPanel recoverView = new JPanel(new BorderLayout());

		JPanel recoverHeader = new JPanel();
		recoverHeader.setLayout(new BoxLayout(recoverHeader, BoxLayout.Y_AXIS));
		recoverView.add(recoverHeader, BorderLayout.NORTH);
		
		JPanel paneTextRecover = new JPanel(new FlowLayout(FlowLayout.LEFT));
		recoverHeader.add(paneTextRecover);
		this.lblRecover = new JLabel(RECOVERING_LABEL_TEXT);
		paneTextRecover.add(this.lblRecover);

		JPanel paneRecoverDir = new JPanel(new BorderLayout());
		recoverHeader.add(paneRecoverDir);
		//
		this.txtRecoverDir = new TextField(Paths.get(WORKING_RECOVER).toString());
		this.txtRecoverDir.setPreferredSize(preferredSize);
		paneRecoverDir.add(this.txtRecoverDir, BorderLayout.CENTER);
		//
		JPanel recoverDirButtons = new JPanel();
		recoverDirButtons.setLayout(new BoxLayout(recoverDirButtons, BoxLayout.X_AXIS));
		paneRecoverDir.add(recoverDirButtons, BorderLayout.EAST);
		this.btnRecoverDirBrowse = UIUtil.makeIconButton(
			"open-16x16.png", 
			"browse_recover_dir", 
			"Browse recovering directory", 
			"Browse recovering directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					browseRecoverDir();
				}
				
			});
		this.btnRecoverDirBrowse.setMargin(new Insets(0, 0 , 0, 0));
		recoverDirButtons.add(this.btnRecoverDirBrowse);
		this.btnRecoverDirLoad = UIUtil.makeIconButton(
			"load-16x16.png", 
			"load_recover_dir", 
			"Load recovering directory", 
			"Load recovering directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					loadRecoverDir();
				}
				
			});
		this.btnRecoverDirLoad.setMargin(new Insets(0, 0 , 0, 0));
		recoverDirButtons.add(this.btnRecoverDirLoad);

		this.recoverRasters.setModal(false);
		recoverView.add(new JScrollPane(this.recoverRasters), BorderLayout.CENTER);
		
		JPanel recoverFooter = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		recoverView.add(recoverFooter, BorderLayout.SOUTH);
		
		this.btnRecoverAddRasters = UIUtil.makeIconButton(
			"add-16x16.png",
			"add_recover_rasters", 
			"Add recovering rasters", 
			"Add recovering rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					addRecoverRastersStarter();
				}
				
			});
		this.btnRecoverAddRasters.setMargin(new Insets(0, 0 , 0, 0));
		recoverFooter.add(this.btnRecoverAddRasters);

		this.btnRecoverClearRasters = UIUtil.makeIconButton(
			"clear-16x16.png",
			"clear_recover_rasters", 
			"Clear recovering rasters", 
			"Clear recovering rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					recoverRasters.clearItems();
				}
				
			});
		this.btnRecoverClearRasters.setMargin(new Insets(0, 0 , 0, 0));
		recoverFooter.add(this.btnRecoverClearRasters);

		this.chkRecoverToTest = new JCheckBox("Test");
//		this.chkRecoverToTest.addMouseListener(new MouseAdapter() {
//
//			@Override
//			public void mouseClicked(MouseEvent e) {
//				super.mouseClicked(e);
//				
//				if (chkRecoverToTest.isSelected()) {
//					setGenDir(Paths.get(WORKING_RECOVER));
//					if (lblGen != null) lblGen.setText(RECOVERED_LABEL_TEXT);
//					
//					setRecoverDir(Paths.get(WORKING_TEST));
//					if (lblRecover != null) lblRecover.setText(TEST_LABEL_TEXT);
//				}
//				else {
//					setGenDir(Paths.get(WORKING_GEN));
//					if (lblGen != null) lblGen.setText(GEN_LABEL_TEXT);
//					
//					setRecoverDir(Paths.get(WORKING_RECOVER));
//					if (lblRecover != null) lblRecover.setText(RECOVERING_LABEL_TEXT);
//				}
//			}
//			
//		});
		this.chkRecoverToTest.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {
				if (chkRecoverToTest.isSelected()) {
					setGenDir(Paths.get(WORKING_RECOVER));
					if (lblGen != null) lblGen.setText(RECOVERED_LABEL_TEXT);
					
					setRecoverDir(Paths.get(WORKING_TEST));
					if (lblRecover != null) lblRecover.setText(TEST_LABEL_TEXT);
				}
				else {
					setGenDir(Paths.get(WORKING_GEN));
					if (lblGen != null) lblGen.setText(GEN_LABEL_TEXT);
					
					setRecoverDir(Paths.get(WORKING_RECOVER));
					if (lblRecover != null) lblRecover.setText(RECOVERING_LABEL_TEXT);
				}
			}
			
		});
		recoverFooter.add(chkRecoverToTest);

		return recoverView;
	}
	
	
	/**
	 * Creating body panel.
	 * @return body panel.
	 */
	private JPanel createBody() {
		this.baseView = createBaseView();
		this.genView = createGenView();
		
		JPanel body = new JPanel(new BorderLayout());
		JSplitPane splitter = new JSplitPane(JSplitPane.VERTICAL_SPLIT, this.baseView, this.genView);
		splitter.setOneTouchExpandable(true);
		splitter.setDividerLocation(SPLITTER_DIVISION_LOCATION);
		body.add(splitter, BorderLayout.CENTER);
		return body;
	}

	
	/**
	 * Creating base view.
	 * @return base view.
	 */
	private JPanel createBaseView() {
		JPanel baseView = new JPanel(new BorderLayout());
		
		JPanel baseHeader = new JPanel(new FlowLayout(FlowLayout.LEFT));
		baseView.add(baseHeader, BorderLayout.NORTH);

		JPanel paneTextBase = new JPanel(new FlowLayout(FlowLayout.LEFT));
		baseHeader.add(paneTextBase);
		this.lblBase = new JLabel(BASE_LABEL_TEXT);
		paneTextBase.add(this.lblBase);
		
		this.baseRasters.setModal(false);
		baseView.add(new JScrollPane(this.baseRasters), BorderLayout.CENTER);
		
		JPanel baseFooter = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		baseView.add(baseFooter, BorderLayout.SOUTH);
		
		this.btnBaseAddRasters = UIUtil.makeIconButton(
			"add-16x16.png",
			"add_base_rasters", 
			"Add base rasters", 
			"Add base rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					addBaseRastersStarter();
				}
				
			});
		this.btnBaseAddRasters.setMargin(new Insets(0, 0 , 0, 0));
		baseFooter.add(this.btnBaseAddRasters);

		this.btnBaseClearRasters = UIUtil.makeIconButton(
			"clear-16x16.png",
			"clear_base_rasters", 
			"Clear base rasters", 
			"Clear base rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					baseRasters.clearItems();
				}
				
			});
		this.btnBaseClearRasters.setMargin(new Insets(0, 0 , 0, 0));
		baseFooter.add(this.btnBaseClearRasters);
		
		return baseView;
	}
	
	
	/**
	 * Creating generation view.
	 * @return generation view.
	 */
	private JPanel createGenView() {
		JPanel genView = new JPanel(new BorderLayout());
		
		JPanel genHeader = new JPanel(new FlowLayout(FlowLayout.LEFT));
		genView.add(genHeader, BorderLayout.NORTH);
		
		JPanel paneTextGen = new JPanel(new FlowLayout(FlowLayout.LEFT));
		genHeader.add(paneTextGen);
		this.lblGen = new JLabel(GEN_LABEL_TEXT);
		paneTextGen.add(this.lblGen);

		this.genRasters.setModal(false);
		genView.add(new JScrollPane(this.genRasters), BorderLayout.CENTER);
		
		JPanel genFooter = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		genView.add(genFooter, BorderLayout.SOUTH);

		this.btnGenSave = UIUtil.makeIconButton(
			"save-16x16.png",
			"save_gen_rasters", 
			"Save generated rasters", 
			"Save generated rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					genSave();
				}
				
			});
		this.btnGenSave.setMargin(new Insets(0, 0 , 0, 0));
		genFooter.add(this.btnGenSave);

		this.btnGenClearRasters = UIUtil.makeIconButton(
			"clear-16x16.png",
			"clear_gen_rasters", 
			"Clear generated rasters", 
			"Clear generated rasters", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					genRasters.clearItems();
				}
				
			});
		this.btnGenClearRasters.setMargin(new Insets(0, 0 , 0, 0));
		genFooter.add(this.btnGenClearRasters);

		this.btnGenInfo = UIUtil.makeIconButton(
			"info-16x16.png",
			"gen_info", 
			"Information about generation", 
			"Information about generation", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					genInfo();
				}
				
			});
		this.btnGenInfo.setMargin(new Insets(0, 0 , 0, 0));
		genFooter.add(this.btnGenInfo);

		this.btnGenGen = UIUtil.makeIconButton(
			"generate-sonic-16x16.png",
			"gen_info", 
			"Generating", 
			"Generating", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					genGen();
				}
				
			});
		this.btnGenGen.setMargin(new Insets(0, 0 , 0, 0));
		genFooter.add(this.btnGenGen);

		return genView;
	}
	
	
	/**
	 * Creating footer panel.
	 * @return footer panel.
	 */
	private JPanel createFooter() {
		JPanel main = new JPanel(new BorderLayout());
		
		JPanel header = new JPanel(new BorderLayout());
		main.add(header, BorderLayout.NORTH);
		
		this.prgRunning = new JProgressBar();
		this.prgRunning.setStringPainted(true);
		this.prgRunning.setToolTipText("Running");
		this.prgRunning.setVisible(false);
		header.add(this.prgRunning, BorderLayout.CENTER);

		return main;
	}

	
	/**
	 * Creating main menu bar.
	 * @return main menu bar.
	 */
	JMenuBar createMenuBar() {
		JMenuBar mnBar = new JMenuBar();
		
		JMenu mnFile = new JMenu("File");
		mnFile.setMnemonic('f');

		JMenuItem mniChangeModel = new JMenuItem(
			new AbstractAction("Change model") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					changeModel();
				}
				
			});
		mniChangeModel.setMnemonic('m');
		mniChangeModel.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_M, InputEvent.CTRL_DOWN_MASK));
		if (!exclusive) mnFile.add(mniChangeModel);
		
		JMenuItem mniStopTask = new JMenuItem(
			new AbstractAction("Stop task") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					if (!isRunning()) {
						JOptionPane.showMessageDialog(getGenUI(), "No task running", "No task running", JOptionPane.ERROR_MESSAGE);
						return;
					}
					
					runner.forceStop();
					runner = null;
					updateControls();
				}
				
			});
		mniStopTask.setMnemonic('t');
		mniStopTask.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_T, InputEvent.CTRL_DOWN_MASK));
		mnFile.add(mniStopTask);
		
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
		
		JMenu mnTools = new JMenu("Tools");
		mnTools.setMnemonic('t');

		JMenuItem mniClassify = new JMenuItem(
			new AbstractAction("Classify") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					classify();
				}
				
			});
		mniClassify.setMnemonic('c');
		mniClassify.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_C, InputEvent.CTRL_DOWN_MASK));
		mnTools.add(mniClassify);

		if (mnTools.getMenuComponentCount() > 0) mnBar.add(mnTools);

		return mnBar.getMenuCount() > 0 ? mnBar : null;
	}


	/**
	 * Changing model.
	 */
	private void changeModel() {
		if (isRunning()) {
			JOptionPane.showMessageDialog(this, "Some task running", "Some task running", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (!exclusive) queryLocalGenModel(gm, this);
	}
	
	
	/**
	 * Setting some UI parameters.
	 */
	private void setting() {
		if (isRunning()) {
			JOptionPane.showMessageDialog(this, "Some task running", "Some task running", JOptionPane.ERROR_MESSAGE);
			return;
		}

		NumberFormatter formatter = new NumberFormatter();
		formatter.setAllowsInvalid(false);

		JDialog dlgSetting = new JDialog(this, "Settings", true);
		dlgSetting.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		dlgSetting.setSize(DIALOG_INFO_SIZE);
		dlgSetting.setLocationRelativeTo(this);
		dlgSetting.setLayout(new BorderLayout());
		
		JPanel header = new JPanel(new BorderLayout());
		dlgSetting.add(header, BorderLayout.NORTH);
		
		JPanel left = new JPanel(new GridLayout(0, 1));
		header.add(left, BorderLayout.WEST);
		
		left.add(new JLabel("ImageList icon size:"));
		left.add(new JLabel("ImageList store image:"));
		left.add(new JLabel("Result directory:"));
		left.add(new JLabel("Task background:"));
		left.add(new JLabel("Debug mode:"));
		
		JPanel right = new JPanel(new GridLayout(0, 1));
		header.add(right, BorderLayout.CENTER);

		JFormattedTextField txtImageListIconSize = new JFormattedTextField(formatter);
		txtImageListIconSize.setValue(imageListIconSize);
		right.add(txtImageListIconSize);

		JCheckBox chkImageListStoreImage = new JCheckBox();
		chkImageListStoreImage.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {

			}
			
		});
		chkImageListStoreImage.setSelected(imageListStoreImage);
		right.add(chkImageListStoreImage);
		
		JPanel paneResultDir = new JPanel(new BorderLayout());
		right.add(paneResultDir);
		//
		TextField txtResultDir = new TextField(Paths.get(WORKING_TESTRESULT).toString());
		paneResultDir.add(txtResultDir, BorderLayout.CENTER);
		//
		JPanel resultDirButtons = new JPanel();
		resultDirButtons.setLayout(new BoxLayout(resultDirButtons, BoxLayout.X_AXIS));
		paneResultDir.add(resultDirButtons, BorderLayout.EAST);
		JButton btnResultDirBrowse = UIUtil.makeIconButton(
			"open-16x16.png", 
			"browse_result_dir", 
			"Browse result directory", 
			"Browse result directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					browseResultDir(txtResultDir);
				}
				
			});
		btnResultDirBrowse.setMargin(new Insets(0, 0 , 0, 0));
		resultDirButtons.add(btnResultDirBrowse);
		JButton btnResultDirLoad = UIUtil.makeIconButton(
			"load-16x16.png", 
			"load_result_dir", 
			"Load result directory", 
			"Load result directory", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					loadResultDir(txtResultDir);
				}
				
			});
		btnResultDirLoad.setMargin(new Insets(0, 0 , 0, 0));
//		resultDirButtons.add(btnResultDirLoad);

		JCheckBox chkTaskBackground = new JCheckBox();
		chkTaskBackground.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {

			}
			
		});
		chkTaskBackground.setSelected(taskBackground);
		right.add(chkTaskBackground);

		JCheckBox chkDebug = new JCheckBox();
		chkDebug.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {

			}
			
		});
		chkDebug.setSelected(debug);
		right.add(chkDebug);

		JPanel footer = new JPanel();
		dlgSetting.add(footer, BorderLayout.SOUTH);
		
		JButton ok = new JButton("OK");
		ok.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				Object iconSizeValue = txtImageListIconSize.getValue();
				if ((iconSizeValue == null) || !(iconSizeValue instanceof Number)) return;
				int iconSize = ((Number)iconSizeValue).intValue();
				if (iconSize < ICON_MINSIZE || iconSize > ICON_MAXSIZE) {
					JOptionPane.showMessageDialog(dlgSetting, "Wrong icon size", "Wrong icon size", JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				String resultDirText = txtResultDir.getText();
				if (resultDirText == null) {
					JOptionPane.showMessageDialog(dlgSetting, "Null result directory", "Null result directory", JOptionPane.ERROR_MESSAGE);
					return;
				}
				Path resultDir = Paths.get(resultDirText);
				if (!validateDir(resultDir)) {
					JOptionPane.showMessageDialog(dlgSetting, "Wrong result directory", "Wrong result directory", JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				txtImageListIconSize.setValue(getGenUI().imageListIconSize = iconSize);
				getGenUI().imageListStoreImage = chkImageListStoreImage.isSelected();
				setResultDir(resultDir, txtResultDir);
				getGenUI().taskBackground = chkTaskBackground.isSelected();
				getGenUI().debug = chkDebug.isSelected();
				
				dlgSetting.dispose();
			}
		});
		footer.add(ok);
		
		JButton reset = new JButton("Reset");
		reset.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				txtImageListIconSize.setValue(ICON_MINSIZE);
				chkImageListStoreImage.setSelected(IMAGELIST_STORE_IMAGE);
				chkTaskBackground.setSelected(TASK_BACKGROUND);
				chkDebug.setSelected(DEBUG);
			}
		});
		footer.add(reset);

		JButton cancel = new JButton("Cancel");
		cancel.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				dlgSetting.dispose();
			}
		});
		footer.add(cancel);

		dlgSetting.setVisible(true);
	}

	
	/**
	 * Getting directory from text field.
	 * @param txtDir specified text dir.
	 * @return directory from text field.
	 */
	private static Path getDir(JTextField txtDir) {
		if (txtDir == null) return null;
		String dirText = txtDir.getText();
		Path dir = (dirText == null || dirText.isEmpty()) ? null : Paths.get(dirText);
		return (dir != null && Files.exists(dir) && Files.isDirectory(dir)) ? dir : null;
	}

	
	/**
	 * Getting base directory.
	 * @return base directory.
	 */
	private Path getBaseDir() {
		return getDir(txtBaseDir);
	}
	
	
	/**
	 * Validating directory.
	 * @param dir specified directory.
	 * @return true if specified directory is valid.
	 */
	private static boolean validateDir(Path dir) {
		if ((dir == null) || (Files.exists(dir) && !Files.isDirectory(dir))) return false;
		if (!Files.exists(dir)) {
			try {
				dir = Files.createDirectory(dir);
			} catch (Throwable e) {
				dir = null;
				Util.trace(e);
			}
			if (dir == null) return false;
		}
		
		return Files.exists(dir) && Files.isDirectory(dir);
	}
	
	
	/**
	 * Setting directory.
	 * @param txtDir specified directory.
	 * @return true if setting is successful.
	 */
	private static boolean setDir(Path dir, JTextField txtDir) {
		if (validateDir(dir)) {
			if (txtDir != null) txtDir.setText(dir.toString());
			return true;
		}
		else
			return false;
	}

	
	/**
	 * Setting base directory.
	 * @param baseDir base directory.
	 * @return true if setting is successful.
	 */
	private boolean setBaseDir(Path baseDir) {
		return setDir(baseDir, txtBaseDir);
	}

	
	/**
	 * Browsing base directory.
	 */
	private void browseBaseDir() {
		JFileChooser fc = new JFileChooser(getBaseDir() != null ? getBaseDir().toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		fc.showOpenDialog(this);
		
		File baseDir = fc.getSelectedFile();
		if (baseDir == null || !baseDir.exists()) return;
		baseRasters.clearItems();
		if (!setBaseDir(baseDir.toPath())) return;
		
		int ret = JOptionPane.showConfirmDialog(this, "Would you like to load base rasters?", "Load base rasters", JOptionPane.OK_CANCEL_OPTION);
		if (ret == JOptionPane.OK_OPTION) loadBaseDir();
	}

	
	/**
	 * Loading base directory.
	 */
	private void loadBaseDir() {
		Path baseDir = getBaseDir();
		if (baseDir == null) {
			JOptionPane.showMessageDialog(this, "Wrong base directory", "Wrong base directory", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (chkAllowAdd.isSelected() || LOAD_RASTER_ALWAYS)
			addRasters(baseDir, baseRasters);
		else
			baseRasters.setListData(baseDir, imageListIconSize, imageListStoreImage);
	}

	
	/**
	 * Getting generated directory.
	 * @return generated directory.
	 */
	private Path getGenDir() {
		return getDir(txtGenDir);
	}

	
	/**
	 * Setting generated directory.
	 * @param genDir generated directory.
	 * @return true if setting is successful.
	 */
	private boolean setGenDir(Path genDir) {
		return setDir(genDir, txtGenDir);
	}
	
	
	/**
	 * Browsing generation directory.
	 */
	private void browseGenDir() {
		JFileChooser fc = new JFileChooser(getGenDir() != null ? getGenDir().toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		fc.showOpenDialog(this);
		
		File genDir = fc.getSelectedFile();
		if (genDir == null || !genDir.exists()) return;
		genRasters.clearItems();
		if (!setGenDir(genDir.toPath())) return;
		
		int ret = JOptionPane.showConfirmDialog(this, "Would you like to load generated rasters?", "Load generated rasters", JOptionPane.OK_CANCEL_OPTION);
		if (ret == JOptionPane.OK_OPTION) loadGenDir();
	}

	
	/**
	 * Loading generated directory.
	 */
	private void loadGenDir() {
		Path genDir = getGenDir();
		if (genDir == null) {
			JOptionPane.showMessageDialog(this, "Wrong generated directory", "Wrong generated directory", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (chkAllowAdd.isSelected() || LOAD_RASTER_ALWAYS)
			addRasters(genDir, genRasters);
		else
			genRasters.setListData(genDir, imageListIconSize, imageListStoreImage);
	}

	
	/**
	 * Getting result directory.
	 * @return result directory.
	 */
	private Path getResultDir() {
		return (resultDir != null && Files.exists(resultDir) && Files.isDirectory(resultDir)) ? resultDir : null;
	}


	/**
	 * Setting result directory.
	 * @param resultDir result directory.
	 * @param txtResultDir result directory text field.
	 * @return true if setting is successful.
	 */
	private boolean setResultDir(Path resultDir, JTextField txtResultDir) {
		if (validateDir(resultDir)) {
			this.resultDir = resultDir;
			if (txtResultDir != null) txtResultDir.setText(resultDir.toString());
			return true;
		}
		else
			return false;
	}

	
	/**
	 * Browsing result directory.
	 * @param txtResultDir result directory text field.
	 */
	private void browseResultDir(JTextField txtResultDir) {
		JFileChooser fc = new JFileChooser(getResultDir() != null ? getResultDir().toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		fc.showOpenDialog(this);
		
		File resultDir = fc.getSelectedFile();
		if (resultDir == null || !resultDir.exists()) return;
		if (!setDir(resultDir.toPath(), txtResultDir)) return;
	}

	
	/**
	 * Loading result directory.
	 * @param txtResultDir result directory text field.
	 */
	private void loadResultDir(JTextField txtResultDir) {
		Path resultDir = getResultDir();
		if (resultDir == null) {
			JOptionPane.showMessageDialog(this, "Wrong result directory", "Wrong result directory", JOptionPane.ERROR_MESSAGE);
			return;
		}
	}

	
	/**
	 * Generation information.
	 */
	private void genInfo() {
		JDialog dlgInfo = new JDialog(UIUtil.getWindowForComponent(this), "Generation information", ModalityType.APPLICATION_MODAL);
		dlgInfo.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		dlgInfo.setSize(DIALOG_INFO_SIZE);
		dlgInfo.setLocationRelativeTo(UIUtil.getWindowForComponent(this));
		dlgInfo.setLayout(new BorderLayout());

		JPanel body = new JPanel(new BorderLayout());
		dlgInfo.add(body, BorderLayout.CENTER);

		TextArea txtInfo = new TextArea("");
		txtInfo.setEditable(false);
		body.add(new JScrollPane(txtInfo), BorderLayout.CENTER);
		StringBuffer buffer = new StringBuffer();
		try {
			if (chkRecover.isSelected()) {
				String gmName = gm.queryName();
				double lr = 1;
				DataConfig config = gm.queryConfig();
				if (config.containsKey(NetworkAbstract.LEARN_RATE_FIELD)) lr = config.getAsReal(NetworkAbstract.LEARN_RATE_FIELD);
				String name = gmName + "-" + Util.format(lr);
				buffer.append(name + ", error=" + Util.format(this.error) + "\n");
			}
		} catch (Throwable e) {Util.trace(e);}
		txtInfo.setText(buffer.toString());
		
		JPanel footer = new JPanel();
		dlgInfo.add(footer, BorderLayout.SOUTH);
		
		JButton close = new JButton("Close");
		close.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				dlgInfo.dispose();
			}
			
		});
		footer.add(close);

		dlgInfo.setVisible(true);
	}
	
	
	/**
	 * Generating based on current model.
	 */
	private void genGen() {
		List<Raster> newGenRasters = null;
		int nGens = (txtGenNum.getValue() != null && txtGenNum.getValue() instanceof Number) ? ( (Number) txtGenNum.getValue()).intValue() : GenModelAbstract.GENS_DEFAULT; 
		nGens = nGens < GenModelAbstract.GENS_DEFAULT ? GenModelAbstract.GENS_DEFAULT : nGens;
		txtGenNum.setValue(nGens);
		try {
			newGenRasters = gm.genRasters(nGens);
		} catch (Throwable e) {Util.trace(e);}
		
		if (newGenRasters == null || newGenRasters.size() == 0) {
			JOptionPane.showMessageDialog(this, "No generated raster", "No generated raster", JOptionPane.WARNING_MESSAGE);
			return;
		}

		String gmName = Util.NONAME;
		try {
			gmName = gm.queryName();
		} catch (Throwable e) {Util.trace(e);}
		
		Path genDir = getGenDir();
		if (genDir != null && chkGenAutoSave.isSelected()) {
			RasterAssoc.saveDirector(newGenRasters, genDir, gmName, true);
			genRasters.setListData(genDir, imageListIconSize, imageListStoreImage);
		}
		else {
			genRasters.addRasters(gmName, newGenRasters);
		}

	}
	
	
	/**
	 * Configuring generative model.
	 */
	private void configGM() {
		if (gm instanceof GenModel) {
			new AlgConfigDlg(this, (GenModel)gm).setVisible(true);
			return;
		}
		
		GenModelRemoteWrapper wrapper = new GenModelRemoteWrapper(gm);
		AlgConfigDlg dlgConfig = new AlgConfigDlg(this, wrapper);
		dlgConfig.setVisible(true);
		if (!dlgConfig.getPropPane().isApplied()) return;
		
		DataConfig config = null;
		PropList propList = dlgConfig.getPropPane().getPropTable().getPropList();
		if (propList instanceof DataConfig)
			config = (DataConfig)propList;
		else {
			config = new DataConfig();
			config.putAll(propList);
		}
		
		try {
			gm.putConfig(config);
		} catch (Throwable e) {Util.trace(e);}
	}

	/**
	 * Applying parameters.
	 */
	private void applyParams() {
		JOptionPane.showMessageDialog(this, "Not implemented yet", "Not implemented yet", JOptionPane.ERROR_MESSAGE);
	}

	
	/**
	 * Refreshing parameters.
	 */
	private void refreshParams() {
		JOptionPane.showMessageDialog(this, "Not implemented yet", "Not implemented yet", JOptionPane.ERROR_MESSAGE);
	}

	
	/**
	 * Resetting parameters.
	 */
	private void resetParams() {
		JOptionPane.showMessageDialog(this, "Not implemented yet", "Not implemented yet", JOptionPane.ERROR_MESSAGE);
	}

	
	/**
	 * Resetting all things.
	 */
	void reset() {
		try {
			if (gm != null) this.lblModelName.setText(gm.queryName());
		} catch (Throwable e) {Util.trace(e);}
		txtBaseDir.setText(Paths.get(WORKING_BASE).toString());
		chkLoad3D.setSelected(false);
		chkRecover.setSelected(false);
		chkAllowAdd.setSelected(false);
		txtGenNum.setValue(GenModelAbstract.GENS_DEFAULT);
		chkGenAutoSave.setSelected(false);
		
		baseRasters.clearItems();
		
		genRasters.clearItems();
		
		recoverRasters.clearItems();
		chkRecoverToTest.setSelected(false);
		
		prgRunning.setMaximum(0);
		prgRunning.setValue(0);
		prgRunning.setVisible(false);
		
		if (runner != null) {
			runner.stop();
			runner = null;
		}
		
		updateControls();
	}

	
	/**
	 * Loading rasters.
	 * @param dirOrFile directory or file.
	 * @return list of rasters.
	 */
	private List<Raster> loadRasters(Path dirOrFile) {
		return chkLoad3D.isSelected() ? RasterAssoc.load3D(dirOrFile) : RasterAssoc.load(dirOrFile);
	}
	
	
	/**
	 * Loading CIFAR rasters.
	 * @param dirOrFile directory or file.
	 * @param nImages number images to be loaded.
	 * @return list of rasters.
	 */
	private List<Raster> loadRastersCIFAR(Path dirOrFile, int nImages) {
		if (chkLoad3D.isSelected())
			return Util.newList(0);
		else
			return nImages > 0 ? RasterAssoc.loadCIFAR(dirOrFile, nImages) : RasterAssoc.loadCIFAR(dirOrFile);
	}

	
	/**
	 * Loading rasters by folders.
	 * @param dir directory.
	 * @return list of rasters by folders.
	 */
	private List<Raster> loadRastersFolders(Path dir) {
		return chkLoad3D.isSelected() ? RasterAssoc.loadFolders3D(dir) : RasterAssoc.loadFolders(dir);
	}

	
	/**
	 * Adding rasters.
	 * @param imageList image list.
	 * @param files array of files.
	 * @return true if adding is successful.
	 */
	private boolean addRasters(ImagePathListExt imageList, File...files) {
		if (files == null || files.length == 0) {
			JOptionPane.showMessageDialog(this, "No rasters", "No rasters", JOptionPane.ERROR_MESSAGE);
			return false;
		}
		
		boolean added = false;
		for (File baseFile : files) {
			try {
				List<Raster> rasters = loadRasters(baseFile.toPath());
				if (rasters.size() == 0 || rasters.get(0) == null) continue;
				ImageListItem<Path> item = imageList.addItem(baseFile.toPath(), imageListIconSize, imageListStoreImage);
				if (item != null) {
					imageList.setItemRaster(item, rasters.get(0));
					added = true;
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		return added;
	}
	
	
	/**
	 * Adding rasters from specified directory.
	 * @param dir specified directory.
	 * @param imageList image list.
	 * @return true if adding is successful.
	 */
	private boolean addRasters(Path dir, ImagePathListExt imageList) {
		if (dir == null || !Files.isDirectory(dir)) return false;
		
		return addRasters(imageList, dir.toFile().listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return pathname.isFile();
			}
		}));
	}
	
	
	/**
	 * Adding rasters.
	 * @param imageList image list.
	 * @param curDir current directory.
	 */
	private void addRasters(ImagePathListExt imageList, Path curDir) {
		JFileChooser fc = new JFileChooser(curDir != null ? curDir.toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
		fc.setMultiSelectionEnabled(true);
		fc.showOpenDialog(this);
		File[] files = fc.getSelectedFiles();
		if (files == null || files.length == 0) {
			JOptionPane.showMessageDialog(this, "No selected rasters", "No selected rasters", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		if(!addRasters(imageList, files)) JOptionPane.showMessageDialog(this, "Finish to add rasters", "Finish to add rasters", JOptionPane.INFORMATION_MESSAGE);
	}

	
	/**
	 * Adding CIFAR rasters.
	 * @param imageList image list.
	 * @param curDir current directory.
	 */
	private void addRastersCIFAR(ImagePathListExt imageList, Path curDir) {
		JFileChooser fc = new JFileChooser(curDir != null ? curDir.toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
		fc.setMultiSelectionEnabled(true);
		fc.showOpenDialog(this);
		
		File[] baseFiles = fc.getSelectedFiles();
		if (baseFiles == null || baseFiles.length == 0) {
			JOptionPane.showMessageDialog(this, "No selected CIFAR files", "No selected CIFAR files", JOptionPane.ERROR_MESSAGE);
			return;
		}

		int nImages = -1;
		try {
			String txtNumber = JOptionPane.showInputDialog(this, "Enter number of images", nImages);
			if (txtNumber != null) nImages = Integer.parseInt(txtNumber.trim());
		} catch (Throwable e) {}

		boolean added = false;
		for (File baseFile : baseFiles) {
			try {
				List<Raster> rasters = loadRastersCIFAR(baseFile.toPath(), nImages);
				for (int i = 0; i < rasters.size(); i++) {
					Raster raster = rasters.get(i);
					String name = RasterAssoc.genDefaultName(ImageAssoc.CIFAR, raster.getDefaultFormat(), i+1);
					ImageListItem<Path> item = imageList.addItem(name, raster.getRepImage(), imageListIconSize);
					if (item != null) {
						imageList.setItemRaster(item, raster);
						added = true;
					}
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		
		if (added) JOptionPane.showMessageDialog(this, "Finish to add CIFAR rasters", "Finish to add CIFAR rasters", JOptionPane.INFORMATION_MESSAGE);
	}
	
	
	/**
	 * Adding rasters by folders.
	 * @param imageList image list.
	 * @param curDir current directory.
	 */
	private void addRastersFolders(ImagePathListExt imageList, Path curDir) {
		JFileChooser fc = new JFileChooser(curDir != null ? curDir.toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		fc.showOpenDialog(this);
		
		File baseDir = fc.getSelectedFile();
		if (baseDir == null) {
			JOptionPane.showMessageDialog(this, "No selected folder", "No selected folder", JOptionPane.ERROR_MESSAGE);
			return;
		}

		boolean added = false;
		List<Raster> rasters = loadRastersFolders(baseDir.toPath());
		for (int i = 0; i < rasters.size(); i++) {
			Raster raster = rasters.get(i);
			String name = RasterAssoc.genDefaultName(baseDir.getName(), raster.getDefaultFormat(), i+1);
			Path path = raster instanceof RasterWrapper ? ((RasterWrapper)raster).getPath() : null;
			ImageListItem<Path> item = path != null ? imageList.addItem(path, imageListIconSize, imageListStoreImage) :
				imageList.addItem(name, raster.getRepImage(), imageListIconSize);
			if (item != null) {
				imageList.setItemRaster(item, raster);
				added = true;
			}
		}
		
		if (added) JOptionPane.showMessageDialog(this, "Finish to add rasters by folders", "Finish to add rasters by folders", JOptionPane.INFORMATION_MESSAGE);
	}

	
	/**
	 * Adding base rasters.
	 */
	private void addBaseRasters() {
		if (chkAllowAdd.isSelected()) addRasters(baseRasters, getBaseDir());
	}

	
	/**
	 * Adding base CIFAR rasters.
	 */
	private void addBaseRastersCIFAR() {
		if (chkLoad3D.isSelected()) {
			JOptionPane.showMessageDialog(this, "Loading 3D not allowed", "Loading 3D not allowed", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (chkAllowAdd.isSelected()) addRastersCIFAR(baseRasters, getBaseDir());
	}

	
	/**
	 * Adding base rasters by folders.
	 */
	private void addBaseRastersFolders() {
		if (chkAllowAdd.isSelected()) addRastersFolders(baseRasters, getBaseDir());
	}

	
	/**
	 * Starting adding base rasters.
	 */
	private void addBaseRastersStarter( ) {
		List<String> addRasters = Util.newList(0);
		addRasters.addAll(Arrays.asList(ADD_RASTERS));
		Collections.sort(addRasters);
		final StartDlg dlgStarter = new StartDlg(this, "Add rasters") {
			
			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected void start() {
				String flag = (String)getItemControl().getSelectedItem();
				switch (flag) {
				case ADD_RASTERS_NORMAL:
					addBaseRasters();
					break;
				case ADD_RASTERS_CIFAR:
					addBaseRastersCIFAR();
					break;
				case ADD_RASTERS_FOLDERS:
					addBaseRastersFolders();
					break;
				default:
					addBaseRasters();
					break;
				}
				dispose();
			}
			
			@Override
			protected JComboBox<?> createItemControl() {
				return new JComboBox<String>(addRasters.toArray(new String[] {}));
			}
			
		};
		dlgStarter.getItemControl().setSelectedItem(ADD_RASTERS_NORMAL);
		dlgStarter.setSize(DIALOG_INFO_SIZE);
		dlgStarter.setLocationRelativeTo(this);
        dlgStarter.setVisible(true);
	}
	
	
	/**
	 * Getting recovering directory.
	 * @return recovering directory.
	 */
	private Path getRecoverDir() {
		return getDir(txtRecoverDir);
	}
	
	
	/**
	 * Setting recovering directory.
	 * @param genDir generated directory.
	 * @return true if setting is successful.
	 */
	private boolean setRecoverDir(Path recoverDir) {
		return setDir(recoverDir, txtRecoverDir);
	}

	
	/**
	 * Browsing recovering directory.
	 */
	private void browseRecoverDir() {
		if (!chkRecover.isSelected()) return;
		JFileChooser fc = new JFileChooser(getRecoverDir() != null ? getRecoverDir().toFile() : new File("."));
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		fc.showOpenDialog(this);
		
		File recoverDir = fc.getSelectedFile();
		if (recoverDir == null || !recoverDir.exists()) return;
		recoverRasters.clearItems();
		if (!setRecoverDir(recoverDir.toPath())) return;
		
		int ret = JOptionPane.showConfirmDialog(this, "Would you like to load recovering rasters?", "Load recovering rasters", JOptionPane.OK_CANCEL_OPTION);
		if (ret == JOptionPane.OK_OPTION) loadRecoverDir();
	}

	
	/**
	 * Loading recovering directory.
	 */
	private void loadRecoverDir() {
		if (!chkRecover.isSelected()) return;
		Path recoverDir = getRecoverDir();
		if (recoverDir == null) {
			JOptionPane.showMessageDialog(this, "Wrong recovering directory", "Wrong recovering directory", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (chkAllowAdd.isSelected() || LOAD_RASTER_ALWAYS)
			addRasters(recoverDir, recoverRasters);
		else
			recoverRasters.setListData(recoverDir, imageListIconSize, imageListStoreImage);
	}

	
	/**
	 * Adding recovering rasters.
	 */
	private void addRecoverRasters() {
		if (chkRecover.isSelected() && chkAllowAdd.isSelected()) addRasters(recoverRasters, getRecoverDir());
	}
	
	
	/**
	 * Adding recovering CIFAR rasters.
	 */
	private void addRecoverRastersCIFAR() {
		if (chkLoad3D.isSelected()) {
			JOptionPane.showMessageDialog(this, "Loading 3D not allowed", "Loading 3D not allowed", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (chkRecover.isSelected() && chkAllowAdd.isSelected()) addRastersCIFAR(recoverRasters, getRecoverDir());
	}

	
	/**
	 * Adding recovering rasters by folders.
	 */
	private void addRecoverRastersFolders() {
		if (chkRecover.isSelected() && chkAllowAdd.isSelected()) addRastersFolders(recoverRasters, getRecoverDir());
	}

	
	/**
	 * Starting adding base rasters.
	 */
	private void addRecoverRastersStarter( ) {
		List<String> addRasters = Util.newList(0);
		addRasters.addAll(Arrays.asList(ADD_RASTERS));
		Collections.sort(addRasters);
		final StartDlg dlgStarter = new StartDlg(this, "Add rasters") {
			
			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected void start() {
				String flag = (String)getItemControl().getSelectedItem();
				switch (flag) {
				case ADD_RASTERS_NORMAL:
					addRecoverRasters();
					break;
				case ADD_RASTERS_CIFAR:
					addRecoverRastersCIFAR();
					break;
				case ADD_RASTERS_FOLDERS:
					addRecoverRastersFolders();
					break;
				default:
					addRecoverRasters();
					break;
				}
				dispose();
			}
			
			@Override
			protected JComboBox<?> createItemControl() {
				return new JComboBox<String>(addRasters.toArray(new String[] {}));
			}
			
		};
		dlgStarter.getItemControl().setSelectedItem(ADD_RASTERS_NORMAL);
		dlgStarter.setSize(DIALOG_INFO_SIZE);
		dlgStarter.setLocationRelativeTo(this);
        dlgStarter.setVisible(true);
	}
	
	
	/**
	 * Saving generated rasters.
	 */
	private void genSave() {
		int n = genRasters.getItemCount();
		if (n == 0) {
			JOptionPane.showMessageDialog(this, "No generated raster", "No generated raster", JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		
		Path genDir = getGenDir();
		if (genDir == null) {
			JOptionPane.showMessageDialog(this, "Generated directory not to exist", "Generated directory not to exist", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		for (int i = 0; i < n; i++) {
			try {
				ImageListItem<Path> item = genRasters.getItem(i);
				if (!item.isPseudoPath()) continue;
				
				String name = item.queryPath().getFileName().toString();
				Object tag = item.getTag();
				if ((tag != null) && (tag instanceof Raster)) {
					Raster raster = (Raster)tag;
					String ext = "." + raster.getDefaultFormat();
					if (name.endsWith(ext))
						raster.save(genDir.resolve(name));
					else
						raster.save(genDir.resolve(name + ext));
				}
				else {
					java.awt.Image image = item.queryImage();
					if (image == null) continue;
					BufferedImage bufImage = null;
					if (image instanceof BufferedImage)
						bufImage = (BufferedImage)image;
					else
						bufImage = UIUtil.convertToBufferedImage(image);
					
					String ext = "." + Image.getDefaultFormat();
					if (name.endsWith(ext))
						new ImageWrapper(bufImage).save(genDir.resolve(name));
					else
						new ImageWrapper(bufImage).save(genDir.resolve(name + ext));
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		
		JOptionPane.showMessageDialog(this, "Finish to save generated rasters", "Successful saving", JOptionPane.INFORMATION_MESSAGE);
	}
	
	
	/**
	 * Doing main task.
	 */
	private synchronized void doTask0() {
		SimpleDateFormat df = new SimpleDateFormat(Util.DATE_FORMAT);
		
		System.out.println("Begin task at " + df.format(new Date()));
		if (chkRecover.isSelected()) {
			System.out.println("Doing recovering...");
			recover();
		}
		else {
			System.out.println("Doing generating...");
			generate();
		}
		System.out.println("End task at " + df.format(new Date()));
	}
	
	
	/**
	 * Doing main task.
	 */
	private void doTask() {
		if (taskBackground) {
			if (runner != null && runner.isRunning()) runner.stop();
			runner = new AbstractRunner() {
				
				@Override
				protected void task() {
					updateControls();
					doTask0();
					
					thread = null;
					if (paused) {
						paused = false;
						notifyAll();
					}
				}
				
				@Override
				protected void clear() {
					updateControls();
				}
				
			};
			runner.start();
		}
		else if (!debug) {
			WaitDialog.doTask(new Task<Void>() {

				@Override
				public Void doSomeTask(Object... params) {
					updateControls();
					doTask0();
					updateControls();
					
					return null;
				}
				
			}, this);
		}
		else {
			doTask0();
		}
		
	}
	
	
	/**
	 * Generating rasters.
	 */
	private void generate() {
		genRasters.clearItems();
		List<Raster> bRasters = Util.newList(0);
		if (chkAllowAdd.isSelected()) {
			bRasters = baseRasters.queryItemRasters();
		}
		else {
			Path baseDir = getBaseDir();
			if (baseDir == null || !Files.exists(baseDir) || !Files.isDirectory(baseDir)) {
				JOptionPane.showMessageDialog(this, "Wrong base directory", "Wrong base directory", JOptionPane.ERROR_MESSAGE);
				return;
			}
			bRasters = loadRasters(baseDir);
			if (baseRasters.getItemCount() != bRasters.size() && (baseRasters.getItemCount() == 0 || !chkLoad3D.isSelected()))
				baseRasters.setListData(baseDir, imageListIconSize, imageListStoreImage);
		}
		
		if (bRasters.size() == 0) {
			JOptionPane.showMessageDialog(this, "Empty base rasters", "Empty base rasters", JOptionPane.ERROR_MESSAGE);
			return;
		}

		List<Raster> newGenRasters = null;
		int nGens = (txtGenNum.getValue() != null && txtGenNum.getValue() instanceof Number) ? ( (Number) txtGenNum.getValue()).intValue() : GenModelAbstract.GENS_DEFAULT; 
		nGens = nGens < GenModelAbstract.GENS_DEFAULT ? GenModelAbstract.GENS_DEFAULT : nGens;
		txtGenNum.setValue(nGens);
		try {
			prgRunning.setValue(0);
			prgRunning.setVisible(taskBackground && isLocalGenModel());
			
			newGenRasters = gm.genRasters(bRasters, nGens);
			
			prgRunning.setVisible(false);
		} catch (Throwable e) {Util.trace(e);}
		
		if (newGenRasters == null || newGenRasters.size() == 0) {
			JOptionPane.showMessageDialog(this, "No generated raster", "No generated raster", JOptionPane.WARNING_MESSAGE);
			return;
		}

		String gmName = Util.NONAME;
		try {
			gmName = gm.queryName();
		} catch (Throwable e) {Util.trace(e);}
		
		Path genDir = getGenDir();
		if (genDir != null && chkGenAutoSave.isSelected()) {
			RasterAssoc.saveDirector(newGenRasters, genDir, gmName, true);
			genRasters.setListData(genDir, imageListIconSize, imageListStoreImage);
		}
		else {
			genRasters.setRasters(gmName, newGenRasters);
		}
		
	}

	
	/**
	 * Recovering rasters.
	 */
	void recover() {
		genRasters.clearItems();
		List<Raster> bRasters = Util.newList(0);
		List<Raster> rRasters = Util.newList(0);
		if (chkAllowAdd.isSelected()) {
			bRasters = baseRasters.queryItemRasters();
			rRasters = recoverRasters.queryItemRasters();
		}
		else {
			Path baseDir = getBaseDir();
			if (baseDir == null || !Files.exists(baseDir) || !Files.isDirectory(baseDir)) {
				JOptionPane.showMessageDialog(this, "Wrong base directory", "Wrong base directory", JOptionPane.ERROR_MESSAGE);
				return;
			}
			bRasters = loadRasters(baseDir);
			if (baseRasters.getItemCount() != bRasters.size() && (baseRasters.getItemCount() == 0 || !chkLoad3D.isSelected()))
				baseRasters.setListData(baseDir, imageListIconSize, imageListStoreImage);

			Path recoverDir = getRecoverDir();
			if (recoverDir == null || !Files.exists(recoverDir) || !Files.isDirectory(recoverDir)) {
				JOptionPane.showMessageDialog(this, "Wrong recovering directory", "Wrong recovering directory", JOptionPane.ERROR_MESSAGE);
				return;
			}
			rRasters = loadRasters(recoverDir);
			if (recoverRasters.getItemCount() != rRasters.size())
				recoverRasters.setListData(recoverDir, imageListIconSize, imageListStoreImage);
		}
		
		if (bRasters.size() == 0) {
			JOptionPane.showMessageDialog(this, "Empty base rasters", "Empty base rasters", JOptionPane.ERROR_MESSAGE);
			return;
		}
		if (rRasters.size() == 0) {
			JOptionPane.showMessageDialog(this, "Empty recovering rasters", "Empty recovering rasters", JOptionPane.ERROR_MESSAGE);
			return;
		}

		List<G> glist = null;
		int nGens = (txtGenNum.getValue() != null && txtGenNum.getValue() instanceof Number) ? ( (Number) txtGenNum.getValue()).intValue() : GenModelAbstract.GENS_DEFAULT; 
		nGens = nGens < GenModelAbstract.GENS_DEFAULT ? GenModelAbstract.GENS_DEFAULT : nGens;
		txtGenNum.setValue(nGens);
		try {
			prgRunning.setValue(0);
			prgRunning.setVisible(taskBackground && isLocalGenModel());
			
			glist = gm.recoverRasters(bRasters, rRasters, nGens);
			
			prgRunning.setVisible(false);
		} catch (Throwable e) {Util.trace(e);}
		
		if (glist == null || glist.size() == 0) {
			JOptionPane.showMessageDialog(this, "No recovered raster", "No recovered raster", JOptionPane.WARNING_MESSAGE);
			return;
		}

		String gmName = Util.NONAME;
		try {
			gmName = gm.queryName();
		} catch (Throwable e) {Util.trace(e);}
		
		Path genDir = getGenDir();
		double errorSum = 0;
		int errorCount = 0;
		if (genDir != null && chkGenAutoSave.isSelected()) {
			for (int i = 0; i < glist.size(); i++) {
				String genName = gmName;
				G g = glist.get(i);
				if ((g.tag != null) && (g.tag instanceof RasterWrapper))
					genName = ((g.tag != null) && (g.tag instanceof RasterWrapper)) ? gmName + "." + ((RasterWrapper)g.tag).getNamePlain() : gmName;
				else if (recoverRasters.getItemCount() > i) {
					String itemName = recoverRasters.getItem(i).getPossibleName();
					genName = (itemName != null) ? gmName + "." + itemName : gmName;
				}
				
				Raster recoveredRaster = (Raster)g.xgenUndefined;
				Path path = RasterAssoc.genDefaultPath(genDir, genName, recoveredRaster.getDefaultFormat());
				recoveredRaster.save(path);
				
				errorSum += g.error;
				errorCount++;
			}
			genRasters.setListData(genDir, imageListIconSize, imageListStoreImage);
		}
		else {
			List<ImageListItem<Path>> items = Util.newList(glist.size());
			for (int i = 0; i < glist.size(); i++) {
				String genName = gmName;
				G g = glist.get(i);
				if ((g.tag != null) && (g.tag instanceof RasterWrapper))
					genName = ((g.tag != null) && (g.tag instanceof RasterWrapper)) ? gmName + "." + ((RasterWrapper)g.tag).getNamePlain() : gmName;
				else if (glist.size() == nGens*recoverRasters.getItemCount()) {
					int k = i/nGens;
					String itemName = k < recoverRasters.getItemCount() ? recoverRasters.getItem(k).getPossibleName() : null;
					genName = (itemName != null) ? gmName + "." + itemName : gmName;
				}
				
				Raster recoveredRaster = (Raster)g.xgenUndefined;
				java.awt.Image image = recoveredRaster.getRepImage();
				String name = RasterAssoc.genDefaultName(genName, recoveredRaster.getDefaultFormat(), i+1);
				ImageListItem<Path> item = null;
				if (image != null)
					item = ImagePathList.createItem(name, image, imageListIconSize);
				else
					item = ImagePathList.createItem(name);
				if (item != null) {
					genRasters.setItemRaster(item, recoveredRaster);
					items.add(item);
				}
				
				errorSum += g.error;
				errorCount++;
			}
			genRasters.setListData(items);
		}
		
		if (errorCount != 0) {
			this.error = errorSum / (double)errorCount;
			double lr = 1;
			try {
				DataConfig config = gm.queryConfig();
				if (config.containsKey(NetworkAbstract.LEARN_RATE_FIELD)) lr = config.getAsReal(NetworkAbstract.LEARN_RATE_FIELD);
			} catch (Throwable e) {Util.trace(e);}
			
			try {
				BufferedWriter writer = Files.newBufferedWriter(resultDir.resolve(TEST_FILE_NAME), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
				String name = gmName + "-" + Util.format(lr);
				writer.write(name + ", error=" + Util.format(this.error) + "\n");
				writer.flush();
				writer.close();
			} catch (Throwable e) {Util.trace(e);}
		}
	}

	
	/**
	 * Classifying rasters.
	 */
	private void classify() {
		List<String> classifyRasters = Util.newList(0);
		classifyRasters.addAll(Arrays.asList(VIEWS));
		Collections.sort(classifyRasters);
		final StartDlg dlgStarter = new StartDlg(this, "Classify rasters") {
			
			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected void start() {
				String flag = (String)getItemControl().getSelectedItem();
				dispose();
				
				switch (flag) {
				case BASE_VIEW:
					new RasterClassfier(baseRasters.queryItemRasters(), flag).setVisible(true);;
					break;
				case GEN_VIEW:
					new RasterClassfier(genRasters.queryItemRasters(), flag).setVisible(true);;
					break;
				case RECOVER_VIEW:
					new RasterClassfier(recoverRasters.queryItemRasters(), flag).setVisible(true);;
					break;
				default:
					new RasterClassfier(baseRasters.queryItemRasters(), flag).setVisible(true);;
					break;
				}
			}
			
			@Override
			protected JComboBox<?> createItemControl() {
				return new JComboBox<String>(classifyRasters.toArray(new String[] {}));
			}
			
		};
		dlgStarter.getItemControl().setSelectedItem(BASE_VIEW);
		dlgStarter.setSize(DIALOG_INFO_SIZE);
		dlgStarter.setLocationRelativeTo(this);
        dlgStarter.setVisible(true);

	}
	
	
	/**
	 * This class represents raster classifier.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	protected class RasterClassfier extends ClassifierUI {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with parent component and title.
		 * @param trainingRasters training rasters.
		 * @param view specified view.
		 */
		public RasterClassfier(List<Raster> trainingRasters, String view) {
			super(trainingRasters, view);
		}
		
		@Override
		protected Classifier getClassifier() {
			if (gm == null) return (classifier = null);
			try {
				boolean isNorm = gm.queryConfig().getAsBoolean(Raster.NORM_FIELD);
				return classifier != null ? classifier : StackClassifier.create(gm.getRasterChannel(), isNorm);
			} catch (Throwable e) {Util.trace(e);}
			return (classifier = null);
		}

		/**
		 * Getting training directory.
		 * @return training directory.
		 */
		private Path getTrainDir() {
			Path dir = null;
			switch (view) {
			case BASE_VIEW:
				dir = getBaseDir();
				break;
			case GEN_VIEW:
				dir = getGenDir();
				break;
			case RECOVER_VIEW:
				dir = getRecoverDir();
				break;
			default:
				dir = getBaseDir();
				break;
			}
			return dir;
		}
		
		@Override
		protected void loadTrainDir() {
			Path trainDir = getTrainDir();
			if (trainDir == null) {
				JOptionPane.showMessageDialog(this, "Wrong training directory", "Wrong training directory", JOptionPane.ERROR_MESSAGE);
				return;
			}
			if (chkAllowAdd.isSelected() || LOAD_RASTER_ALWAYS)
				addRasters(trainDir, trainRasters);
			else
				trainRasters.setListData(trainDir, imageListIconSize, imageListStoreImage);
		}

		/**
		 * Adding training rasters.
		 */
		private void addTrainRasters() {
			addRasters(trainRasters, getTrainDir());
		}

		/**
		 * Adding training CIFAR rasters.
		 */
		private void addTrainRastersCIFAR() {
			if (chkLoad3D.isSelected()) {
				JOptionPane.showMessageDialog(this, "Loading 3D not allowed", "Loading 3D not allowed", JOptionPane.ERROR_MESSAGE);
				return;
			}
			addRastersCIFAR(trainRasters, getTrainDir());
		}

		/**
		 * Adding training rasters by folders.
		 */
		private void addTrainRastersFolders() {
			addRastersFolders(trainRasters, getTrainDir());
		}

		@Override
		protected void addTrainRastersStarter() {
			List<String> addRasters = Util.newList(0);
			addRasters.addAll(Arrays.asList(ADD_RASTERS));
			Collections.sort(addRasters);
			final StartDlg dlgStarter = new StartDlg(this, "Add tested rasters") {
				
				/**
				 * Serial version UID for serializable class.
				 */
				private static final long serialVersionUID = 1L;

				@Override
				protected void start() {
					String flag = (String)getItemControl().getSelectedItem();
					switch (flag) {
					case ADD_RASTERS_NORMAL:
						addTrainRasters();
						break;
					case ADD_RASTERS_CIFAR:
						addTrainRastersCIFAR();
						break;
					case ADD_RASTERS_FOLDERS:
						addTrainRastersFolders();
						break;
					default:
						addTrainRasters();
						break;
					}
					dispose();
				}
				
				@Override
				protected JComboBox<?> createItemControl() {
					return new JComboBox<String>(addRasters.toArray(new String[] {}));
				}
				
			};
			dlgStarter.getItemControl().setSelectedItem(ADD_RASTERS_FOLDERS);
			dlgStarter.setSize(DIALOG_INFO_SIZE);
			dlgStarter.setLocationRelativeTo(this);
	        dlgStarter.setVisible(true);
		}

		/**
		 * Getting tested directory.
		 * @return tested directory.
		 */
		private Path getTestDir() {
			return getRecoverDir();
		}
		
		@Override
		protected void loadTestDir() {
			Path testDir = getTestDir();
			if (testDir == null) {
				JOptionPane.showMessageDialog(this, "Wrong tested directory", "Wrong tested directory", JOptionPane.ERROR_MESSAGE);
				return;
			}
			if (chkAllowAdd.isSelected() || LOAD_RASTER_ALWAYS)
				addRasters(testDir, testRasters);
			else
				testRasters.setListData(testDir, imageListIconSize, imageListStoreImage);
		}

		/**
		 * Adding tested rasters.
		 */
		private void addTestRasters() {
			addRasters(testRasters, getTestDir());
		}

		/**
		 * Adding tested CIFAR rasters.
		 */
		private void addTestRastersCIFAR() {
			if (chkLoad3D.isSelected()) {
				JOptionPane.showMessageDialog(this, "Loading 3D not allowed", "Loading 3D not allowed", JOptionPane.ERROR_MESSAGE);
				return;
			}
			addRastersCIFAR(testRasters, getTestDir());
		}

		/**
		 * Adding tested rasters by folders.
		 */
		private void addTestRastersFolders() {
			addRastersFolders(testRasters, getTestDir());
		}

		@Override
		protected void addTestRastersStarter() {
			List<String> addRasters = Util.newList(0);
			addRasters.addAll(Arrays.asList(ADD_RASTERS));
			Collections.sort(addRasters);
			final StartDlg dlgStarter = new StartDlg(this, "Add tested rasters") {
				
				/**
				 * Serial version UID for serializable class.
				 */
				private static final long serialVersionUID = 1L;

				@Override
				protected void start() {
					String flag = (String)getItemControl().getSelectedItem();
					switch (flag) {
					case ADD_RASTERS_NORMAL:
						addTestRasters();
						break;
					case ADD_RASTERS_CIFAR:
						addTestRastersCIFAR();
						break;
					case ADD_RASTERS_FOLDERS:
						addTestRastersFolders();
						break;
					default:
						addTestRasters();
						break;
					}
					dispose();
				}
				
				@Override
				protected JComboBox<?> createItemControl() {
					return new JComboBox<String>(addRasters.toArray(new String[] {}));
				}
				
			};
			dlgStarter.getItemControl().setSelectedItem(ADD_RASTERS_NORMAL);
			dlgStarter.setSize(DIALOG_INFO_SIZE);
			dlgStarter.setLocationRelativeTo(this);
	        dlgStarter.setVisible(true);
		}
		
	}
	
	
	@Override
	public void inspect() {
		setVisible(true);
	}


	@Override
	public void receivedSetup(SetupAlgEvent evt) throws RemoteException {
		if (taskBackground) {
			if (prgRunning.getMaximum() < evt.getProgressTotalEstimated())
				prgRunning.setMaximum(evt.getProgressTotalEstimated());
			if (prgRunning.getValue() < evt.getProgressStep()) 
				prgRunning.setValue(evt.getProgressStep());
		}
	}


	@Override
	public boolean classPathContains(String className) throws RemoteException {
    	try {
    		net.hudup.core.Util.getPluginManager().loadClass(className, false);
    		return true;
    	} catch (Exception e) {}
    	
		return false;
	}


	@Override
	public boolean ping() throws RemoteException {
		return false;
	}


	@Override
	public void dispose() {
		super.dispose();
		
		if (runner != null) {
			runner.stop();
			runner = null;
		}

	    try {
	    	if (gm != null) gm.removeSetupListener(this);
	    } catch (Throwable e) {Util.trace(e);}
	    
	}

	
	/**
	 * Querying local generative model.
	 * @return local generative model.
	 */
	public static GenUI queryLocalGenModel() {
		return queryLocalGenModel(null, null);
	}

	
	/**
	 * Querying local generative model.
	 * @param initialGM initial generative model.
	 * @return local generative model.
	 */
	public static GenUI queryLocalGenModel(GenModelRemote initialGM) {
		return queryLocalGenModel(initialGM, null);
	}
	
	
	/**
	 * Querying local generative model.
	 * @param initialGM initial generative model.
	 * @param initialUI generative model UI.
	 * @return local generative model.
	 */
	protected static GenUI queryLocalGenModel(GenModelRemote initialGM, GenUI initialUI) {
		initialGM = (initialGM == null && initialUI != null) ? initialUI.gm : initialGM;
		int ret = JOptionPane.showConfirmDialog(initialUI, "Would you like to query automatically?", "Automatical query", JOptionPane.OK_CANCEL_OPTION);
		List<GenModelRemote> gmList = Util.newList(0);
		if (ret == JOptionPane.OK_OPTION) {
			try {
				gmList = net.hudup.core.Util.getPluginManager().loadInstances(GenModelRemote.class);
			}
			catch (Throwable e) {
				Util.trace(e);
				return new GenUI(new VAE(), false);
			}
		}
		else {
			gmList.add(new VAE());
			gmList.add(new GAN());
			gmList.add(new AVA());
			gmList.add(new AVAExt());
		}
		if (gmList.size() == 0) return initialUI;
		
		Collections.sort(gmList, new Comparator<GenModelRemote>() {

			@Override
			public int compare(GenModelRemote gm1, GenModelRemote gm2) {
				String name1 = null, name2 = null;
				try {
					name1 = gm1.queryName();
					name2 = gm2.queryName();
				} catch (Throwable e) {}
				
				return name1 != null && name2 != null ? name1.compareTo(name2) : 0;
			}
			
		});
		
		List<GenModelRemote> finalGMList = gmList;
		Wrapper selectedObject = new Wrapper(null);
		final StartDlg dlgStarter = new StartDlg(initialUI, "Generative models") {
			
			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected void start() {
				selectedObject.setObject((GenModelRemote)getItemControl().getSelectedItem());
				dispose();
			}
			
			@Override
			protected JComboBox<?> createItemControl() {
				return new JComboBox<GenModelRemote>(finalGMList.toArray(new GenModelRemote[] {}));
			}
			
		};

		GenModelRemote selectedGM = null;
		if (initialGM != null) {
			for (GenModelRemote gm : finalGMList) {
				try {
					if (gm.queryName().equals(initialGM.queryName())) selectedGM = gm;
				} catch (Throwable e) {}
			}
		}
		dlgStarter.getItemControl().setSelectedItem(selectedGM);
		
		dlgStarter.setSize(DIALOG_INFO_SIZE);
		dlgStarter.setLocationRelativeTo(initialUI);
        dlgStarter.setVisible(true);
		
        if (selectedObject.getObject() == null) return null;
        GenModelRemote gm = (GenModelRemote)selectedObject.getObject();
        if (initialUI != null) {
        	try {
	        	if (!initialGM.queryName().equals(gm.queryName())) {
	        		initialUI.gm = gm;
	        		initialUI.reset();
	        		return initialUI;
	        	}
	        	else
	        		return initialUI;
        	} catch (Throwable e) {Util.trace(e);}
        	return initialUI;
        }
        else
        	return new GenUI(gm, false);
	}
	
	
	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		GenUI genUI = queryLocalGenModel(new VAE());
		if (genUI != null) genUI.setVisible(true);
	}
	
	
}
