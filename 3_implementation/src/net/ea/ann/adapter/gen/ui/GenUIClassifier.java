/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.ui;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.List;

import javax.swing.JMenuBar;
import javax.swing.JOptionPane;

import net.ea.ann.adapter.gen.ClassifierModelAbstract;
import net.ea.ann.adapter.gen.GenModelRemote;
import net.ea.ann.adapter.gen.beans.ForestClassifier;
import net.ea.ann.adapter.gen.beans.MatrixClassifier;
import net.ea.ann.adapter.gen.beans.NiN;
import net.ea.ann.adapter.gen.beans.StackClassifier;
import net.ea.ann.adapter.gen.beans.TransformerClassifier;
import net.ea.ann.adapter.gen.beans.VGG;
import net.ea.ann.classifier.Classifier;
import net.ea.ann.classifier.ClassifierAbstract;
import net.ea.ann.classifier.ClassifierAssoc;
import net.ea.ann.classifier.ClassifierAssoc.ClassifyInfo;
import net.ea.ann.classifier.ClassifierAssoc.ClassifyParams;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.WeightSpec;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.hudup.core.data.DataConfig;

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
		
		this.txtGenNum.setValue(1);
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
	void changeModel() {
		if (isRunning()) {
			JOptionPane.showMessageDialog(this, "Some task running", "Some task running", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		if (!exclusive) queryLocalGenModel(gm, this, new GenModelRemote[] {new MatrixClassifier(), new StackClassifier()}, new GenUICreator() {
			@Override
			public GenUI create(GenModelRemote gm) {
				return new GenUIClassifier(gm, false);
			}
		});
	}


	@Override
	void recover() {
		long beginTime = System.currentTimeMillis();
		super.recover();
		long endTime = System.currentTimeMillis();

		List<Raster> sources = this.recoverRasters.queryItemRasters();
		List<Raster> results = this.genRasters.queryItemRasters();
		int n = Math.min(sources.size(), results.size());
		if (n == 0) return;
		try {
			DataConfig config = gm.queryConfig();
			ClassifyParams params = importParams(gm);
			params.dataset = "dataset";
			params.time = endTime - beginTime;
			if (config.containsKey(net.ea.ann.classifier.MatrixClassifier.EPOCHS_PSEUDO_FILED))
				params.maxIteration = config.getAsInt(net.ea.ann.classifier.MatrixClassifier.EPOCHS_PSEUDO_FILED);

			String classifiedName = RasterAssoc.genDefaultName(gm.queryName() + "-" + Util.format(params.learningRate), null);
			BufferedWriter listWriter = Files.newBufferedWriter(resultDir.resolve(classifiedName + "-" + "list" + ".csv"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			ClassifierAssoc.saveClassifyInfo(listWriter, sources, results);
			listWriter.close();
			
			ClassifyInfo info = new ClassifyInfo();
			info.collect(sources, results);
			BufferedWriter statWriter = Files.newBufferedWriter(resultDir.resolve(classifiedName + "-" + "stat" + ".csv"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			ClassifierAssoc.saveClassifyInfo(statWriter, info, params);
			statWriter.close();
		} catch (Throwable e) {Util.trace(e);}
		
	}

	
	/**
	 * Extracting parameters from generative model.
	 * @param gm generative model.
	 * @return parameters.
	 */
	private static ClassifyParams importParams(GenModelRemote gm) {
		ClassifyParams params = new ClassifyParams();
		try {
			DataConfig config = gm.queryConfig();
			params.model = ClassifierModelAbstract.extractClassifierModel(gm);
			if (config.containsKey(NetworkAbstract.LEARN_RATE_FIELD))
				params.learningRate = config.getAsReal(NetworkAbstract.LEARN_RATE_FIELD);
			if (config.containsKey(NetworkAbstract.LEARN_MAX_ITERATION_FIELD))
				params.batches = config.getAsInt(NetworkAbstract.LEARN_MAX_ITERATION_FIELD);
			if (config.containsKey(ClassifierAbstract.CONV_FIELD))
				params.conv = config.getAsBoolean(ClassifierAbstract.CONV_FIELD);
			if (config.containsKey(ClassifierAbstract.FILTER_SIZE_FIELD))
				params.filterSize = config.getAsInt(ClassifierAbstract.FILTER_SIZE_FIELD);
			if (config.containsKey(MatrixNetworkAbstract.VECTORIZED_FIELD))
				params.vectorized = config.getAsBoolean(MatrixNetworkAbstract.VECTORIZED_FIELD);
			if (config.containsKey(ClassifierAbstract.BASELINE_FIELD))
				params.baseline = config.getAsBoolean(ClassifierAbstract.BASELINE_FIELD);
			if (config.containsKey(ClassifierAbstract.ADJUST_FIELD))
				params.adjust = config.getAsBoolean(ClassifierAbstract.ADJUST_FIELD);
			if (config.containsKey(ClassifierAbstract.DUAL_FIELD))
				params.dual = config.getAsBoolean(ClassifierAbstract.DUAL_FIELD);
			if (config.containsKey(ClassifierAbstract.ENTROPY_TRAINER_FIELD))
				params.entropyTrainer = config.getAsBoolean(ClassifierAbstract.ENTROPY_TRAINER_FIELD);
			if (config.containsKey(net.ea.ann.classifier.TransformerClassifier.BLOCKS_NUMBER_FIELD))
				params.blocks = config.getAsInt(net.ea.ann.classifier.TransformerClassifier.BLOCKS_NUMBER_FIELD);
			if (config.containsKey(net.ea.ann.mane.beans.VGG.POOL_TYPE_FIELD))
				params.poolType = FilterSpec.intToPoolType(config.getAsInt(net.ea.ann.mane.beans.VGG.POOL_TYPE_FIELD));
			if (config.containsKey(net.ea.ann.mane.beans.VGG.WEIGHT_TYPE_FIELD))
				params.weightType = WeightSpec.intToType(config.getAsInt(net.ea.ann.mane.beans.VGG.WEIGHT_TYPE_FIELD));
			if (config.containsKey(net.ea.ann.mane.beans.VGG.FILTERS_NUMBER_FIELD))
				params.filtersNumber = config.getAsInt(net.ea.ann.mane.beans.VGG.FILTERS_NUMBER_FIELD);
			if (config.containsKey(net.ea.ann.mane.beans.VGG.MIDDLE_SIZE_FIELD))
				params.middleSize = net.ea.ann.mane.beans.VGG.paramGetVGGMiddleSize(config.getAsString(net.ea.ann.mane.beans.VGG.MIDDLE_SIZE_FIELD));
			if (config.containsKey(net.ea.ann.mane.beans.VGG.FFN_LENGTH_FIELD))
				params.ffnLength = config.getAsInt(net.ea.ann.mane.beans.VGG.FFN_LENGTH_FIELD);
			if (config.containsKey(net.ea.ann.classifier.ForestClassifier.TREE_MODEL_FIELD))
				params.treeModel = net.ea.ann.classifier.ForestClassifier.toTreeModel(config.getAsInt(net.ea.ann.classifier.ForestClassifier.TREE_MODEL_FIELD));
			if (config.containsKey(MatrixNetworkAbstract.EPOCHS_PSEUDO_FILED))
				params.maxIteration = config.getAsInt(MatrixNetworkAbstract.EPOCHS_PSEUDO_FILED);
			Classifier classifier = gm.getParameter() != null && gm.getParameter() instanceof Classifier ? (Classifier)gm.getParameter() : null;
			if (classifier != null) {
				params.depth = new ClassifierAssoc(classifier).depth();
				params.paramSize = new ClassifierAssoc(classifier).sizeOfParams();
			}
		} catch (Throwable e) {Util.trace(e);}
		return params;
	}
	
	
	/**
	 * Querying local generative model.
	 * @param initialGM initial generative model.
	 * @param initialUI generative model UI.
	 * @return local generative model.
	 */
	static GenUI queryLocalGenModel(GenModelRemote initialGM, GenUI initialUI) {
		return queryLocalGenModel(initialGM, initialUI, new GenModelRemote[] {new VGG(), new NiN(), new MatrixClassifier(), new TransformerClassifier(), new ForestClassifier(), new StackClassifier()}, new GenUICreator() {
			@Override
			public GenUI create(GenModelRemote gm) {
				return new GenUIClassifier(gm, false);
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
			GenUIClassifier classifierUI = (GenUIClassifier)queryLocalGenModel(new VGG(), null);
			if (classifierUI != null) classifierUI.setVisible(true);
		}
		else if (arg.equalsIgnoreCase("mane") || arg.equalsIgnoreCase("gen") || arg.equalsIgnoreCase("general"))
			ClassifierAssoc.classifyGen(System.in, System.out);
		else
			ClassifierAssoc.classify(System.in, System.out);
	}


}
