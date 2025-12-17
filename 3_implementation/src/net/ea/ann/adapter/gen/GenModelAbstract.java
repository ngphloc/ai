/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import net.ea.ann.adapter.Util;
import net.ea.ann.adapter.gen.ui.GenUI;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.gen.ConvGenModel;
import net.ea.ann.gen.ConvGenModelAssoc;
import net.ea.ann.gen.GenModel.G;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2D;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.ExecuteAsLearnAlgAbstract;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Pointer;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.ui.JImageList.ImageListItem;

/**
 * This class implements partially generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class GenModelAbstract extends ExecuteAsLearnAlgAbstract implements GenModel, GenModelRemote, NetworkListener, /*AllowNullTrainingSet,*/ DuplicatableAlg {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of Z dimension field.
	 */
	public final static String ZDIM_FIELD = "gma_zdim";
	
	
	/**
	 * Default value of Z dimension field.
	 */
	public final static int ZDIM_DEFAULT = net.ea.ann.gen.GenModel.ZDIM_DEFAULT;

	
	/**
	 * Name of zoom-out field.
	 */
	public final static String ZOOMOUT_FIELD = "gma_zoomout";

	
	/**
	 * Default value of zoom-out field.
	 */
	public final static int ZOOMOUT_DEFAULT = NetworkAbstract.ZOOMOUT_DEFAULT;

	
	/**
	 * Name of minimum width field.
	 */
	public final static String XMINWIDTH_FIELD = "gma_xminwidth";

	
	/**
	 * Default value of minimum width field.
	 */
	public final static int XMINWIDTH_DEFAULT = ImageListItem.ICON_MINSIZE / ZOOMOUT_DEFAULT;

	
	/**
	 * Name of minimum height field.
	 */
	public final static String XMINHEIGHT_FIELD = "gma_xminheight";

	
	/**
	 * Default value of minimum height field.
	 */
	public final static int XMINHEIGHT_DEFAULT = XMINWIDTH_DEFAULT;

	
	/**
	 * Name of random recovering field.
	 */
	public final static String RECOVER_RANDOM_FIELD = "gma_recover_random";

	
	/**
	 * Default value of random recovering field.
	 */
	public final static boolean RECOVER_RANDOM_DEFAULT = true;

	
	/**
	 * Name of saving recovering field.
	 */
	public final static String RECOVER_SAVE_FIELD = "gma_recover_save";

	
	/**
	 * Default value of saving recovering field.
	 */
	public final static boolean RECOVER_SAVE_DEFAULT = false;

	
	/**
	 * Name of neuron channel field.
	 */
	public final static String NEURON_CHANNEL_FIELD = RasterAbstract.NEURON_CHANNEL_FIELD;

	
	/**
	 * Default neuron channel.
	 */
	public static final int NEURON_CHANNEL_DEFAULT = RasterAbstract.NEURON_CHANNEL_DEFAULT;

	
	/**
	 * Name of raster channel field.
	 */
	public final static String RASTER_CHANNEL_FIELD = RasterAbstract.RASTER_CHANNEL_FIELD;

	
	/**
	 * Default raster channel.
	 */
	public final static int RASTER_CHANNEL_DEFAULT = RasterAbstract.RASTER_CHANNEL_DEFAULT;
	
	
	/**
	 * Name of filters field.
	 */
	public final static String FILTERS_FIELD = "gma_filters";
	
	
	/**
	 * Default number of generated generated rasters.
	 */
	public final static int GENS_DEFAULT = 10;

	
	/**
	 * Internal generative model.
	 */
	protected ConvGenModel gm = null;
	
	
	/**
	 * Default constructor.
	 */
	public GenModelAbstract() {
		super();
		gm = createGenModel();
		
		try {
			config.putAll(Util.toConfig(gm.getConfig()));
		} catch (Throwable e) {Util.trace(e);}
		
		try {
			gm.addListener(this);
		} catch (Throwable e) {Util.trace(e);}
	}

	
	@Override
	protected Object fetchSample(Dataset dataset) {
		return (dataset != null) && !(dataset instanceof Pointer) ? dataset.fetchSample2() : null;
	}


	@Override
	public void setup(Dataset dataset, Object... info) throws RemoteException {
		super.setup(dataset, info);
	}


	@Override
	public synchronized void unsetup() throws RemoteException {
		super.unsetup();
		try {
			if (gm != null) gm.reset();
		} catch (Exception e) {Util.trace(e);}
	}


	@Override
	public synchronized Remote export(int serverPort) throws RemoteException {
		Remote remote = super.export(serverPort);
		try {
			if (gm != null) gm.export(serverPort);
		} catch (Throwable e) {Util.trace(e);}
		return remote;
	}


	@Override
	public synchronized void unexport() throws RemoteException {
		super.unexport();
		try {
			if (gm != null) gm.unexport();
		} catch (Throwable e) {Util.trace(e);}
	}


	@Override
	public synchronized void forceUnexport() throws RemoteException {
		super.forceUnexport();
		try {
			if (gm != null) gm.unexport();
		} catch (Throwable e) {Util.trace(e);}
	}

	
	@SuppressWarnings("unchecked")
	@Override
	public Object executeAsLearn(Object input) throws RemoteException {
		gm.getConfig().putAll(Util.transferToANNConfig(config));
		
		ConvGenModelAssoc assoc = new ConvGenModelAssoc(gm, config.getAsBoolean(NetworkAbstract.LEARN_ONE_FIELD));
		
		if (input == null) { //Running in setup method.
			if (sample == null) return null;
			int count = 0;
			for (Profile profile : (Collection<Profile>)sample) {
				if (profile == null || profile.getAttCount() < 2) continue;
				
				String sourceText = profile.getValueAsString(0);
				if (sourceText == null) return null;
				Path sourceDirectory = Paths.get(sourceText);
				String targetText = profile.getValueAsString(1);
				if (targetText == null) return null;
				Path targetDirectory = Paths.get(targetText);
	
				int nGens = GENS_DEFAULT;
				if (profile.getAttCount() > 2) nGens = profile.getValueAsInt(2);
				nGens = nGens <= 0 ? GENS_DEFAULT : nGens;
				
				int minWidth = getMinWidth();
				int minHeight = getMinHeight();
				
				List<Raster> rasters = RasterAssoc.load(sourceDirectory);
				if (rasters.size() == 0) continue;
				
				rasters = assoc.initGenRasters(rasters, nGens, getZDim(),
					SizeZoom.zoom(getZoomOut(), getZoomOut(), getDim(rasters) > 2 ? getZoomOut() : 1, 1),
					new Size(minWidth, minHeight, 1, 1));
				count += RasterAssoc.saveDirector(rasters, targetDirectory, getName());
			}
			
			return Double.valueOf(count);
		}

		if (!(input instanceof Profile)) return null;
		
		//Running in execution mode.
		Profile profile = (Profile)input;
		if (profile.getAttCount() < 2) return null;
		
		String sourceText = profile.getValueAsString(0);
		if (sourceText == null) return null;
		Path sourceDirectory = Paths.get(sourceText);
		String targetText = profile.getValueAsString(1);
		if (targetText == null) return null;
		Path targetDirectory = Paths.get(targetText);
		
		int nRecovs = GENS_DEFAULT;
		if (profile.getAttCount() > 2) nRecovs = profile.getValueAsInt(2);
		nRecovs = nRecovs <= 0 ? GENS_DEFAULT : nRecovs;
		
		List<Raster> rasters = RasterAssoc.load(sourceDirectory);
		if (rasters.size() == 0) return null;
		
		double error = 0;
		int count = 0;
		for (Raster raster : rasters) {
			ConvGenModel clonedGM = (ConvGenModel)net.ea.ann.core.Util.cloneBySerialize(gm);
			clonedGM.learnRasterOneByOne(Arrays.asList(raster));
			for (int k = 0; k < nRecovs; k++) {
				G g = clonedGM.recoverRaster(raster, null, config.getAsBoolean(RECOVER_RANDOM_FIELD), true);
				if ((g == null) || (g.xgenUndefined == null) || !(g.xgenUndefined instanceof Raster))
					continue;
				
				error += g.error;
				count++;
				
				if (RECOVER_SAVE_DEFAULT) {
					Path path = RasterAssoc.genDefaultPath(targetDirectory, raster.getDefaultFormat(), getName());
					((Raster)g.xgenUndefined).save(path);
				}
			}
		}
		
		if (count > 0)
			return Double.valueOf(error/count);
		else
			return null;
	}


	/**
	 * Create generative model instance.
	 * @return generative model instance.
	 */
	protected abstract ConvGenModel createGenModel();
	
	
	/**
	 * Creating and updating generative model.
	 * @return generative model.
	 */
	ConvGenModel createUpdateGenModel() {
		try {
			ConvGenModelAssoc assoc = new ConvGenModelAssoc(gm);
			if (gm == null || gm.getNeuronChannel() != getNeuronChannel() || gm.getRasterChannel() != getRasterChannel() || assoc.isNorm() != isNorm())
				gm = createGenModel();
			gm.getConfig().putAll(Util.transferToANNConfig(config));
		} catch (Throwable e) {Util.trace(e);}
		return gm;
	}
	
	
	/**
	 * Getting generative model instance.
	 * @return generative model instance.
	 */
	public ConvGenModel getGenModel() {
		return gm;
	}
	
	
	@Override
	public List<Raster> genRasters(Iterable<Raster> sample, int nGens) throws RemoteException {
		createUpdateGenModel();
		
		ConvGenModelAssoc assoc = new ConvGenModelAssoc(gm, config.getAsBoolean(NetworkAbstract.LEARN_ONE_FIELD));
		SizeZoom zoomOut = getSizeZoomOut(sample);
		Size minSize = new Size(getMinWidth(), getMinHeight(), 1, 1);
		return config.getAsBoolean(Raster2D.LEARN_FIELD) ?
			assoc.initGenRastersFeatureExtractor2D(sample, nGens, getZDim(), zoomOut, minSize)
			:
			assoc.initGenRasters(sample, nGens, getZDim(), zoomOut, minSize);
	}
	
	
	@Override
	public List<Raster> genRasters(int nGens) throws RemoteException {
		List<Raster> result = Util.newList(0);
		for (int i = 0; i < nGens; i++) {
			try {
				G g = nGens <= 1 ? gm.generateRasterBest() : gm.generateRaster();
				Raster raster = g != null ? g.getXGenRaster() : null;
				if (raster != null) result.add(raster);
			} catch (Exception e) {Util.trace(e);}
		}
		return result;
	}


	@Override
	public List<G> recoverRasters(Iterable<Raster> sample, Iterable<Raster> rasters, int nGens) throws RemoteException {
		createUpdateGenModel();

		ConvGenModelAssoc assoc = new ConvGenModelAssoc(gm, config.getAsBoolean(NetworkAbstract.LEARN_ONE_FIELD));
		SizeZoom zoomOut = getSizeZoomOut(sample);
		Size minSize = new Size(getMinWidth(), getMinHeight(), 1, 1);
		boolean randomGen = config.getAsBoolean(RECOVER_RANDOM_FIELD);
		return config.getAsBoolean(Raster2D.LEARN_FIELD) ?
			assoc.initRecoverRastersFeatureExtractor2D(getName(), sample, getZDim(), zoomOut, minSize, rasters, null, randomGen, nGens)
			:
			assoc.initRecoverRasters(getName(), sample, getZDim(), zoomOut, minSize, rasters, null, randomGen, nGens);
	}


	@Override
	public int getNeuronChannel() throws RemoteException {
		int neuronChannel = NEURON_CHANNEL_DEFAULT;
		if (config.containsKey(NEURON_CHANNEL_FIELD)) neuronChannel = config.getAsInt(NEURON_CHANNEL_FIELD);
		return neuronChannel < 1 ? NEURON_CHANNEL_DEFAULT : neuronChannel;
	}

	
	@Override
	public int getRasterChannel() throws RemoteException {
		int rasterChannel = RASTER_CHANNEL_DEFAULT;
		if (config.containsKey(RASTER_CHANNEL_FIELD)) rasterChannel = config.getAsInt(RASTER_CHANNEL_FIELD);
		return rasterChannel < 1 ? RASTER_CHANNEL_DEFAULT : rasterChannel;
	}

	
	/**
	 * Getting Z dimension.
	 * @return Z dimension.
	 */
	private int getZDim() {
		int zDim = ZDIM_DEFAULT;
		if (config.containsKey(ZDIM_FIELD)) zDim = config.getAsInt(ZDIM_FIELD);
		return zDim <= 0 ? ZDIM_DEFAULT : zDim;
	}
	
	
	/**
	 * Getting dimension of sample.
	 * @param sample specified sample.
	 * @return dimension of sample.
	 */
	private static int getDim(Iterable<Raster> sample) {
		for (Raster raster : sample) {
			if (raster != null) return new RasterAssoc(raster).getDim();
		}
		return 0;
	}
	
	
	/**
	 * Getting zooming out ratio.
	 * @return zooming out ratio.
	 */
	private int getZoomOut() {
		int zoomOut = ZOOMOUT_DEFAULT;
		if (config.containsKey(ZOOMOUT_FIELD)) zoomOut = config.getAsInt(ZOOMOUT_FIELD);
		return zoomOut < 1 ? ZOOMOUT_DEFAULT : zoomOut;
	}

	
	/**
	 * Getting size zooming out.
	 * @param sample sample.
	 * @return size zooming out.
	 */
	private SizeZoom getSizeZoomOut(Iterable<Raster> sample) {
		int zoomOut = getZoomOut();
		int dim = getDim(sample);
		return SizeZoom.zoom(zoomOut, dim>1?zoomOut:1, dim>2?zoomOut:1, 1);
	}
	
	
	/**
	 * Getting minimum width.
	 * @return minimum width.
	 */
	private int getMinWidth() {
		int minWidth = XMINWIDTH_DEFAULT;
		if (config.containsKey(XMINWIDTH_FIELD)) minWidth = config.getAsInt(XMINWIDTH_FIELD);
		return minWidth <= 0 ? XMINWIDTH_DEFAULT : minWidth;
	}

	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	protected boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}

	
	/**
	 * Getting minimum height.
	 * @return minimum height.
	 */
	private int getMinHeight() {
		int minHeight = XMINHEIGHT_DEFAULT;
		if (config.containsKey(XMINHEIGHT_FIELD)) minHeight = config.getAsInt(XMINHEIGHT_FIELD);
		return minHeight <= 0 ? XMINHEIGHT_DEFAULT : minHeight;
	}

	
	@Override
	public Object getParameter() throws RemoteException {
		return gm;
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		if (parameter == null)
			return "";
		else if (!(parameter instanceof net.ea.ann.gen.vae.ConvVAEImpl))
			return "";
		else
			return ((net.ea.ann.gen.vae.ConvVAEImpl)parameter).toString();
	}

	
	@Override
	public synchronized String getDescription() throws RemoteException {
		return parameterToShownText(getParameter());
	}

	
	@Override
	public Inspector getInspector() {
		return new GenUI(this, true);
	}

	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {GenModelRemote.class.getName()};
	}

	
	@Override
	public void receivedInfo(NetworkInfoEvent evt) throws RemoteException {

	}

	
	@Override
	public void receivedDo(NetworkDoEvent evt) throws RemoteException {
		if (evt.getType() == NetworkDoEvent.Type.doing) {
			fireSetupEvent(new SetupAlgEvent(this, Type.doing, getName(), null,
				evt.getLearnResult(),
				evt.getProgressStep(), evt.getProgressTotalEstimated()));
		}
		else if (evt.getType() == NetworkDoEvent.Type.done) {
			fireSetupEvent(new SetupAlgEvent(this, Type.done, getName(), null,
					evt.getLearnResult(),
					evt.getProgressStep(), evt.getProgressTotalEstimated()));
		}
	}


	@Override
	public void setName(String name) {
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(ZDIM_FIELD, ZDIM_DEFAULT);
		config.put(ZOOMOUT_FIELD, ZOOMOUT_DEFAULT);
		config.put(XMINWIDTH_FIELD, XMINWIDTH_DEFAULT);
		config.put(XMINHEIGHT_FIELD, XMINHEIGHT_DEFAULT);
		config.put(RECOVER_RANDOM_FIELD, RECOVER_RANDOM_DEFAULT);
		config.put(NetworkAbstract.LEARN_ONE_FIELD, NetworkAbstract.LEARN_ONE_DEFAULT);
		config.put(NEURON_CHANNEL_FIELD, NEURON_CHANNEL_DEFAULT);
		config.put(RASTER_CHANNEL_FIELD, RASTER_CHANNEL_DEFAULT);
		config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		
		try {
			if (gm != null) config.putAll(Util.toConfig(gm.getConfig()));
		} catch (Throwable e) {Util.trace(e);}

		return config;
	}

	
}
