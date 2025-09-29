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
import java.util.List;

import net.ea.ann.adapter.Util;
import net.ea.ann.adapter.gen.ui.GenUIClassifier;
import net.ea.ann.classifier.Classifier;
import net.ea.ann.conv.stack.StackNetworkAbstract;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.gen.GenModel.G;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterWrapper;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.ExecuteAsLearnAlgAbstract;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Pointer;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.Inspector;

/**
 * This class implements partially classifier model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ClassifierModelAbstract extends ExecuteAsLearnAlgAbstract implements GenModel, GenModelRemote, NetworkListener, /*AllowNullTrainingSet,*/ DuplicatableAlg {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of neuron channel field.
	 */
	public final static String NEURON_CHANNEL_FIELD = "cma_neuron_channel";

	
	/**
	 * Default neuron channel.
	 */
	public static final int NEURON_CHANNEL_DEFAULT = GenModelAbstract.NEURON_CHANNEL_DEFAULT;

	
	/**
	 * Name of raster channel field.
	 */
	public final static String RASTER_CHANNEL_FIELD = "cma_raster_channel";

	
	/**
	 * Default raster channel.
	 */
	public static final int RASTER_CHANNEL_DEFAULT = GenModelAbstract.RASTER_CHANNEL_DEFAULT;

	
	/**
	 * Internal classifier.
	 */
	protected Classifier gm = null;
	
	
	/**
	 * Default constructor.
	 */
	public ClassifierModelAbstract() {
		super();
		gm = createGenModel();
		
		try {
			if (gm instanceof Network) config.putAll(Util.toConfig(((Network)gm).getConfig()));
		} catch (Throwable e) {Util.trace(e);}
		
		try {
			if (gm instanceof Network) ((Network)gm).addListener(this);
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
		if (gm == null) return;
		try {
			if (gm instanceof StackNetworkAbstract)
				((StackNetworkAbstract)gm).reset();
			else if (gm instanceof MatrixNetworkAbstract)
				((MatrixNetworkAbstract)gm).reset();
		} catch (Exception e) {Util.trace(e);}
	}

	
	@Override
	public synchronized Remote export(int serverPort) throws RemoteException {
		Remote remote = super.export(serverPort);
		try {
			if (gm != null && gm instanceof Network) ((Network)gm).export(serverPort);
		} catch (Throwable e) {Util.trace(e);}
		return remote;
	}


	@Override
	public synchronized void unexport() throws RemoteException {
		super.unexport();
		try {
			if (gm != null && gm instanceof Network) ((Network)gm).unexport();
		} catch (Throwable e) {Util.trace(e);}
	}


	@Override
	public synchronized void forceUnexport() throws RemoteException {
		super.forceUnexport();
		try {
			if (gm != null && gm instanceof Network) ((Network)gm).unexport();
		} catch (Throwable e) {Util.trace(e);}
	}

	
	@Override
	public Object executeAsLearn(Object input) throws RemoteException {
		try {
			if (gm != null && gm instanceof Network) ((Network)gm).getConfig().putAll(Util.transferToANNConfig(config));
		} catch (Throwable e) {Util.trace(e);}

		if (input == null) return null; //Running in setup method.

		if (!(input instanceof Profile)) return null;

		//Running in execution mode.
		Profile profile = (Profile)input;
		if (profile.getAttCount() < 2) return null;
		
		String sourceText = profile.getValueAsString(0);
		if (sourceText == null) return null;
		Path sourceDirectory = Paths.get(sourceText);
		
		List<Raster> rasters = RasterAssoc.load(sourceDirectory);
		if (rasters.size() == 0) return null;

		List<Raster> results = gm.classify(rasters);
		return results.size();
	}


	/**
	 * Create classifier.
	 * @return classifier.
	 */
	protected abstract Classifier createGenModel();


	/**
	 * Creating and updating generative model.
	 * @return generative model.
	 */
	Classifier createUpdateGenModel() {
		try {
			if (gm == null || gm.getNeuronChannel() != getNeuronChannel())
				gm = createGenModel();
			if (gm instanceof Network) ((Network)gm).getConfig().putAll(Util.transferToANNConfig(config));
		} catch (Throwable e) {Util.trace(e);}
		return gm;
	}

	
	@Override
	public List<Raster> genRasters(Iterable<Raster> sample, int nGens) throws RemoteException {
		createUpdateGenModel();
		
//		classifier.learnRaster(sample);
		return gm.classify(sample);
	}


	@Override
	public List<Raster> genRasters(int nGens) throws RemoteException {
		throw new IllegalArgumentException("Lack of sample");
	}


	@Override
	public List<G> recoverRasters(Iterable<Raster> sample, Iterable<Raster> rasters, int nGens) throws RemoteException {
		createUpdateGenModel();
		
		gm.learnRaster(sample);
		List<Raster> results = gm.classify(rasters);
		List<G> glist = Util.newList(results.size());
		for (Raster result : results) {
			G g = new G();
			g.xgenUndefined = result;
			glist.add(g);
			if (!(result instanceof RasterWrapper)) continue;
			
			RasterWrapper wResult = (RasterWrapper)result;
			Raster source = null;
			for (Raster raster : rasters) {
				if (wResult.getRaster() == raster) {
					source = raster;
					break;
				}
			}
			if (source == null) continue;
			
			RasterProperty sourceProperty = source.getProperty();
			RasterProperty resultProperty = result.getProperty();
			int labelCount = Math.min(sourceProperty.getLabelCount(), resultProperty.getLabelCount());
			int correct = 0;
			for (int i = 0; i < labelCount; i++) {
				int sourceLabelId = sourceProperty.getLabelId(i);
				int resultLabelId = resultProperty.getLabelId(i);
				if (sourceLabelId == resultLabelId) correct++;
			}
			if (labelCount > 0) g.error = 1.0 - (double)correct/(double)labelCount;
		}
		return glist;
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
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	protected boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}

	
	@Override
	public Object getParameter() throws RemoteException {
		return gm;
	}


	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		if (parameter == null)
			return "";
		else if (parameter instanceof StackNetworkAbstract)
			return "Stack classifier";
		else if (parameter instanceof MatrixNetworkAbstract)
			return "Matrix classifier";
		else
			return parameter.toString();
	}

	
	@Override
	public synchronized String getDescription() throws RemoteException {
		return parameterToShownText(getParameter());
	}

	
	@Override
	public Inspector getInspector() {
		return new GenUIClassifier(this, true);
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
		config.put(NEURON_CHANNEL_FIELD, NEURON_CHANNEL_DEFAULT);
		config.put(RASTER_CHANNEL_FIELD, RASTER_CHANNEL_DEFAULT);
		config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		
		try {
			if (gm instanceof Network) config.putAll(Util.toConfig(((Network)gm).getConfig()));
		} catch (Throwable e) {Util.trace(e);}

		return config;
	}


}
