/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import java.rmi.RemoteException;
import java.util.List;
import java.util.Set;

import net.ea.ann.adapter.Util;
import net.ea.ann.adapter.gen.ui.GenUIIR;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.gen.GenModel.G;
import net.ea.ann.ir.Corpus;
import net.ea.ann.ir.IRDefault;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterWrapper;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.ExecuteAsLearnAlgAbstract;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Pointer;
import net.hudup.core.logistic.Inspector;

/**
 * This class implements partially information retrieval (IR) system.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class IRAbstract extends ExecuteAsLearnAlgAbstract implements IR, IRRemote, NetworkListener, /*AllowNullTrainingSet,*/ DuplicatableAlg {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of neuron channel field.
	 */
	public final static String NEURON_CHANNEL_FIELD = GenModelAbstract.NEURON_CHANNEL_FIELD;

	
	/**
	 * Default neuron channel.
	 */
	public static final int NEURON_CHANNEL_DEFAULT = GenModelAbstract.NEURON_CHANNEL_DEFAULT;

	
	/**
	 * Name of raster channel field.
	 */
	public final static String RASTER_CHANNEL_FIELD = GenModelAbstract.RASTER_CHANNEL_FIELD;

	
	/**
	 * Default raster channel.
	 */
	public static final int RASTER_CHANNEL_DEFAULT = GenModelAbstract.RASTER_CHANNEL_DEFAULT;

	
	/**
	 * Reference to current IR.
	 */
	private IRDefault ir = null;

	
	/**
	 * Default constructor.
	 */
	public IRAbstract() {
		super();
		IRDefault ir = createIR();
		this.config.putAll(Util.toConfig(ir.getConfig()));
	}

	
	@Override
	protected Object fetchSample(Dataset dataset) {
		return (dataset != null) && !(dataset instanceof Pointer) ? dataset.fetchSample2() : null;
	}


	@Override
	public Object executeAsLearn(Object input) throws RemoteException {return null;}


	/**
	 * Create information retrieval (IR) system.
	 * @return information retrieval (IR) system.
	 */
	protected abstract IRDefault createIR();


	@Override
	public IRDefault createDelegate() {
		IRDefault ir = createIR();
		ir.getConfig().putAll(Util.transferToANNConfig(this.config));
		return this.ir = ir;
	}


	@Override
	public List<Raster> genRasters(Iterable<Raster> sample, int nGens) throws RemoteException {
		return searchRasters(sample, sample);
	}


	@Override
	public List<Raster> genRasters(int nGens) throws RemoteException {
		throw new IllegalArgumentException("Lack of sample");
	}

	
	@Override
	public List<G> recoverRasters(Iterable<Raster> sample, Iterable<Raster> rasters, int nGens) throws RemoteException {
		List<Raster> results = searchRasters(sample, rasters);
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
	
	
	/**
	 * Searching rasters.
	 * @param sample sample.
	 * @param rasters query rasters.
	 * @return list of found rasters.
	 */
	private List<Raster> searchRasters(Iterable<Raster> sample, Iterable<Raster> rasters) {
		IRDefault ir = createDelegate();
		Corpus.RasterCorpus corpus = Corpus.RasterCorpus.create(sample);
		ir.build(corpus, true);
		Set<Raster> results = Util.newSet(0);
		for (Raster raster : sample) {
			List<Raster> result = ir.search(raster);
			results.addAll(result);
		}
		
		List<Raster> resultList = Util.newList(results.size());
		resultList.addAll(results);
		return resultList;
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
	
	@Override
	public Object getParameter() throws RemoteException {return this.ir;}


	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		if (parameter == null)
			return "";
		else if (parameter instanceof IRDefault)
			return "IR";
		else
			return parameter.toString();
	}

	
	@Override
	public synchronized String getDescription() throws RemoteException {
		return parameterToShownText(getParameter());
	}

	
	@Override
	public Inspector getInspector() {return new GenUIIR(this, true);}

	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {GenModelRemote.class.getName()};
	}

	
	@Override
	public void receivedInfo(NetworkInfoEvent evt) throws RemoteException {}

	
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
		
		return config;
	}


}
