/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.ir.Corpus.RasterCorpus;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.mane.beans.VGG;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.Size;

/**
 * This class implements default search engine.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SearchEngineDefault extends SearchEngineAbstract<RasterRecord, MatrixFeature> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for maximum size.
	 */
	public final static String MAX_INPUT_SIZE_FIELD = "se_max_input_size";
	
	
	/**
	 * Default value for maximum size.
	 */
	public final static Size MAX_INPUT_SIZE_DEFAULT = new Size(2*VGG.MINSIZE, 2*VGG.MINSIZE);

	
	/**
	 * Default text value for maximum size.
	 */
	public final static String MAX_INPUT_SIZE_DEFAULT_TEXT = 2*VGG.MINSIZE + ", " + 2*VGG.MINSIZE;

	
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();

	
	/**
	 * Information retrieval system.
	 */
	protected IRAbstract<RasterRecord, MatrixFeature> ir = null;
	
	
	/**
	 * Default constructor.
	 */
	public SearchEngineDefault() {
		this.config.put(RasterAbstract.NEURON_CHANNEL_FIELD, RasterAbstract.NEURON_CHANNEL_DEFAULT);
		this.config.put(VGG.MIDDLE_SIZE_FIELD, VGG.MIDDLE_SIZE_DEFAULT_TEXT);
		this.config.put(MAX_INPUT_SIZE_FIELD, MAX_INPUT_SIZE_DEFAULT_TEXT);
		
		if (this.ir instanceof IRDefault) ((IRDefault)this.ir).getConfig().putAll(this.config);
		if (this.irShadow instanceof IRDefault) ((IRDefault)this.irShadow).getConfig().putAll(this.config);
	}


	@Override
	public List<RasterRecord> search(RasterRecord query) throws RemoteException {
		MatrixFeature featureQuery = ir.extractor().featureOf(query);
		if (featureQuery == null) return Util.newList(0);
		List<ScoredFeature<MatrixFeature>> scoredFeatures = ir.search(featureQuery);
		List<RasterRecord> results = Util.newList(0);
		for (ScoredFeature<MatrixFeature> scoredFeature : scoredFeatures) {
			Record record = scoredFeature.getRecord();
			if (record != null && record instanceof RasterRecord) results.add((RasterRecord)record);
		}
		return results;
	}


	@Override
	protected IRAbstract<RasterRecord, MatrixFeature> createIR() {
		IRDefault ir = new IRProxyNCA();
		if (this.config != null) ir.getConfig().putAll(this.config);
		return ir;
	}

	
	/**
	 * Getting information retrieval system.
	 * @return information retrieval system.
	 */
	private IRDefault ir() {return (IRDefault)this.ir;}
	

	@Override
	protected Crawler<RasterRecord> createCrawler() {
		System.out.println("Crawler not implemented yet");
		return null;
	}

	
	/**
	 * Building search engine from corpus.
	 * @param corpus corpus.
	 * @param refresh refreshment flag.
	 * @return true if building is successful.
	 */
	public synchronized boolean build(RasterCorpus corpus, boolean refresh) {
		ir().getConfig().putAll(this.config);
		return ir().build(corpus, refresh);
	}

	
	@Override
	protected void taskBackground() {
		
	}


	@Override
	protected void clearBackground() {
		
	}


	/**
	 * Getting configuration.
	 * @return configuration.
	 */
	public NetworkConfig getConfig() {return config;}

	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	int paramGetNeuronChannel() {
		int neuronChannel = config.getAsInt(RasterAbstract.NEURON_CHANNEL_FIELD);
		return neuronChannel < 1 ? RasterAbstract.NEURON_CHANNEL_DEFAULT : neuronChannel;
	}
	
	
	/**
	 * Setting neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return this model.
	 */
	SearchEngineDefault paramSetNeuronChannel(int neuronChannel) {
		neuronChannel = neuronChannel < 1 ? RasterAbstract.NEURON_CHANNEL_DEFAULT : neuronChannel;
		config.put(RasterAbstract.NEURON_CHANNEL_FIELD, neuronChannel);
		return this;
	}
	
	
	/**
	 * Getting VGG middle size.
	 * @return VGG middle size.
	 */
	Size paramGetVGGMiddleSize() {
		String sizeText = config.containsKey(VGG.MIDDLE_SIZE_FIELD) ? config.getAsString(VGG.MIDDLE_SIZE_FIELD) : VGG.MIDDLE_SIZE_DEFAULT_TEXT;
		return VGG.paramGetVGGMiddleSize(sizeText);
	}
	
	
	/**
	 * Setting VGG middle size.
	 * @param middleSize VGG middle size.
	 * @return this model.
	 */
	SearchEngineDefault paramSetVGGMiddleSize(Size middleSize) {
		int width = middleSize.width < 1 ? VGG.MINSIZE : middleSize.width;
		int height = middleSize.height < 1 ? VGG.MINSIZE : middleSize.height;
		config.put(VGG.MIDDLE_SIZE_FIELD, width + ", " + height);
		return this;
	}


	/**
	 * Getting maximum input size.
	 * @return maximum input size.
	 */
	Size paramGetMaxInputSize() {
		String sizeText = config.containsKey(MAX_INPUT_SIZE_FIELD) ? config.getAsString(MAX_INPUT_SIZE_FIELD) : MAX_INPUT_SIZE_DEFAULT_TEXT;
		return VGG.paramGetVGGMiddleSize(sizeText);
	}
	
	
	/**
	 * Setting maximum input size.
	 * @param maxInputSize maximum input size.
	 * @return this model.
	 */
	SearchEngineDefault paramSetMaxInputSize(Size maxInputSize) {
		int width = maxInputSize.width < 1 ? 2*VGG.MINSIZE : maxInputSize.width;
		int height = maxInputSize.height < 1 ? 2*VGG.MINSIZE : maxInputSize.height;
		config.put(MAX_INPUT_SIZE_FIELD, width + ", " + height);
		return this;
	}


}
