/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir.ml;

import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.ir.Corpus.RasterCorpus;
import net.ea.ann.ir.Feature;
import net.ea.ann.ir.MLAbstract.RasterMatrixMLGeneral;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.mane.beans.VGG;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.Size;

/**
 * This class implements the deep metric learning component based on Proxy-NCA (Proxy-Neighborhood Component Analysis) algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProxyNCA extends RasterMatrixMLGeneral {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();
	
	
	/**
	 * Core Proxy-NCA model.
	 */
	protected net.ea.ann.mane.beans.ProxyNCA core = null;
	
	
	/**
	 * Default constructor.
	 */
	public ProxyNCA() {
		super();
		this.config.put(RasterAbstract.NEURON_CHANNEL_FIELD, RasterAbstract.NEURON_CHANNEL_DEFAULT);
		this.config.put(VGG.MIDDLE_SIZE_FIELD, VGG.MIDDLE_SIZE_DEFAULT_TEXT);
		
		this.core = createCore();
		try {
			if (this.core instanceof net.ea.ann.mane.beans.ProxyNCA)
				((net.ea.ann.mane.beans.ProxyNCA)this.core).getConfig().putAll(this.config);
			this.config.putAll(((net.ea.ann.mane.beans.ProxyNCA)this.core).getConfig());
		} catch (Throwable e) {Util.trace(e);}
	}

	
	/**
	 * Creating core Proxy-NCA model.
	 * @return core Proxy-NCA model.
	 */
	protected net.ea.ann.mane.beans.ProxyNCA createCore() {
		net.ea.ann.mane.beans.ProxyNCA core = new net.ea.ann.mane.beans.ProxyNCA(paramGetNeuronChannel());
		try {
			if (this.config != null) {
				core.getConfig().putAll(this.config);
				this.config.putAll(core.getConfig());
			}
		} catch (Throwable e) {Util.trace(e);}
		return core;
	}
	
	
	@Override
	public Feature.MatrixFeature featureOf(RasterRecord record) {
		if (core == null || record == null || record.raster() == null) return null;
		Matrix output = core.evaluate(record.raster());
		return output != null ? new Feature.MatrixFeature(output, record.raster()) : null;
	}


	/**
	 * Getting configuration.
	 * @return configuration.
	 */
	public NetworkConfig getConfig() {return config;}
	
	
	@Override
	public boolean build(RasterCorpus corpus, boolean refresh) {
		if (core == null) return false;
		try {
			core.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		
		if (refresh) {
			Size middleSize = paramGetVGGMiddleSize();
			return core.learnRaster(corpus.rasters(), middleSize, middleSize) != null;
		}
		else
			return core.learnRaster(corpus.rasters()) != null;
	}
	
	
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
	ProxyNCA paramSetNeuronChannel(int neuronChannel) {
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
	ProxyNCA paramSetVGGMiddleSize(Size middleSize) {
		int width = middleSize.width < 1 ? VGG.MINSIZE : middleSize.width;
		int height = middleSize.height < 1 ? VGG.MINSIZE : middleSize.height;
		config.put(VGG.MIDDLE_SIZE_FIELD, width + ", " + height);
		return this;
	}
	
	
}
