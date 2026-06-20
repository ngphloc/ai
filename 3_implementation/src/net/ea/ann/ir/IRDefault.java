/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.util.List;

import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.IRAbstract.IRSimple;
import net.ea.ann.ir.LibraryAbstract.RasterMatrixLibraryAbstract;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.mane.beans.VGG;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.Size;

/**
 * This class represents the default information Retrieval (IR) system.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class IRDefault extends IRSimple {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();

	
	/**
	 * Field of maximum found features.
	 */
	public final static String MAX_FOUND_FIELD = "ir_maxfound";
	
	
	/**
	 * Default value for maximum found features.
	 */
	public final static int MAX_FOUND_DEFAULT = 10;
	
	
	/**
	 * Default constructor.
	 */
	public IRDefault() {
		super();
		this.config.put(RasterAbstract.NEURON_CHANNEL_FIELD, RasterAbstract.NEURON_CHANNEL_DEFAULT);
		this.config.put(net.ea.ann.mane.beans.ProxyNCA.AUGMENTED_FIELD, net.ea.ann.mane.beans.ProxyNCA.AUGMENTED_DEFAULT);
		this.config.put(net.ea.ann.mane.beans.ProxyNCA.PIECE_SIZE_FIELD, net.ea.ann.mane.beans.ProxyNCA.PIECE_SIZE_DEFAULT);
		this.config.put(MAX_FOUND_FIELD, MAX_FOUND_DEFAULT);
		this.config.put(VGG.MIDDLE_SIZE_FIELD, VGG.MIDDLE_SIZE_DEFAULT_TEXT);
		this.config.put(VGG.FILTERS_NUMBER_MAX_FIELD, VGG.FILTERS_NUMBER_MAX_DEFAULT);
	}

	
	@Override
	protected RasterMatrixLibraryAbstract createLibrary() {return super.createLibrary();}

	
	/**
	 * Searching for raster.
	 * @param query query raster.
	 * @return found rasters.
	 */
	public List<Raster> search(Raster query) {
		MatrixFeature feature = extractor().featureOf(new RasterRecord(query));
		List<ScoredFeature<MatrixFeature>> scoredFeatures = search(feature, paramGetMaxFound());
		List<Raster> results = Util.newList(0);
		for (ScoredFeature<MatrixFeature> scoredFeature : scoredFeatures) {
			Record record = scoredFeature.getRecord();
			Raster raster = null;
			if (record != null && record instanceof RasterRecord) raster = ((RasterRecord)record).raster();
			if (raster != null) results.add(raster);
		}
		return results;
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
	IRDefault paramSetNeuronChannel(int neuronChannel) {
		neuronChannel = neuronChannel < 1 ? RasterAbstract.NEURON_CHANNEL_DEFAULT : neuronChannel;
		config.put(RasterAbstract.NEURON_CHANNEL_FIELD, neuronChannel);
		return this;
	}
	
	
	/**
	 * Getting number of blocks.
	 * @return number of blocks.
	 */
	int paramGetMaxFound() {
		if (config.containsKey(MAX_FOUND_FIELD))
			return config.getAsInt(MAX_FOUND_FIELD);
		else
			return MAX_FOUND_DEFAULT;
	}
	
	
	/**
	 * Setting number of blocks.
	 * @param blocks number of blocks.
	 * @return this IR.
	 */
	IRDefault paramSetMaxFound(int blocks) {
		blocks = blocks < 0 ? MAX_FOUND_DEFAULT : blocks;
		config.put(MAX_FOUND_FIELD, blocks);
		return this;
	}


	/**
	 * Getting augmentation flag.
	 * @return augmentation flag.
	 */
	boolean paramIsAugmented() {
		return config.containsKey(net.ea.ann.mane.beans.ProxyNCA.AUGMENTED_FIELD) ? config.getAsBoolean(net.ea.ann.mane.beans.ProxyNCA.AUGMENTED_FIELD) : net.ea.ann.mane.beans.ProxyNCA.AUGMENTED_DEFAULT;
	}
	
	
	/**
	 * Setting augmentation flag.
	 * @param augmented augmentation flag.
	 * @return this IR.
	 */
	IRDefault paramSetAugmented(boolean augmented) {
		config.put(net.ea.ann.mane.beans.ProxyNCA.AUGMENTED_FIELD, augmented);
		return this;
	}


	/**
	 * Getting piece size.
	 * @return piece size.
	 */
	int paramGetPieceSize() {
		int pieceSize = config.getAsInt(net.ea.ann.mane.beans.ProxyNCA.PIECE_SIZE_FIELD);
		return pieceSize < 2 ? net.ea.ann.mane.beans.ProxyNCA.PIECE_SIZE_DEFAULT : pieceSize;
	}
	
	
	/**
	 * Setting piece size.
	 * @param pieceSize piece size.
	 * @return this IR.
	 */
	IRDefault paramSetPieceSize(int pieceSize) {
		pieceSize = pieceSize < 2 ? net.ea.ann.mane.beans.ProxyNCA.PIECE_SIZE_DEFAULT : pieceSize;
		config.put(net.ea.ann.mane.beans.ProxyNCA.PIECE_SIZE_FIELD, pieceSize);
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
	IRDefault paramSetVGGMiddleSize(Size middleSize) {
		int width = middleSize.width < 1 ? VGG.MINSIZE : middleSize.width;
		int height = middleSize.height < 1 ? VGG.MINSIZE : middleSize.height;
		config.put(VGG.MIDDLE_SIZE_FIELD, width + ", " + height);
		return this;
	}


	/**
	 * Getting maximum number of filters per layer (also layer depth).
	 * @return maximum number of filters per layer.
	 */
	int paramGetFiltersNumberMax() {
		if (config.containsKey(VGG.FILTERS_NUMBER_MAX_FIELD))
			return config.getAsInt(VGG.FILTERS_NUMBER_MAX_FIELD);
		else
			return VGG.FILTERS_NUMBER_MAX_DEFAULT;
	}
	
	
	/**
	 * Setting maximum number of filters per layer (also layer depth).
	 * @param maxFiltersNumber maximum number of filters per layer.
	 * @return this IR.
	 */
	IRDefault paramSetFiltersNumberMax(int maxFiltersNumber) {
		maxFiltersNumber = maxFiltersNumber < 0 ? VGG.FILTERS_NUMBER_MAX_DEFAULT : maxFiltersNumber;
		config.put(VGG.FILTERS_NUMBER_MAX_FIELD, maxFiltersNumber);
		return this;
	}
	
	
}
