/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.ir.Corpus.RecordCorpus;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.MLAbstract.RasterMatrixMLGeneral;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.ir.ml.ProxyNCA;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of information Retrieval (IR) system based on Proxy-NCA (Proxy-Neighborhood Component Analysis) algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IRProxyNCA extends IRDefault {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
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
	public IRProxyNCA() {
		super();
		if (this.ml instanceof ProxyNCA) {
			((ProxyNCA)this.ml).getConfig().putAll(this.config);
			this.config.putAll(((ProxyNCA)this.ml).getConfig());
		}
		
		//Removing following lines after debugging.
		paramSetAugmented(true);
		paramSetVGGMiddleSize(new Size(32, 32));
		paramSetFiltersNumberMax(1);
		System.out.println("IRProxyNCA: Removing following lines after debugging.");
	}

	
	@Override
	protected RasterMatrixMLGeneral createML() {
		ProxyNCA ml = new ProxyNCA();
		if (this.config != null) {
			ml.getConfig().putAll(this.config);
			this.config.putAll(ml.getConfig());
		}
		return ml;
	}


	@Override
	protected ProxyNCA ml() {return (ProxyNCA)super.ml();}


	@Override
	public boolean build(RecordCorpus<RasterRecord> corpus, boolean refresh) {
		ml().getConfig().putAll(this.config);
		return super.build(corpus, refresh);
	}


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
	
	
}
