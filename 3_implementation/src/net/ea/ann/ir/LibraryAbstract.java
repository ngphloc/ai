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
import net.ea.ann.ir.Corpus.FeatureCorpus;
import net.ea.ann.ir.Corpus.RecordCorpus;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.Record.RasterRecord;

/**
 * This class implements partially library.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <U> record type.
 * @param <V> feature type.
 */
public abstract class LibraryAbstract<U extends Record, V extends Feature> implements Library<V> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Temporal pool of features.
	 */
	protected List<V> pool = Util.newList(0);
	
	
	/**
	 * Default constructor.
	 */
	public LibraryAbstract() {
		
	}

	
	/**
	 * Getting Approximate Nearest Neighbor (AppNN) model.
	 * @return Approximate Nearest Neighbor (AppNN) model.
	 */
	protected abstract AppNN<V> getAppNN();


	@Override
	public boolean build(FeatureCorpus<V> corpus, boolean refresh) {
		if (refresh) pool.clear();
		
		AppNN<V> app = getAppNN();
		if (app != null) return app.build(corpus, refresh);
		
		for (V feature : corpus) {
			if (feature != null) pool.add(feature);
		}
		return true;
	}


	/**
	 * Building library from record corpus with extractor.
	 * @param extractor extractor.
	 * @param corpus record corpus.
	 * @param refresh  refreshment flag.
	 * @return true if building is successful.
	 */
	public boolean build(Extractor<U, V> extractor, RecordCorpus<U> corpus, boolean refresh) {
		if (refresh) pool.clear();

		AppNN<V> app = getAppNN();
		if (app != null) {
			List<V> features = Util.newList(0);
			for (U record: corpus) {
				V feature = extractor.featureOf(record);
				if (feature != null) features.add(feature);
			}
			return app.build(new FeatureCorpus<>(features), refresh);
		}
		
		for (U record: corpus) {
			V feature = extractor.featureOf(record);
			if (feature != null) pool.add(feature);
		}
		return true;
	}
	
	
	@Override
	public List<V> obtainCandidates(V query) {
		AppNN<V> app = getAppNN();
		return app != null ? app.search(query) : pool;
	}
	
	
	/**
	 * This class represents a library with respect to raster and matrix feature.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public abstract static class RasterMatrixLibraryAbstract extends LibraryAbstract<RasterRecord, MatrixFeature> {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public RasterMatrixLibraryAbstract() {super();}

	}

		
}
