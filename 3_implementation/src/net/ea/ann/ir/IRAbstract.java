/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.util.List;

import net.ea.ann.ir.AppNNAbstract.MatrixFeatureAppNNAbstract;
import net.ea.ann.ir.Corpus.RecordCorpus;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.LibraryAbstract.RasterMatrixLibraryAbstract;
import net.ea.ann.ir.MLAbstract.MLGeneral;
import net.ea.ann.ir.MLAbstract.RasterMatrixMLGeneral;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.ir.SearcherAbstract.MatrixFeatureSearcherAbstract;

/**
 * This class implements basically information retrieval system.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <U> record type.
 * @param <V> feature type.
 */
public abstract class IRAbstract<U extends Record, V extends Feature> implements IR<U, V>, Searcher<V> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public IRAbstract() {

	}

	
	/**
	 * Getting extractor.
	 * @return extractor.
	 */
	protected abstract Extractor<U, V> extractor();
	
	
	/**
	 * Getting metric learning component.
	 * @return metric learning component.
	 */
	protected abstract ML<U> ml();


	/**
	 * Getting library.
	 * @return library.
	 */
	protected abstract Library<V> library();
	
	
	/**
	 * Getting searching component.
	 * @return searching component.
	 */
	protected abstract Searcher<V> searcher();

	
	@Override
	public List<ScoredFeature<V>> search(V query, int maxFound) {return searcher().search(query, maxFound);}


	/**
	 * This class implements general IR system.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public abstract static class IRGeneral<U extends Record, V extends Feature> extends IRAbstract<U, V> {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * General metric learning component.
		 */
		protected MLGeneral<U, V> ml = null;
		
		/**
		 * Library.
		 */
		protected LibraryAbstract<U, V> library = null;
		
		/**
		 * Searching component.
		 */
		protected SearcherAbstract<V> searcher = null;
		
		/**
		 * Default constructor.
		 */
		public IRGeneral() {
			super();
			this.ml = createML();
			this.library = createLibrary();
			this.searcher = createSearcher();
		}

		/**
		 * Creating default metric learning component.
		 * @return default metric learning component.
		 */
		protected abstract MLGeneral<U, V> createML();
		
		/**
		 * Creating library.
		 * @return library.
		 */
		protected abstract LibraryAbstract<U, V> createLibrary();
		
		/**
		 * Creating searcher.
		 * @return searcher.
		 */
		protected abstract SearcherAbstract<V> createSearcher();
		
		@Override
		protected Extractor<U, V> extractor() {return ml;}

		@Override
		protected MLGeneral<U, V> ml() {return ml;}

		@Override
		protected LibraryAbstract<U, V> library() {return library;}

		@Override
		protected SearcherAbstract<V> searcher() {return searcher;}

		@Override
		public boolean build(RecordCorpus<U> corpus, boolean refresh) {
			boolean buildML = ml().build(corpus, refresh);
			boolean buildLibrary = library().build(extractor(), corpus, refresh);
			return buildML && buildLibrary;
		}

	}

	
	/**
	 * This class implements default IR system.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public abstract static class IRSimple extends IRGeneral<RasterRecord, MatrixFeature> {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Default constructor.
		 */
		public IRSimple() {
			super();
		}

		@Override
		protected abstract RasterMatrixMLGeneral createML();
		
		@Override
		protected RasterMatrixLibraryAbstract createLibrary() {
			return new RasterMatrixLibraryAbstract() {
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;
				
				@Override
				protected MatrixFeatureAppNNAbstract getAppNN() {return null;}
			};
		}
		
		@Override
		protected MatrixFeatureSearcherAbstract createSearcher() {
			return new MatrixFeatureSearcherAbstract() {
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;
				
				@Override
				protected Library<MatrixFeature> getLibrary() {return getThisIR().library;}
			};
		}
		
		/**
		 * Getting this IR.
		 * @return this IR.
		 */
		private IRSimple getThisIR() {return this;}
		
	}

	
	
}
