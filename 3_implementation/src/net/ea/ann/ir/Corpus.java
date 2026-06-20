/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.raster.Raster;
import net.hudup.core.logistic.NextUpdate;

/**
 * This interface represents a corpus of records. 
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> type of data element.
 */
public interface Corpus<T> extends Cloneable, Serializable, Iterable<T> {

	
	/**
	 * This class implements default corpus.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 * @param <T> record type.
	 */
	static class CorpusAbstract<T> implements Corpus<T> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Collection of elements.
		 */
		protected Collection<T> elements = Util.newList(0);
		
		/**
		 * Default constructor.
		 */
		public CorpusAbstract() {
			super();
		}
		
		/**
		 * Constructor with collection of elements.
		 * @param elements collection of elements.
		 */
		public CorpusAbstract(Collection<T> elements) {
			this();
			if (elements != null) this.elements = elements;
		}
		
		/**
		 * Constructor with array of elements.
		 * @param elements array of elements.
		 */
		@SafeVarargs
		public CorpusAbstract(T...elements) {
			this();
			if (elements != null && elements.length > 0) this.elements.addAll(Arrays.asList(elements));
		}

		@Override
		public Iterator<T> iterator() {return elements.iterator();}
		
	}
	
	
	/**
	 * This class implements record corpus.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 * @param <T> record type.
	 */
	static class RecordCorpus<T extends Record> extends CorpusAbstract<T> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public RecordCorpus() {
			super();
		}

		/**
		 * Constructor with collection of records.
		 * @param elements collection of records.
		 */
		public RecordCorpus(Collection<T> records) {
			super(records);
		}

		/**
		 * Constructor with array of records.
		 * @param elements array of records.
		 */
		@SafeVarargs
		public RecordCorpus(T... records) {
			super(records);
		}
		
	}
	
	
	/**
	 * This class implements raster corpus.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class RasterCorpus extends RecordCorpus<RasterRecord> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public RasterCorpus() {
			super();
		}
		
		/**
		 * Constructor with collection of raster records.
		 * @param records collection of raster records.
		 */
		public RasterCorpus(Collection<RasterRecord> records) {
			super(records);
		}
		
		/**
		 * Constructor with array of raster records.
		 * @param records array of raster records.
		 */
		public RasterCorpus(RasterRecord...records) {
			super(records);
		}

		/**
		 * Returns an collection of rasters. This method will be improved so as not to create a new collection of rasters.
		 * @return iterable collection of rasters.
		 */
		@NextUpdate
		public List<Raster> rasters() {
			List<Raster> rasters = Util.newList(0);
			for (RasterRecord record : elements) {
				if (record != null && record.raster() != null) rasters.add(record.raster());
			}
			return rasters;
		}
		
		/**
		 * Creating raster corpus.
		 * @param rasters rasters
		 * @return raster corpus.
		 */
		public static RasterCorpus create(Iterable<Raster> rasters) {
			List<RasterRecord> records = Util.newList(0);
			for (Raster raster: rasters) {
				if (raster != null) records.add(new RasterRecord(raster));
			}
			return new RasterCorpus(records);
		}
		
	}
	
	
	/**
	 * This class implements feature corpus.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 * @param <T> feature type.
	 */
	static class FeatureCorpus<T extends Feature> extends CorpusAbstract<T> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public FeatureCorpus() {
			super();
		}

		/**
		 * Constructor with collection of features.
		 * @param elements collection of features.
		 */
		public FeatureCorpus(Collection<T> features) {
			super(features);
		}

		/**
		 * Constructor with array of features.
		 * @param elements array of features.
		 */
		@SafeVarargs
		public FeatureCorpus(T... features) {
			super(features);
		}
		
	}
	

	/**
	 * This class implements matrix feature corpus.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class MatrixFeatureCorpus extends FeatureCorpus<MatrixFeature> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public MatrixFeatureCorpus() {
			super();
		}
		
		/**
		 * Constructor with collection of matrix features.
		 * @param records collection of matrix features.
		 */
		public MatrixFeatureCorpus(Collection<MatrixFeature> features) {
			super(features);
		}
		
		/**
		 * Constructor with array of matrix features.
		 * @param features array of matrix features.
		 */
		public MatrixFeatureCorpus(MatrixFeature...features) {
			super(features);
		}

	}
	
	
}
