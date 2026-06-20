/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import net.ea.ann.ir.Corpus.RasterCorpus;
import net.ea.ann.ir.Corpus.RecordCorpus;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.Record.RasterRecord;

/**
 * This class is abstract class for (deep) metric learning component.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> record type.
 */
public abstract class MLAbstract<T extends Record> implements ML<T> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public MLAbstract() {

	}

	
	/**
	 * This class is general class for metric learning component.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static abstract class MLGeneral<U extends Record, V extends Feature> extends MLAbstract<U> implements Extractor<U, V> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public MLGeneral() {
			super();
		}

	}
	
	
	/**
	 * This class is abstract class for raster metric learning component with respect to matrix feature.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static abstract class RasterMatrixMLGeneral extends MLGeneral<RasterRecord, MatrixFeature> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public RasterMatrixMLGeneral() {
			super();
		}

		/**
		 * Building metric learning component from raster corpus.
		 * @param corpus raster corpus.
		 * @param refresh refreshment flag.
		 * @return true if building is successful.
		 */
		public abstract boolean build(RasterCorpus corpus, boolean refresh);

		@Override
		public boolean build(RecordCorpus<RasterRecord> corpus, boolean refresh) {return  build((RasterCorpus)corpus, refresh);}
		
	}


}
