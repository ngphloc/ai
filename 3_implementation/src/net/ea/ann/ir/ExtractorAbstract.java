/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.Record.RasterRecord;

/**
 * This class is abstract class of feature extraction component.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <U> record type.
 * @param <V> feature type.
 */
public abstract class ExtractorAbstract<U extends Record, V extends Feature> implements Extractor<U, V> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public ExtractorAbstract() {
		
	}

	
	/**
	 * This class is abstract class of matrix feature extraction component with respect to raster record.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static abstract class RasterMatrixExtractorAbstract extends ExtractorAbstract<RasterRecord, MatrixFeature> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public RasterMatrixExtractorAbstract() {
			super();
		}

	}


}
