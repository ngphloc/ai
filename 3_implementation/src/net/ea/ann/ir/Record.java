/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.raster.Raster;

/**
 * This interface represents record.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Record extends Cloneable, Serializable {

	
	/**
	 * This class represents raster record.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class RasterRecord implements Record {


		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Internal raster.
		 */
		protected Raster raster = null;
		
		/**
		 * Constructor with raster.
		 * @param raster raster.
		 */
		public RasterRecord(Raster raster) {this.raster = raster;}
		
		/**
		 * Getting raster.
		 * @return raster.
		 */
		public Raster raster() {return this.raster;}
		
		/**
		 * Extracting rasters from records.
		 * @param records records.
		 * @return list of rasters.
		 */
		public static List<Raster> toRasters(List<RasterRecord> records) {
			List<Raster> rasters = Util.newList(0);
			for (RasterRecord record : records) {
				if (record != null && record.raster != null) rasters.add(record.raster);
			}
			return rasters;
		}
		
	}
	
	
}
