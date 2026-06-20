/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Sim;
import net.ea.ann.ir.Record.RasterRecord;
import net.ea.ann.raster.Raster;

/**
 * This interface represents feature.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Feature extends Cloneable, Serializable {

	
	/**
	 * Calculating similarity of this feature and other feature.
	 * @param other other feature.
	 * @return similarity of this feature and other feature.
	 */
	NeuronValue sim(Feature other);
	
	
	/**
	 * Calculating distance of this feature and other feature.
	 * @param other other feature.
	 * @return distance of this feature and other feature.
	 */
	NeuronValue distance(Feature other);

	
	/**
	 * Getting attached record.
	 * @return attached record.
	 */
	Record getRecord();
	
	
	/**
	 * This class implements matrix feature.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class MatrixFeature implements Feature {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Matrix feature.
		 */
		protected Matrix feature = null;

		/**
		 * Attached record.
		 */
		protected Record attachedRecord = null;
		
		/**
		 * Constructor with matrix and record.
		 * @param feature matrix as feature.
		 * @param attachedRecord attached record.
		 * @param attachedRecord attached record.
		 */
		public MatrixFeature(Matrix feature, Record attachedRecord) {
			this.feature = feature;
			this.attachedRecord = attachedRecord;
		}

		/**
		 * Constructor with matrix and raster.
		 * @param feature matrix as feature.
		 * @param raster raster.
		 */
		public MatrixFeature(Matrix feature, Raster raster) {
			this(feature, new RasterRecord(raster));
		}

		/**
		 * Constructor with feature.
		 * @param feature feature.
		 */
		public MatrixFeature(MatrixFeature feature) {
			this(feature.feature, feature.attachedRecord);
		}

		/**
		 * Constructor with matrix as feature.
		 * @param feature matrix as feature.
		 */
		public MatrixFeature(Matrix feature) {this(feature, (Record)null);}
		
		@Override
		public NeuronValue sim(Feature other) {
			Sim.ProductMatrixSim sim = new Sim.ProductMatrixSim();
			Matrix otherFeature = ((MatrixFeature)other).feature;
			NeuronValue module = sim.sim(this.feature, this.feature).multiply(sim.sim(otherFeature, otherFeature));
			return sim.sim(this.feature, otherFeature).divide(module.sqrt()); //Cosine similarity.
		}
		
		@Override
		public NeuronValue distance(Feature other) {
			Sim.ProductMatrixSim sim = new Sim.ProductMatrixSim();
			Matrix otherFeature = ((MatrixFeature)other).feature;
			Matrix d = this.feature.subtract(otherFeature);
			return sim.sim(d, d).sqrt(); //Euclidean distance.
		}

		@Override
		public Record getRecord() {return this.attachedRecord;}

		/**
		 * Setting attached record.
		 * @param attachedRecord attached record.
		 * @return this matrix feature.
		 */
		public MatrixFeature setRecord(Record attachedRecord) {
			this.attachedRecord = attachedRecord;
			return this;
		}
		
		/**
		 * Getting feature.
		 * @return feature.
		 */
		public Matrix feature() {return this.feature;}
		
		/**
		 * Getting raster.
		 * @return raster.
		 */
		public Raster raster() {
			return attachedRecord != null && attachedRecord instanceof RasterRecord ? ((RasterRecord)attachedRecord).raster() : null;
		}
		
	}
	
	
}
