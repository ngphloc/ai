/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.io.Serializable;
import java.rmi.RemoteException;

import net.ea.ann.core.Network;
import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Raster;

/**
 * This interface represents generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface GenModel extends Network {

	
	/**
	 * Default z-dimension where z is random data to generate data X.
	 */
	static final int ZDIM_DEFAULT = 10;
	
	
	/**
	 * This class represents generative data.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class G implements Serializable, Cloneable {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Original data
		 */
		public NeuronValue[] x = null;
		
		/**
		 * Encoded/randomized data
		 */
		public NeuronValue[] z = null;
		
		/**
		 * Decoded/generated data
		 */
		public NeuronValue[] xgen = null;
		
		/**
		 * Undefined decoded/generated data which is often decoded/generated raster.
		 */
		public Object xgenUndefined = null;
		
		/**
		 * Error in generating model.
		 */
		public double error = 0;
		
		/**
		 * Internal tag which is often original raster / recovering raster.
		 */
		public Object tag = null;
		
		/**
		 * Default constructor.
		 */
		public G() {
			
		}
		
		/**
		 * Constructor with encoded/randomized data and decoded/generated data.
		 * @param z encoded/randomized data.
		 * @param xgen decoded/generated data.
		 */
		public G(NeuronValue[] z, NeuronValue[] xgen) {
			this.z = z;
			this.xgen = xgen;
		}
		
		/**
		 * Getting raster of generated X data.
		 * @return raster of generated X data.
		 */
		public Raster getXGenRaster() {
			return xgenUndefined != null && xgenUndefined instanceof Raster ? (Raster)xgenUndefined : null;
		}
		
		/**
		 * Getting raster of generated X data.
		 * @param xgenRaster raster of generated X data.
		 */
		public void setXGenRaster(Raster xgenRaster) {
			this.xgenUndefined = xgenRaster;
		}
		
		/**
		 * Getting raster of original X data.
		 * @return raster of original X data.
		 */
		public Raster getXOriginalRaster() {
			return tag != null && tag instanceof Raster ? (Raster)tag : null;
		}
		
		/**
		 * Getting raster of original X data.
		 * @param xoriginalRaster raster of original X data.
		 */
		public void setXOriginalRaster(Raster xoriginalRaster) {
			this.tag = xoriginalRaster;
		}

	}
	
	
	/**
	 * Learning generative model one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnOneByOne(Iterable<Record> sample) throws RemoteException;


	/**
	 * Learning generative model.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;

	
	/**
	 * Generate values (X values).
	 * @return generated structure.
	 * @throws RemoteException if any error raises.
	 */
	G generate() throws RemoteException;
	
	
	/**
	 * Generate the best X values.
	 * @return generated structure.
	 * @throws RemoteException if any error raises.
	 */
	G generateBest() throws RemoteException;

	
	/**
	 * Recovering values (X values) from original data X.
	 * @param dataX original data X.
	 * @param region specified region.
	 * @param random flag to indicate whether or not to random generation.
	 * @param calcError flag to indicate whether or not to calculate error.
	 * @return generated structure.
	 * @throws RemoteException if any error raises.
	 */
	G recover(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException;


	/**
	 * Reproducing values (X values) from original data X.
	 * @param dataX original data X.
	 * @param region specified region.
	 * @param random flag to indicate whether or not to random generation.
	 * @param calcError flag to indicate whether or not to calculate error.
	 * @return generated structure.
	 * @throws RemoteException if any error raises.
	 */
	G reproduce(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException;

	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 * @throws RemoteException if any error raises.
	 */
	int getNeuronChannel() throws RemoteException;


	/**
	 * Resetting network.
	 * @throws RemoteException if any error raises.
	 */
	void reset() throws RemoteException;
	
	
}
