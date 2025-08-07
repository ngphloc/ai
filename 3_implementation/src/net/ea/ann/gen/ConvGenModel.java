/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.rmi.RemoteException;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Raster;

/**
 * This interface represents convolutional generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvGenModel extends GenModel {

	
	/**
	 * Setting parameters of generative model.
	 * @param setting parameters of generative model.
	 * @throws RemoteException if any error raises.
	 */
	void setSetting(ConvGenSetting setting) throws RemoteException;	

	
	/**
	 * Getting parameters of generative model.
	 * @return parameters of generative model.
	 * @throws RemoteException if any error raises.
	 */
	ConvGenSetting getSetting() throws RemoteException;

	
	/**
	 * Getting raster channel.
	 * @return raster channel.
	 * @throws RemoteException if any error raises.
	 */
	int getRasterChannel() throws RemoteException;


	/**
	 * Initialize with Z dimension, convolutional filters and deconvolutional filters.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param deconvFilterArrays arrays of deconvolutional filters. Deconvolutional filters in the same array have the same size.
	 * @return true if initialization is successful.
	 * @throws RemoteException if any error raises.
	 */
	boolean initialize(int zDim,
		Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) throws RemoteException;

	
	/**
	 * Initialize with Z dimension, convolutional filters and deconvolutional filters.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 * @throws RemoteException if any error raises.
	 */
	boolean initialize(int zDim,
		Filter[] convFilters, Filter[] deconvFilters) throws RemoteException;
	
	
	/**
	 * Learning the convolutional generative model by raster, one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException;

	
	/**
	 * Learning the convolutional generative model by raster sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException;

	
	/**
	 * Generate raster.
	 * @return generated raster.
	 * @throws RemoteException if any error raises.
	 */
	G generateRaster() throws RemoteException;

	
	/**
	 * Generate raster by Z data.
	 * @param dataZ Z data.
	 * @return generated raster.
	 * @throws RemoteException if any error raises.
	 */
	G generateRaster(NeuronValue...dataZ) throws RemoteException;

	
	/**
	 * Generate best raster.
	 * @return best generated raster.
	 * @throws RemoteException if any error raises.
	 */
	G generateRasterBest() throws RemoteException;

	
	/**
	 * Recovering raster.
	 * @param raster original raster.
	 * @param region specified region. If it is null, entire raster will be recovered.
	 * @param random flag to indicate whether or not to random generation.
	 * @param calcError flag to indicate whether or not to calculate error.
	 * @return generated structure.
	 * @throws RemoteException if any error raises.
	 */
	G recoverRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException;


	/**
	 * Reproducing raster, which is similar to method {@link #recoverRaster(Raster, Cube, boolean, boolean)} except that
	 * reproducing method firstly learns from the raster itself that will be reproduced.
	 * @param raster original raster.
	 * @param region specified region. If it is null, entire raster will be reproduced.
	 * @param random flag to indicate whether or not to random generation.
	 * @param calcError flag to indicate whether or not to calculate error.
	 * @return generated structure.
	 * @throws RemoteException if any error raises.
	 */
	G reproduceRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException;


}
