/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.beans.wi.SwarmClassifier;
import net.ea.ann.raster.Raster;

/**
 * This class represents swarm classifier.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Swarm extends NetworkAbstract implements Classifier {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal classifier.
	 */
	protected SwarmClassifier classifier = null;
	
	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 */
	public Swarm(int neuronChannel, int rasterChannel) {
		super();
		
		this.classifier = new SwarmClassifier(neuronChannel);
		this.classifier.paramSetRasterChannel(rasterChannel);
		
		try {
			this.config.putAll(this.classifier.getConfig());
		} catch (Throwable e) {Util.trace(e);}
	}

	
	@Override
	public int getNeuronChannel() throws RemoteException {
		return classifier.getNeuronChannel();
	}

	
	@Override
	public NeuronValue[] learnRasterOneByOne(Iterable<Raster> sample) throws RemoteException {
		return learnRaster(sample);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		this.classifier.setConfig(this.config);
		
		Error [] errors = this.classifier.learnRasterByCoreClassesWithImplicitMiddleSize(sample);
		
		NeuronValue[] errorArray = null;
		for (Error error : errors) {
			NeuronValue[] values = MatrixUtil.extractValues(error.error());
			errorArray = errorArray == null ? values : NeuronValue.concatArray(errorArray, values);
		}
		return errorArray;
	}

	
	@Override
	public List<Raster> classify(Iterable<Raster> sample) throws RemoteException {
		return classifier.classifyRaster(sample);
	}

	
}
