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
import net.ea.ann.mane.beans.VGGClassifier;
import net.ea.ann.raster.Raster;

/**
 * This class is an extensive implementation of classifier within VGG model developed by Simonyan and Zisserman.
 * @author Simonyan and Zisserman, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class VGGExt extends NetworkAbstract implements Classifier {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal classifier.
	 */
	protected VGGClassifier classifier = null;
	
	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public VGGExt(int neuronChannel) {
		super();
		
		//Removing following lines after debugging.
		this.classifier = new VGGClassifier(neuronChannel);
//		this.classifier.paramSetDropoutMode(false);
//		this.classifier.paramSetResidualMode(false);
//		this.classifier.paramSetGAP(false);
//		this.classifier.paramSetLayerNorm(false);
//		this.classifier.paramSetFiltersNumberMax(1);
//		this.classifier.paramSetFiltersNumberInit(16);
//		this.classifier.paramSetVGGMiddleSize(new Size(32, 32));
		System.out.println("net.ea.ann.classifier.VGGExt: Removing following lines after debugging.");
		
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

	
//	@Override
//	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
//		classifier.setConfig(this.config);
//		
//		classifier.initializeByCoreClassesWithImplicitMiddleSize(sample);
//		for (int i = 0; i < classifier.size(); i++) {
//			classifier.get(i).setWeightActivateRef(IdentityDefault.identity());
//			classifier.get(i).setFilterActivateRef(IdentityDefault.identity());
//		}
//		
//		return new NeuronValue[] {};
//	}

	
	@Override
	public List<Raster> classify(Iterable<Raster> sample) throws RemoteException {
		return classifier.classifyRaster(sample);
	}

	
}
