/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.conv;

import java.rmi.RemoteException;

import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.conv.ConvNetworkImpl;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.Record;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is the extensive implementation of convolutional network which support layer stack.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class ConvNetworkImpl2 extends ConvNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	public ConvNetworkImpl2(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Default constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 */
	public ConvNetworkImpl2(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	@Override
	public ConvLayerSingle newLayer(Size size, Filter filter) {
		return super.newLayer(size, filter);
	}


	@Override
	protected ConvLayerSingle addConvLayers(Filter[] filters, Size size, ConvLayerSingle prevLayer) {
		return super.addConvLayers(filters, size, prevLayer);
	}


	@Override
	public synchronized NeuronValue[] evaluateRaster(Raster inputRaster) throws RemoteException {
		return super.evaluateRaster(inputRaster);
	}


	@Override
	public synchronized NeuronValue[] evaluate(NeuronValue[] input) {
		return super.evaluate(input);
	}


	@Override
	public NeuronValue[] learnOne(Iterable<Record> sample) throws RemoteException {
		return super.learnOne(sample);
	}


	@Override
	protected ConvLayerSingle unifyOutputContent() {
		return super.unifyOutputContent();
	}


	/**
	 * Creating convolutional network with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return convolutional network.
	 */
	public static ConvNetworkImpl2 create(int neuronChannel, Function activateRef, Id idRef) {
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ConvNetworkImpl2(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Creating convolutional network with neuron channel and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @return convolutional network.
	 */
	public static ConvNetworkImpl2 create(int neuronChannel, Function activateRef) {
		return create(neuronChannel, activateRef, null);
	}


}
