/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.conv;

import net.ea.ann.conv.ConvLayer;
import net.ea.ann.conv.ConvLayer2DImpl;
import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.conv.ConvNetwork;
import net.ea.ann.conv.ConvNetworkImpl;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Size;

/**
 * This interface represents a deconvolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public interface DeconvLayer extends ConvLayer {

	
}



/**
 * This class is the default implementation of deconvolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class DeconvLayerImpl extends ConvLayer2DImpl implements DeconvLayer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected DeconvLayerImpl(int neuronChannel, Function activateRef, int width, int height, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, width, height, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 */
	protected DeconvLayerImpl(int neuronChannel, Function activateRef, int width, int height, Filter filter) {
		this(neuronChannel, activateRef, width, height, filter, null);
	}

	
	/**
	 * Creating deconvolutional layer with neuron channel, activation function, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return deconvolutional layer.
	 */
	public static DeconvLayerImpl create(int neuronChannel, Function activateRef, int width, int height, Filter filter, Id idRef) {
		if (width <= 0 || height <= 0) return null;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		
		return new DeconvLayerImpl(neuronChannel, activateRef, width, height, filter, idRef);
	}


	/**
	 * Creating deconvolutional layer with neuron channel, activation function, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @return deconvolutional layer.
	 */
	public static DeconvLayerImpl create(int neuronChannel, Function activateRef, int width, int height, Filter filter) {
		return create(neuronChannel, activateRef, width, height, filter, null);
	}
	
	
}



/**
 * This interface represents a deconvolutional network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
interface DeconvNetwork extends ConvNetwork {

	
}



/**
 * This class is the default implementation of deconvolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class DeconvNetworkImpl extends ConvNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param idRef ID reference.
	 */
	protected DeconvNetworkImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Default constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 */
	protected DeconvNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	@Override
	public ConvLayerSingle newLayer(Size size, Filter filter) {
		return DeconvLayerImpl.create(neuronChannel, activateRef, size.width, size.height, filter);
	}

	
	/**
	 * Creating deconvolutional network with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param idRef ID reference.
	 * @return deconvolutional network.
	 */
	public static DeconvNetworkImpl create(int neuronChannel, Function activateRef, Id idRef) {
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new DeconvNetworkImpl(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Creating deconvolutional network with neuron channel and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @return deconvolutional network.
	 */
	public static DeconvNetworkImpl create(int neuronChannel, Function activateRef) {
		return create(neuronChannel, activateRef, null);
	}


}
