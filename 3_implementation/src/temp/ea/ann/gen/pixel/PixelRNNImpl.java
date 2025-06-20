/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.gen.pixel;

import net.ea.ann.core.Id;
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of generative pixel recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class PixelRNNImpl extends PixelRNNAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected PixelRNNImpl(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected PixelRNNImpl(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}

	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	protected PixelRNNImpl(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected PixelRNNImpl(int neuronChannel) {
		this(neuronChannel, neuronChannel, new Size(1, 1, 1, 1), null);
	}

	
	/**
	 * Creating with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return generative pixel recurrent neural network.
	 */
	public static PixelRNNImpl create(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		size.width = size.width < 1 ? 1 : size.width;
		size.height = size.height < 1 ? 1 : size.height;
		size.depth = size.depth < 1 ? 1 : size.depth;
		size.time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		rasterChannel = rasterChannel < neuronChannel ? neuronChannel : rasterChannel;
		return new PixelRNNImpl(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Creating with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return generative pixel recurrent neural network.
	 */
	public static PixelRNNImpl create(int neuronChannel, Size size, Id idRef) {
		return create(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Creating with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size raster size.
	 * @return generative pixel recurrent neural network.
	 */
	public static PixelRNNImpl create(int neuronChannel, Size size) {
		return create(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Creating with neuron channel and raster channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return generative pixel recurrent neural network.
	 */
	public static PixelRNNImpl create(int neuronChannel, int rasterChannel) {
		return create(neuronChannel, rasterChannel, new Size(1, 1, 1, 1), null);
	}

	
	/**
	 * Creating with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return generative pixel recurrent neural network.
	 */
	public static PixelRNNImpl create(int neuronChannel) {
		return create(neuronChannel, neuronChannel, new Size(1, 1, 1, 1), null);
	}
	
	
}
