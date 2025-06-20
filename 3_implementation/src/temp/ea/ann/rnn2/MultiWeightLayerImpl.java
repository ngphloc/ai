/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn2;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.LayerStandardImpl;
import net.ea.ann.core.function.Function;

/**
 * This class is default implementation of multiple weight layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class MultiWeightLayerImpl extends LayerStandardImpl implements MultiWeightLayer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Previous extra layer.
	 */
	protected LayerStandard[] prevExtraLayers = null;
	
	
	/**
	 * Implicit previous extra layer.
	 */
	protected LayerStandard[] prevExtraLayerImplicits = null;

	
	/**
	 * Next extra layer.
	 */
	protected LayerStandard[] nextExtraLayers = null;
	
	
	/**
	 * Input rib extra layer.
	 */
	protected LayerStandard[] ribinExtraLayers = null;

	
	/**
	 * Output rib extra layer.
	 */
	protected LayerStandard[] riboutExtraLayers = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public MultiWeightLayerImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MultiWeightLayerImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	public MultiWeightLayerImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
}
