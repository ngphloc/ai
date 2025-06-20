/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn2;

import java.util.List;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandardImpl;
import net.ea.ann.core.WeightedNeuron;

/**
 * This interface is the default implementation of multiple weight neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class MultiWeightNeuronImpl extends NeuronStandardImpl implements MultiWeightNeuron {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Next extra neurons.
	 */
	protected List<WeightedNeuron>[] nextExtraNeurons = null;
	
	
	/**
	 * Output extra rib neurons.
	 */
	protected List<WeightedNeuron>[] riboutExtraNeurons = null;

	
	/**
	 * Default constructor.
	 * @param layer this layer.
	 */
	public MultiWeightNeuronImpl(LayerStandard layer) {
		super(layer);
	}

	
}
