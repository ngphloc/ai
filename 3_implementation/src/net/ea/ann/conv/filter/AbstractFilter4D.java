/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle3D;
import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is an abstract implementation of filter in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractFilter4D extends AbstractFilter3D implements Filter4D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public AbstractFilter4D() {
		super();
	}

	
	@Override
	public NeuronValue apply(int x, int y, int z, ConvLayerSingle3D layer) {
		if (!(layer instanceof ConvLayerSingle4D)) return null;
		return apply(x, y, z, 0, (ConvLayerSingle4D)layer);
	}

	
}
