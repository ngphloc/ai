/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represent a pair of filter and bias. 
 * @author Loc Nguyen
 */
public class BiasFilter implements TextParsable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal filter.
	 */
	public Filter filter = null;
	
	
	/**
	 * Internal bias.
	 */
	public NeuronValue bias = null;
	
	
	/**
	 * Constructor with filter and bias.
	 * @param filter specified filter.
	 * @param bias specified bias.
	 */
	public BiasFilter(Filter filter, NeuronValue bias) {
		this.filter = filter;
		this.bias = bias;
	}

	
	/**
	 * Default constructor.
	 */
	public BiasFilter() {
		super();
	}

	
	@Override
	public String toText() {
		if (filter == null && bias == null) return "";
		StringBuffer buffer = new StringBuffer();
		
		if (filter != null) {
			buffer.append("filter = {");
			if (filter instanceof TextParsable)
				buffer.append(((TextParsable)filter).toText());
			else
				buffer.append(filter);
			buffer.append("}");
		}
		
		if (bias != null) {
			buffer.append("bias = (");
			if (bias instanceof TextParsable)
				buffer.append(((TextParsable)bias).toText());
			else
				buffer.append(bias);
			buffer.append(")");
		}
		
		return buffer.toString();
	}
	
	
}


