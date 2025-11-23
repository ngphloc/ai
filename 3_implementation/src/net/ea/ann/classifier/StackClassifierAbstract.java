/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.generator.GeneratorWeighted;

public abstract class StackClassifierAbstract extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
//	/**
//	 * Neuron channel of classifier nut.
//	 */
//	static final int FULL_NETWORK_NEURON_CHANNEL_DEFAULT = 1;
	
	
	/**
	 * Field of the number elements of a combination.
	 */
	static final String COMB_NUMBER_FIELD = GeneratorWeighted.COMB_NUMBER_FIELD;
	
	
	/**
	 * Default value for the field of the number elements of a combination.
	 */
	static final int COMB_NUMBER_DEFAULT = GeneratorWeighted.COMB_NUMBER_DEFAULT;

	
	/**
	 * Name of zoom-out field.
	 */
	final static String ZOOMOUT_FIELD = "stac_zoomout";

	
	/**
	 * Default value of zoom-out field.
	 */
	public final static int ZOOMOUT_DEFAULT = NetworkAbstract.ZOOMOUT_DEFAULT;

	
	/**
	 * Name of getting feature field.
	 */
	final static String GET_FEATURE_FIELD = "stac_get_feature";

	
	/**
	 * Default value of getting field.
	 */
	final static boolean GET_FEATURE_DEFAULT = false;

	
	/**
	 * Name of simplest field.
	 */
	final static String SIMPLEST_FIELD = "stac_simplest";

	
	/**
	 * Default value of simplest field.
	 */
	final static boolean SIMPLEST_DEFAULT = false;

	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	protected StackClassifierAbstract(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected StackClassifierAbstract(int neuronChannel) {
		this(neuronChannel, null);
	}


}
