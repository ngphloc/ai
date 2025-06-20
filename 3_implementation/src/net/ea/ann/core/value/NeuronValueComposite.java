/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

/**
 * This interface represents composite neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NeuronValueComposite extends NeuronValue {


	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	int getNeuronChannel();
	
	
	/**
	 * Resize value by channel.
	 * @param newChannel new channel.
	 * @return new value with new channel.
	 */
	NeuronValue resizeByChannel(int newChannel);
	
	
	/**
	 * Flattening this value into array of smaller channel.
	 * @param smallerChannel smaller channel.
	 * @return array of smaller channel values.
	 */
	NeuronValue[] flattenByChannel(int smallerChannel);
	
	
	/**
	 * Flattening array
	 * @param array specified array.
	 * @param smallerChannel smaller channel.
	 * @return flattened array.
	 */
	NeuronValue[] flattenByChannel(NeuronValue[] array, int smallerChannel);

	
	/**
	 * Aggregating value array into singular value.
	 * @param array value array.
	 * @return singular value.
	 */
	NeuronValue aggregateByChannel(NeuronValue[] array);
	
	
	/**
	 * Aggregating array according to larger channel.
	 * @param array specified array.
	 * @param largerChannel larger channel.
	 * @return aggregated array.
	 */
	NeuronValue[] aggregateByChannel(NeuronValue[] array, int largerChannel);


}
