/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.rnn;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.Network;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.vector.NeuronValueVector;

/**
 * This interface represents recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RecurrentNetwork extends Network {


	/**
	 * Evaluating network.
	 * @param input specified input.
	 * @throws RemoteException if any error raises.
	 */
	void evaluate(NeuronValue...input) throws RemoteException;
	
	
	/**
	 * Evaluating network.
	 * @param inputs list of inputs.
	 * @throws RemoteException if any error raises.
	 */
	void evaluate(List<NeuronValue[]> inputs) throws RemoteException;

	
	/**
	 * Learning recurrent neural network one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnOne(Iterable<List<Record>> sample) throws RemoteException;
	
	
	/**
	 * Learning recurrent neural network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<List<Record>> sample) throws RemoteException;


	/**
	 * Verifying value.
	 * @param neuron specified neuron.
	 * @param value neuron value.
	 * @return verified value (vector or non-vector).
	 */
	static NeuronValue verify(NeuronValue value, NeuronStandard neuron) {
		if (neuron == null || value == null)
			return value;
		else {
			NeuronValue bias = neuron.getBias(); //Bias is parameter but it conforms value structure.
			return bias == null ? value : NeuronValueVector.addZero(value, bias.zero()); //Trick to convert into vector.
		}
	}


	/**
	 * Verifying non-vector value.
	 * @param value value.
	 * @return non-vector value.
	 */
	static NeuronValue verifyNonvector(NeuronValue value) {
		if ((value == null) || !(value instanceof NeuronValueVector)) return value;
		return ((NeuronValueVector)value).get(0);
	}


}
