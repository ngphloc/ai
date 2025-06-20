/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.rmi.RemoteException;
import java.util.Arrays;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NetworkStandard extends Network, Evaluator {


	/**
	 * Layer type.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	enum LayerType {
		
		/**
		 * Input layer.
		 */
		input,
		
		/**
		 * Hidden layer.
		 */
		hidden,
		
		/**
		 * Output layer.
		 */
		output,
		
		/**
		 * Memory layer.
		 */
		memory,
		
		/**
		 * Input rib layer.
		 */
		ribin,
		
		/**
		 * Memory layer.
		 */
		ribout,
		
		/**
		 * Unknown layer.
		 */
		unknown,
		
	}
	

	@Override
	NeuronValue[] evaluate(Record inputRecord) throws RemoteException;
	
	
	/**
	 * Learning neural network one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnOne(Iterable<Record> sample) throws RemoteException;
	
	
	/**
	 * Learning neural network.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learn(Iterable<Record> sample) throws RemoteException;


	/**
	 * Create array of hidden neurons.
	 * @param nInput number of input neurons.
	 * @param nOutput number of output neurons.
	 * @return array of hidden neurons.
	 */
	static int[] constructHiddenNeuronNumbers(int nInput, int nOutput) {
		if (nInput <= 0 || nOutput <= 0) return null;
		if (nInput == nOutput) return null;
	
		int min = Math.min(nInput, nOutput);
		int max = Math.max(nInput, nOutput);
		if (min == 1) min = 2;
		if (min == max) return null;
		
		int n = (int) (Math.log(max)/Math.log(min) - 2);
		if (n <= 1) {
			int nHiddenNeuron0 = Math.min((int)Math.pow(min, 2), (min+max)/2); //This trick is a solution technique.
			return new int[] {nHiddenNeuron0}; 
		}
		
		int[] nHiddenNeuron = new int[n];
		for (int i = 0; i < n; i++) nHiddenNeuron[i] = (int) (Math.pow(min, i+2));
		
		if (nInput < nOutput)
			return nHiddenNeuron;
		else {
			int[] array = new int[n];
			for (int i = 0; i < n; i++) array[i] = nHiddenNeuron[array.length-i - 1];
			return array;
		}
	}

	
	/**
	 * Create array of hidden neurons.
	 * @param nInput number of input neurons.
	 * @param nOutput number of output neurons.
	 * @param hiddenLayerMin minimum number of hidden layers.
	 * @return array of hidden neurons.
	 */
	static int[] constructHiddenNeuronNumbers(int nInput, int nOutput, int hiddenLayerMin) {
		int[] nHiddenNeuron = constructHiddenNeuronNumbers(nInput, nOutput);
		if (nHiddenNeuron == null || hiddenLayerMin < 1) return nHiddenNeuron;
		int n = nHiddenNeuron.length;
		if (n >= hiddenLayerMin) return nHiddenNeuron;
		
		nHiddenNeuron = Arrays.copyOf(nHiddenNeuron, n + hiddenLayerMin);
		for (int i = 0; i < hiddenLayerMin; i++) nHiddenNeuron[n+i] = hiddenLayerMin;
		return nHiddenNeuron;
	}
	

}
