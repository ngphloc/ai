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
	

	//It is not necessary to re-declare this method because it was declared in remote interface Evaluator.
	@Override
	NeuronValue[] evaluate(Record inputRecord) throws RemoteException;
	
	
	/**
	 * Learning neural network one-by-one record over sample.
	 * @param sample sample for learning.
	 * @return learned error.
	 * @throws RemoteException if any error raises.
	 */
	NeuronValue[] learnOneByOne(Iterable<Record> sample) throws RemoteException;
	
	
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
	@SuppressWarnings("unused")
	@Deprecated
	private static int[] constructHiddenNeuronNumbers0(int nInput, int nOutput) {
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
	 * @param base base.
	 * @param minimum number of hidden layers.
	 * @return array of hidden neurons.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private static int[] constructHiddenNeuronNumbers0(int nInput, int nOutput, int base, int hiddenLayerMin) {
		if (nInput <= 0 || nOutput <= 0 || base < 2) return null;
		if (nInput == nOutput) return null;
	
		//Determining minimum and maximum.
		int min = Math.min(nInput, nOutput);
		int max = Math.max(nInput, nOutput);
		
		//Calculating array of hidden size.
		int[] nHiddenNeuron = null;
		int n = (int) (Math.log(max/min) / Math.log(base) - 1); //min*base^x = max
		if (n < 1)
			nHiddenNeuron = new int[] {(min+max)/2};
		else {
			nHiddenNeuron = new int[n];
			for (int i = 0; i < n; i++) nHiddenNeuron[i] = (int) (min*Math.pow(base, i+1));
		}

		//Re-ordering the hidden sizes.
		if (nInput > nOutput) {
			int[] array = new int[nHiddenNeuron.length];
			for (int i = 0; i < nHiddenNeuron.length; i++) array[i] = nHiddenNeuron[(array.length-i) - 1];
			nHiddenNeuron = array;
		}
		
		//Checking hidden layer minimum.
		if (hiddenLayerMin < 1 || nHiddenNeuron.length >= hiddenLayerMin) return nHiddenNeuron;
		int length = nHiddenNeuron.length;
		int pad = hiddenLayerMin - length;
		nHiddenNeuron = Arrays.copyOf(nHiddenNeuron, length + pad);
		Arrays.fill(nHiddenNeuron, length, length + pad, nHiddenNeuron[length-1]);
		return nHiddenNeuron;
	}
	
	
	/**
	 * Create array of hidden neurons.
	 * @param nInput number of input neurons.
	 * @param nOutput number of output neurons.
	 * @param base base.
	 * @param minimum number of hidden layers.
	 * @return array of hidden neurons.
	 */
	static int[] constructHiddenNeuronNumbers(int nInput, int nOutput, int base, int hiddenLayerMin) {
		if (nInput <= 0 || nOutput <= 0 || base < 2) return null;
		if (nInput == nOutput) return null;
	
		//Determining minimum and maximum.
		int min = Math.min(nInput, nOutput);
		int max = Math.max(nInput, nOutput);
		
		//Calculating array of hidden size.
		int[] nHiddenNeuron = null;
		int n = (int) (Math.log(max/min) / Math.log(base)); //min*base^x = max
		if (n <= 1)
			nHiddenNeuron = new int[] {nOutput};
		else {
			nHiddenNeuron = new int[n];
			for (int i = 0; i < n; i++) nHiddenNeuron[i] = (int) (min*Math.pow(base, i+1));
		}

		//Re-ordering the hidden sizes.
		if (nInput > nOutput) {
			int[] array = new int[nHiddenNeuron.length];
			for (int i = 0; i < nHiddenNeuron.length; i++) array[i] = nHiddenNeuron[(array.length-i) - 1];
			nHiddenNeuron = array;
		}
		
		//Checking hidden layer minimum.
		if (hiddenLayerMin < 1 || nHiddenNeuron.length >= hiddenLayerMin) return nHiddenNeuron;
		int length = nHiddenNeuron.length;
		int pad = hiddenLayerMin - length;
		nHiddenNeuron = Arrays.copyOf(nHiddenNeuron, length + pad);
		Arrays.fill(nHiddenNeuron, length, length + pad, nHiddenNeuron[length-1]);
		return nHiddenNeuron;
	}

	
	/**
	 * Create array of hidden neurons.
	 * @param nInput number of input neurons.
	 * @param nOutput number of output neurons.
	 * @param hiddenLayerMin minimum number of hidden layers.
	 * @return array of hidden neurons.
	 */
	static int[] constructHiddenNeuronNumbers(int nInput, int nOutput, int hiddenLayerMin) {
		return constructHiddenNeuronNumbers(nInput, nOutput, 2, hiddenLayerMin);
	}
	

	/**
	 * Create array of hidden neurons.
	 * @param nInput number of input neurons.
	 * @param nOutput number of output neurons.
	 * @return array of hidden neurons.
	 */
	static int[] constructHiddenNeuronNumbers(int nInput, int nOutput) {
		return constructHiddenNeuronNumbers(nInput, nOutput, 2, 0);
	}
	
	
}
