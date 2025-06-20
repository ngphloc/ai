/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.generator;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.Probability;
import net.ea.ann.core.function.Softmax;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents weighted generator.
 * 
 * @author Loc Nguyen
 * @param <T> type of trainer.
 * @version 1.0
 *
 */
public class GeneratorWeighted<T extends Trainer> extends GeneratorWeighted0<T> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public GeneratorWeighted(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
		config.put(COMB_NUMBER_FIELD, COMB_NUMBER_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public GeneratorWeighted(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public GeneratorWeighted(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	/*
	 * Method can be overridden.
	 */
	@Override
	protected Probability createWeightedFunction() {
		return super.createWeightedFunction();
	}


	/*
	 * Method can be overridden.
	 */
	@Override
	protected boolean requireWeightedFunction(Function activateRef) {
		return super.requireWeightedFunction(activateRef);
	}


	/*
	 * Method can be overridden.
	 */
	@Override
	protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object... params) {
		return super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
	}

	
}



/**
 * This class represents basic weighted generator.
 * 
 * @author Loc Nguyen
 * @param <T> type of trainer.
 * @version 1.0
 *
 */
class GeneratorWeighted0<T extends Trainer> extends GeneratorStandard<T> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field of the number elements of a combination.
	 */
	public static final String COMB_NUMBER_FIELD = "gw_comb_number";
	
	
	/**
	 * Default value for the field of the number elements of a combination.
	 */
	public static final int COMB_NUMBER_DEFAULT = 2;
	
	
	/**
	 * Outputs-classes map.
	 */
	protected Map<Integer, int[]> outputClassMap = Util.newMap(0);
	
	
	/**
	 * Classes-outputs map.
	 */
	protected Map<Integer, int[]> classOutputMap = Util.newMap(0);

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public GeneratorWeighted0(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
		config.put(COMB_NUMBER_FIELD, COMB_NUMBER_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public GeneratorWeighted0(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public GeneratorWeighted0(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	@Override
	public boolean initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, int nMemoryNeuron) {
		nOutputNeuron = nOutputNeuron < 1 ? 1 : nOutputNeuron;
		int comb = paramGetCombNumber();
		comb = comb < 1 ? 1 : comb;
		comb = comb > nOutputNeuron ? nOutputNeuron : comb;
		paramSetCombNumber(comb);

		if (!configClassInfo(nOutputNeuron)) return false;

		if (nOutputNeuron != outputClassMap.size() && nHiddenNeuron != null && nHiddenNeuron.length > 0) {
			int[] nHidden = NetworkStandard.constructHiddenNeuronNumbers(nHiddenNeuron[nHiddenNeuron.length-1], outputClassMap.size());
			if (nHidden != null && nHidden.length > 0) {
				int n = nHiddenNeuron.length;
				nHiddenNeuron = Arrays.copyOf(nHiddenNeuron, n + nHidden.length);
				for (int i = 0; i < nHidden.length; i++) nHiddenNeuron[n+i] = nHidden[i];
			}
		}
		nOutputNeuron = outputClassMap.size();
		
		if (!super.initialize(nInputNeuron, nOutputNeuron, nHiddenNeuron, nMemoryNeuron)) return false;
		Function weightedFunction = createWeightedFunction();
		if (weightedFunction != null) changeOutputLayerActivateRef(weightedFunction);
		return true;
	}

	
	/**
	 * Initializing with classes.
	 * @param nInputNeuron number of input neurons.
	 * @param nClass number of classes.
	 * @return true if initilization is successful.
	 */
	public boolean initializeWithClasses(int nInputNeuron, int nClass) {
		if (nInputNeuron < 1) return false;
		if (!configClassInfo(nClass)) return false;
		
		int[] nHiddenNeuron = NetworkStandard.constructHiddenNeuronNumbers(nInputNeuron, outputClassMap.size());
		if (!super.initialize(nInputNeuron, outputClassMap.size(), nHiddenNeuron, 0)) return false;
		Function weightedFunction = createWeightedFunction();
		if (weightedFunction != null) changeOutputLayerActivateRef(weightedFunction);
		return true;
	}
	
	
	/**
	 * Configure class information.
	 * @param nClass number of classes.
	 * @return true if configuration is successful.
	 */
	private boolean configClassInfo(int nClass) {
		if (nClass < 1) return false;
		int comb = paramGetCombNumber();
		if (comb < 1 || comb > nClass) return false;
		
		outputClassMap.clear();
		classOutputMap.clear();

		CombinationGenerator cg = new CombinationGenerator(nClass, comb);
		int index = 0;
		while (cg.hasMore()) {
			int[] classIndices = cg.getNext();
			classIndices = Arrays.copyOf(classIndices, classIndices.length);
			Arrays.sort(classIndices);
			outputClassMap.put(index, classIndices);
			index++;
		}
		if (outputClassMap.size() == 0) return false;
		
		for (int classIndex = 0; classIndex < nClass; classIndex++) {
			Set<Integer> outputIndices = outputClassMap.keySet();
			List<Integer> foundOutputIndexList = Util.newList(0);
			for (int outputIndex : outputIndices) {
				int[] classIndices = outputClassMap.get(outputIndex);
				if (Arrays.binarySearch(classIndices, classIndex) >= 0) {
					foundOutputIndexList.add(outputIndex);
				}
			}
			if (foundOutputIndexList.size() == 0) continue;
			
			int[] foundOutputIndices = new int[foundOutputIndexList.size()];
			for (int i = 0; i < foundOutputIndices.length; i++) foundOutputIndices[i] = foundOutputIndexList.get(i);
			Arrays.sort(foundOutputIndices);
			classOutputMap.put(classIndex, foundOutputIndices);
		}
		return classOutputMap.size() > 0;
	}
	
	
	/**
	 * Creating weighted function.
	 * @return weighted function.
	 */
	Probability createWeightedFunction() {
		return Softmax.create(neuronChannel, getOutputLayer());
	}
	
	
	/**
	 * Checking whether to require weighted function.
	 * @param activateRef weighted function.
	 * @return whether to require weighted function.
	 */
	boolean requireWeightedFunction(Function activateRef) {
		return activateRef != null && activateRef instanceof Softmax;
	}
	
	
	/**
	 * Checking whether to require specific function.
	 * @param neuron specified neuron.
	 * @param layer specified layer.
	 * @return whether to require specific function.
	 */
	private boolean requireWeightedFunction(NeuronStandard neuron, LayerStandard layer) {
		Function activateRef = neuron != null ? neuron.getActivateRef() : null;
		Function activateRef2 = layer != null ? Layer.getActivateRef2(layer) : null;
		return (activateRef != null && activateRef2 == null && requireWeightedFunction(activateRef)) ||
			(activateRef2 != null && requireWeightedFunction(activateRef2));
	}
	
	
	/**
	 * Getting number of classes.
	 * @return number of classes.
	 */
	public int getNumberOfClass() {
		return classOutputMap.size();
	}
	
	
	/**
	 * Creating output from class index.
	 * @param classIndex class index.
	 * @return output created from class index.
	 */
	public NeuronValue[] createOutputByClass(int classIndex) {
		LayerStandard outputLayer = getOutputLayer();
		NeuronValue zero = outputLayer.newNeuronValue().zero();
		NeuronValue unit = zero.unit();
		NeuronValue[] output = new NeuronValue[getOutputLayer().size()];
		for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
			int[] classIndices = outputClassMap.get(outputIndex);
			if (Arrays.binarySearch(classIndices, classIndex) >= 0)
				output[outputIndex] = unit;
			else
				output[outputIndex] = zero;
		}
		return output;
	}
	
	
	/**
	 * Extracting class given specified input record.
	 * @param inputRecord specified input record.
	 * @return class given specified input record.
	 */
	public int extractClass(Record inputRecord) {
		try {
			NeuronValue[] evaluated = super.evaluate(inputRecord);
			if (evaluated == null || evaluated.length == 0)
				return -1;
			else
				return extractClass();
		} catch (Throwable e) {Util.trace(e);}
		return -1;
	}


	/**
	 * Extracting class of output layer.
	 * @return class of output layer.
	 */
	public int extractClass() {
		return extractClass(getOutputLayer().getOutput());
	}
	
	
	/**
	 * Extracting class of given output.
	 * @param output specified output.
	 * @return class of given output.
	 */
	private int extractClass(NeuronValue[] output) {
		if (output == null || output.length == 0) return -1;
		int nClass = getNumberOfClass();
		if (nClass <= 0) return -1;
		
		double[] weights = weightsOfOutput(output);
		int foundClass = -1;
		double minDistance = Double.MAX_VALUE;
		for (int classIndex = 0; classIndex < nClass; classIndex++) {
			NeuronValue[] output2 = createOutputByClass(classIndex);
			double[] weights2 = weightsOfOutput(output2);
			double distance = 0;
			for (int i = 0; i < weights.length; i++) {
				double d = weights[i] - weights2[i];
				distance += d*d;
			}
			distance = Math.sqrt(distance);
			
			if (distance < minDistance) {
				minDistance = distance;
				foundClass = classIndex;
			}
		}
		return foundClass;
	}

	
//	/**
//	 * Extracting class of given output.
//	 * @param output specified output.
//	 * @return class of given output.
//	 */
//	private int extractClass2(NeuronValue[] output) {
//		if (output == null || output.length == 0) return -1;
//		double[] weights = weightsOfOutput(output);
//		IndexedWeight[] indexedWeights = IndexedWeight.create(weights);
//		IndexedWeight.sort(indexedWeights, false);
//		
//		int outputIndex = indexedWeights[0].index;
//		int[] classIndices = outputClassMap.get(outputIndex);
//		List<Integer> classResults = Util.newList(classIndices.length);
//		for (int j = 0; j < classIndices.length; j++) classResults.add(classIndices[j]);
//		
//		for (int i = 1; i < indexedWeights.length; i++) {
//			outputIndex = indexedWeights[i].index;
//			classIndices = outputClassMap.get(outputIndex);
//			List<Integer> classCandidates = Util.newList(0);
//			for (int j = 0; j < classIndices.length; j++) classCandidates.add(classIndices[j]);
//			classResults.removeAll(classCandidates); //Unique aspect.
//			if (classResults.size() == 1) return classResults.get(0);
//		}
//		
//		return classResults.size() > 0 ? classResults.get(0) : -1;
//	}
	
	
	/**
	 * Getting weights of specified output.
	 * @param output specified output.
	 * @return weights of specified output.
	 */
	private double[] weightsOfOutput(NeuronValue[] output) {
		double[] weights = new double[output.length];
		for (int i = 0; i < weights.length; i++) weights[i] = output[i].mean();
		return weights;
	}
	
	
	/**
	 * Getting the number elements of a combination.
	 * @return the number elements of a combination.
	 */
	public int paramGetCombNumber() {
		int combNumber = config.getAsInt(COMB_NUMBER_FIELD);
		return combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
	}
	
	
	/**
	 * Setting the number elements of a combination.
	 * @param combNumber the number elements of a combination.
	 * @return this generator.
	 */
	public GeneratorWeighted0<T> paramSetCombNumber(int combNumber) {
		combNumber = combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
		config.put(COMB_NUMBER_FIELD, combNumber);
		return this;
	}
	
	
	@Override
	protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		if (outputNeuronIndex < 0 || realOutputs == null || realOutputs.length == 0)
			return super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		if (!requireWeightedFunction(outputNeuron, outputLayer))
			return super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		
		if (realOutput == null) return null;
		NeuronValue errorSum = realOutput.zero();
		for (int i = 0; i < realOutputs.length; i++) {
			NeuronValue output = outputNeuron != null ? outputNeuron.getOutput() : null;
			NeuronValue error = calcOutputErrorWeighted(output, realOutputs[i], i==outputNeuronIndex);
			errorSum = errorSum.add(error);
		}
		return errorSum;
	}


	/**
	 * Calculating error by weighted function function.
	 * @param output neuron output.
	 * @param realOutput real output.
	 * @param match matching flag.
	 * @return error.
	 */
	private NeuronValue calcOutputErrorWeighted(NeuronValue output, NeuronValue realOutput, boolean match) {
		if (output == null || realOutput == null) return null;
		NeuronValue unit = realOutput.unit();
		if (match)
			return realOutput.multiply(unit.subtract(output));
		else
			return realOutput.multiply(output.negative());
	}

	
}


/**
 * This class represents combination generator.
 * @author Someone on internet (http://www.merriampark.com/comb.htm)
 * @version 1.0
 *
 */
class CombinationGenerator implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Combination index array.
	 */
	private int[] a;
	
	/**
	 * Number of elements.
	 */
	private int n;
	
	
	/**
	 * Number of elements of a combination.
	 */
	private int r;
	
	
	/**
	 * Remaining number of combinations.
	 */
	private BigInteger numLeft;
	
	
	/**
	 * Total number of combinations.
	 */
	private BigInteger total;
	
	
	/**
	 * Constructor with number of elements and number of a combination.
	 * @param n number of elements.
	 * @param r number of a combination.
	 */
	public CombinationGenerator (int n, int r) {
		if (r > n) throw new IllegalArgumentException ();
		if (n < 1) throw new IllegalArgumentException ();

		this.n = n;
		this.r = r;
		a = new int[r];
		BigInteger nFact = getFactorial (n);
		BigInteger rFact = getFactorial (r);
		BigInteger nminusrFact = getFactorial (n - r);
		total = nFact.divide (rFact.multiply (nminusrFact));
		reset ();
	}
	
	
	/**
	 * Resetting.
	 */
	public void reset () {
		for (int i = 0; i < a.length; i++) {
			a[i] = i;
		}
		numLeft = new BigInteger (total.toString ());
	}

	
	/**
	 * Getting remaining number of combinations.
	 * @return remaining number of combinations.
	 */
	public BigInteger getNumLeft () {
		return numLeft;
	}


	/**
	 * Checking whether there more combinations.
	 * @return whether there more combinations.
	 */
	public boolean hasMore () {
		return numLeft.compareTo (BigInteger.ZERO) == 1;
	}

	
	/**
	 * Getting total number of combinations.
	 * @return total number of combinations.
	 */
	public BigInteger getTotal () {
		return total;
	}

	
	/**
	 * Computing factorial.
	 * @param n specified number.
	 * @return factorial of specified number.
	 */
	private static BigInteger getFactorial (int n) {
		BigInteger fact = BigInteger.ONE;
		for (int i = n; i > 1; i--) {
			fact = fact.multiply (new BigInteger (Integer.toString (i)));
		}
		return fact;
	}

	
	/**
	 * Generate next combination (algorithm from Rosen p. 286).
	 * @return next combination
	 */
	public int[] getNext () {
		if (numLeft.equals (total)) {
			numLeft = numLeft.subtract (BigInteger.ONE);
			return a;
		}

		int i = r - 1;
		while (a[i] == n - r + i) {
			i--;
		}
		a[i] = a[i] + 1;
		for (int j = i + 1; j < r; j++) {
			a[j] = a[i] + j - i;
		}

		numLeft = numLeft.subtract (BigInteger.ONE);
		return a;
	}
	
	
	/**
	 * Main method as example.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		String[] elements = {"a", "b", "c", "d", "e", "f", "g"};
		int[] indices;
		CombinationGenerator x = new CombinationGenerator(elements.length, 3);
		StringBuffer combination;
		while (x.hasMore ()) {
			combination = new StringBuffer ();
			indices = x.getNext ();
			for (int i = 0; i < indices.length; i++) {
				combination.append (elements[indices[i]]);
			}
			System.out.println(combination.toString ());
		}
	}
	
	
}


