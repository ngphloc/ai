package net.ea.ann.gen.gan;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents adversarial network.
 * @author Loc Nguyen
 * @version 1.0
 */
public class AdversarialNetwork extends GeneratorStandard<Trainer> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal previous output.
	 */
	protected NeuronValue[] prevOutput = null;
	
	
	/**
	 * Extra error.
	 */
	protected NeuronValue[] extraError = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public AdversarialNetwork(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public AdversarialNetwork(int neuronChannel, Function activateRef) {
		super(neuronChannel, activateRef);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	public AdversarialNetwork(int neuronChannel) {
		super(neuronChannel);
	}
	
	
	@Override
	public void reset() {
		super.reset();
		prevOutput = null;
		extraError = null;
	}


	/**
	 * Evaluating entire network with accumulative previous output.
	 * @param inputRecord input record.
	 * @return array as output of output layer.
	 */
	public boolean evaluateSetPrevOutputAccum(Record inputRecord) {
		NeuronValue[] prevOutput = null;
		try {
			prevOutput = super.evaluate(inputRecord);
		} catch (Throwable e) {Util.trace(e);}
		if (prevOutput == null) return false;
		
		if (this.prevOutput == null)
			this.prevOutput = prevOutput;
		else {
			for (int i = 0; i < this.prevOutput.length; i++)
				this.prevOutput[i] = this.prevOutput[i].add(prevOutput[i]);
		}
		return true;
	}

	
	/**
	 * Setting previous output.
	 * @param prevOutput specified previous output. 
	 * @return old previous output.
	 */
	public NeuronValue[] setPrevOutput(NeuronValue[] prevOutput) {
		NeuronValue[] prevPrev = this.prevOutput;
		this.prevOutput = prevOutput;
		return prevPrev;
	}

	
	/**
	 * Getting previous output.
	 * @return previous output.
	 */
	public NeuronValue[] getPrevOutput() {
		return prevOutput;
	}
	
	
	/**
	 * Setting extra error.
	 * @param extraError extra error. 
	 * @return previous extra error.
	 */
	public NeuronValue[] setExtraError(NeuronValue[] extraError) {
		NeuronValue[] prevExtraError = this.extraError;
		this.extraError = extraError;
		return prevExtraError;
	}

	
	/**
	 * Getting extra error.
	 * @return extra error.
	 */
	public NeuronValue[] getExtraError() {
		return extraError;
	}

	
	/*
	 * It is derivative of the sum log(d(x)) + log(1-d(g(z))) which results 1/d(x) - 1/(1-d(g(z))).
	 */
	@Override
	protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		NeuronValue error = outputNeuron.getOutput().zero();
		if (prevOutput != null && outputLayer != null) {
			int index = outputLayer.indexOf(outputNeuron);
			if (index >= 0) {
				NeuronValue prevX = prevOutput[index];
				NeuronValue prevError = calcErrorAdv1(outputNeuron, prevX, null);
				error = error.add(prevError);
			}
		}
		
		NeuronValue generatedX = outputNeuron.getOutput();
		NeuronValue generatedError = calcErrorAdv2(outputNeuron, generatedX, null);
		error = error.add(generatedError);
		
		if (extraError != null) {
			NeuronValue extraErrorSum = outputNeuron.getOutput().zero();
			for (NeuronValue e : extraError) extraErrorSum = extraErrorSum.add(e);
			
			//This technique is very important. Please pay attention to it.
			NeuronValue derivative = outputNeuron.derivative();
			extraErrorSum = extraErrorSum.multiplyDerivative(derivative);
			error = error.add(extraErrorSum);
		}
		
		return error;
	}
	
	
	/**
	 * Calculate decoded error regarding adversarial network.
	 * It is minus derivative of log(1 - d(g(z))) which is 1 / (1-d(g(z))).
	 * @param neuron an decoded neuron.
	 * @param adv adversarial network.
	 * @return decoded error regarding adversarial network.
	 */
	public static NeuronValue calcDecodedErrorAdv(NeuronStandard neuron, AdversarialNetwork adv) {
		NeuronValue generatedX = neuron.getOutput();
		if (adv == null) return generatedX.zero();
		
		Function f = adv.getOutputLayerActivateRefOutermost();
		NeuronValue decodedErrorAdv = calcErrorAdv2(neuron, generatedX, f).negative();
		
		//This technique is very important. Please pay attention to it.
		NeuronValue localDerivative = neuron.derivative();
		decodedErrorAdv = decodedErrorAdv.multiplyDerivative(localDerivative);
		
		return decodedErrorAdv;
	}


	/**
	 * Calculate the adversarial error given the real value X.
	 * It is derivative of log(d(x)) which is 1/d(x)
	 * @param neuron specified neuron.
	 * @param realX real value X.
	 * @param f activation function.
	 * @return the adversarial error given the real value X.
	 */
	public static NeuronValue calcErrorAdv1(NeuronStandard neuron, NeuronValue realX, Function f) {
		NeuronValue v = realX.inverse();
		if (v == null) return realX.zero();
		
		f = f != null ? f : neuron.getActivateRef();
		NeuronValue derivative = f.derivative(realX);
		return v.multiplyDerivative(derivative);
	}

	
	/**
	 * Calculate the adversarial error given the generated value X.
	 * It is derivative of log(1 - d(g(z))) which is -(1 / (1-d(g(z)))).
	 * @param neuron specified neuron.
	 * @param generatedX generated value X.
	 * @param f activation function.
	 * @return the adversarial error given the generated value X.
	 */
	public static NeuronValue calcErrorAdv2(NeuronStandard neuron, NeuronValue generatedX, Function f) {
		NeuronValue v = generatedX.subtract(generatedX.unit());
		v = v.inverse();
		if (v == null) return generatedX.zero();
		
		f = f != null ? f : neuron.getActivateRef();
		NeuronValue derivative = f.derivative(generatedX);
		return v.multiplyDerivative(derivative);
	}
	
	
}


