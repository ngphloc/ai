/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn.lstm;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.Evaluator;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This class represents a state (standard network) in recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class State extends NetworkStandardImpl {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal parameter index.
	 */
	private int paramIndex = 0;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public State(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRefIndexed(this.neuronChannel, true)) : activateRef;
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public State(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	public State(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
	@Override
	public List<List<LayerStandard>> getRiboutbones() {
		return super.getShortRiboutbones();
	}
	
	
	/**
	 * Getting the number of parameters.
	 * @return the number of parameters.
	 */
	private int getParamCount() {
		return Neuron.PARAMS_NUM;
	}

	
	/**
	 * Getting parameter index.
	 * @return parameter index.
	 */
	private int getParamIndex() {
		return paramIndex;
	}
	
	
	/**
	 * Setting parameter index.
	 * @param paramIndex parameter index.
	 */
	private void setParamIndex(int paramIndex) {
		this.paramIndex = paramIndex;
	}
	
	
	/**
	 * Getting this state.
	 * @return this state.
	 */
	private State getThisState( ) {
		return this;
	}
	
	
	@Override
	protected LayerStandard newLayer() {
		return new Layer(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected int getParamCount() {
				return getThisState().getParamCount();
			}

			@Override
			protected int getParamIndex() {
				return getThisState().getParamIndex();
			}

			@Override
			protected void setParamIndex(int paramIndex) {
				getThisState().setParamIndex(paramIndex);
			}
			
		};
		
	}
	
	
	@Override
	public NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : sample) {
				if (record == null) continue;
				NeuronValue[] output = record.output != null? NeuronValue.adjustArray(record.output, backbone.get(backbone.size()-1).size(), backbone.get(backbone.size()-1)) : null;
				
				//Evaluating network.
				try {
					evaluate(record);
				} catch (Throwable e) {Util.trace(e);}

				for (int i = 0; i < getParamCount(); i++) {
					//Setting parameter index. This code line is very important.
					setParamIndex(i);
					
					//Learning backbone.
					error = bp.updateWeightsBiases(backbone, output, lr);
					
					//Learning rib-bone and memory.
					learnRibMem(record, lr);
				}
				//Recovering the first index.
				setParamIndex(0);
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "lstm_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) doStarted = false; 
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "lstm_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}
	
	
	@Override
	public NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			boolean[] evaluated = new boolean[] {false};
			for (int i = 0; i < getParamCount(); i++) {
				//Setting parameter index. This code line is very important.
				setParamIndex(i);
				
				//Learning backbone.
				error = bp.updateWeightsBiases(sample, backbone, lr, new Evaluator() {
					
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public NeuronValue[] evaluate(Record inputRecord) throws RemoteException {
						if (evaluated[0]) {
							evaluated[0] = true;
							return evaluate(inputRecord);
						}
						else
							return null;
					}
					
				});			
				
				//Learning rib-bone and memory.
				learnRibMem(sample, lr);
			}
			//Recovering the first index.
			setParamIndex(0);
		
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "lstm_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) doStarted = false; 
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}//End while
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "lstm_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
}


