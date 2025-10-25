/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;
import java.rmi.NoSuchObjectException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.List;

/**
 * This class is basic abstract implementation of neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class NetworkAbstract implements Network, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default value for learning rate.
	 */
	public final static double LEARN_RATE_MINIMUM = Float.MIN_VALUE;

	
	/**
	 * Maximum iteration of learning neural network.
	 */
	public final static String LEARN_MAX_ITERATION_FIELD = "net_learn_max_iteration";
	
	
	/**
	 * Terminated threshold of learning neural network.
	 */
	public final static String LEARN_TERMINATED_THRESHOLD_FIELD = "net_learn_terminated_threshold";

	
	/**
	 * Learning rate.
	 */
	public final static String LEARN_RATE_FIELD = "net_learn_rate";

	
	/**
	 * Name of learning rate fixed field.
	 */
	public final static String LEARN_RATE_FIXED_FIELD = "net_learn_rate_fixed";

	
	/**
	 * Default value of learning rate fixed field.
	 */
	public final static boolean LEARN_RATE_FIXED_DEFAULT = true;

	
	/**
	 * Name of learning terminated error field.
	 */
	public final static String LEARN_TERMINATE_ERROR_FIELD = "net_learn_terminate_error";

	
	/**
	 * Default value of learning terminated error field.
	 */
	public final static boolean LEARN_TERMINATE_ERROR_DEFAULT = false;

	
	/**
	 * Name of re-sampling field.
	 */
	public final static String RESAMPLE_FILED = "net_resample";
	
	
	/**
	 * Default value of re-sampling field.
	 */
	public final static boolean RESAMPLE_DEFAULT = false;
	
	
	/**
	 * Name of re-sampling rating field.
	 */
	public final static String RESAMPLE_RATE_FILED = "net_resample_rate";
	
	
	/**
	 * Default value of re-sampling rating field.
	 */
	public final static double RESAMPLE_RATE_DEFAULT = 1;
	
	
	/**
	 * Name of pseudo-epochs field.
	 */
	public final static String EPOCHS_PSEUDO_FILED = "net_epochs_pseudo";
	
	
	/**
	 * Default value of pseudo-epochs field.
	 */
	public final static int EPOCHS_PSEUDO_DEFAULT = 1;

	
	/**
	 * Default value of zoom-out.
	 */
	public final static int ZOOMOUT_DEFAULT = 2;

	
	/**
	 * Field name of minimum number of hidden layers.
	 */
	public final static String HIDDEN_LAYER_MIN_FILED = "ann_min_hidden";

	
	/**
	 * Default value for minimum number of hidden layers.
	 */
	public final static int HIDDEN_LAYER_MIN_DEFAULT = 0;

	
	/**
	 * Name of learning one-by-one field.
	 */
	public final static String LEARN_ONE_FIELD = "ann_learn_one";

	
	/**
	 * Default value of learning one-by-one field.
	 */
	public final static boolean LEARN_ONE_DEFAULT = false;

	
	/**
	 * Holding a list of listeners.
	 */
    protected transient NetworkListenerList listenerList = new NetworkListenerList();

    
    /**
     * Flag to indicate whether algorithm learning process was started.
     */
    protected volatile boolean doStarted = false;
    
    
    /**
     * Flag to indicate whether algorithm learning process was paused.
     */
    protected volatile boolean doPaused = false;

    
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();
	
	
	/**
	 * Flag to indicate whether this hidden Markov model was exported.
	 */
	protected boolean exported = false;

	
    /**
	 * Internal identifier.
	 */
	protected Id idRef = new Id();
	
	
	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	protected NetworkAbstract(Id idRef) {
		config.put(LEARN_MAX_ITERATION_FIELD, LEARN_MAX_ITERATION_DEFAULT);
		config.put(LEARN_TERMINATED_THRESHOLD_FIELD, LEARN_TERMINATED_THRESHOLD_DEFAULT);
		config.put(LEARN_RATE_FIELD, LEARN_RATE_DEFAULT);
		config.put(LEARN_TERMINATE_ERROR_FIELD, LEARN_TERMINATE_ERROR_DEFAULT);
		config.put(RESAMPLE_FILED, RESAMPLE_DEFAULT);
		config.put(RESAMPLE_RATE_FILED, RESAMPLE_RATE_DEFAULT);
		config.put(LEARN_RATE_FIXED_FIELD, LEARN_RATE_FIXED_DEFAULT);
		config.put(EPOCHS_PSEUDO_FILED, EPOCHS_PSEUDO_DEFAULT);

		if (idRef != null) this.idRef = idRef;
	}

	
	/**
	 * Default constructor.
	 */
	protected NetworkAbstract() {
		this(new Id());
	}

	
	/**
	 * Re-sampling records.
	 * @param <T> record type.
	 * @param records specified records.
	 * @param iteration current iteration.
	 * @param maxIteration maximum iteration
	 * @return re-sampled sample.
	 */
	protected <T> Iterable<T> resample(Iterable<T> records, int iteration, int maxIteration) {
		List<T> sample = null;
		if (config.getAsBoolean(RESAMPLE_FILED)) {
			double resampleRate = paramGetResampleRate();
			sample = Record.resample(records, resampleRate);
		}
		else {
			sample = Record.listOf(records);
		}
		if (maxIteration <= 1) return sample;
		
		int sampleSize = Record.sizeOf(records);
		int batchSize = sampleSize / maxIteration;
		if (batchSize == 0) return sample;
		iteration = iteration < 0 ? 0 : iteration;
		iteration = iteration >= maxIteration ? maxIteration-1 : iteration;
		int index = iteration*batchSize;
		int length = index + batchSize;
		length = length >= sample.size() ? sample.size() : length;
		return sample.subList(index, length);
	}
	
	
	/**
	 * Calculating learning rate.
	 * @param initialLearningRate initial learning rate.
	 * @param iteration current iteration.
	 * @return learning rate.
	 */
	protected double calcLearningRate(double initialLearningRate, int iteration) {
		boolean fixedLearningRate = LEARN_RATE_FIXED_DEFAULT;
		if (config.containsKey(LEARN_RATE_FIXED_FIELD)) fixedLearningRate = config.getAsBoolean(LEARN_RATE_FIXED_FIELD);
		return calcLearningRate(initialLearningRate, iteration, fixedLearningRate);
	}
	
	
	/**
	 * Calculating learning rate.
	 * @param initialLearningRate initial learning rate.
	 * @param iteration current iteration.
	 * @param fixedLearningRate fixed learning rate.
	 * @return learning rate.
	 */
	public static double calcLearningRate(double initialLearningRate, int iteration, boolean fixedLearningRate) {
		initialLearningRate = Double.isNaN(initialLearningRate) || initialLearningRate <= 0 || initialLearningRate > 1 ? LEARN_RATE_DEFAULT : initialLearningRate;
		if (iteration <= 1 || fixedLearningRate) return initialLearningRate;
		double learningRate = initialLearningRate * (1.0/Math.sqrt(iteration));
		return Math.max(learningRate, LEARN_RATE_MINIMUM);
	}

	
	/**
	 * Getting learning rate.
	 * @return learning rate.
	 */
	public double paramGetLearningRate() {
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
	}
	
	
	/**
	 * Setting learning rate.
	 * @param learningRate learning rate.
	 * @return this network.
	 */
	public NetworkAbstract paramSetLearningRate(double learningRate) {
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		config.put(LEARN_RATE_FIELD, learningRate);
		return this;
	}
	
	

	/**
	 * Getting terminated threshold.
	 * @return terminated threshold.
	 */
	public double paramGetTerminatedThreshold() {
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		return Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
	}
	
	
	/**
	 * Setting terminated threshold.
	 * @param terminatedThreshold terminated threshold.
	 * @return this network.
	 */
	public NetworkAbstract paramSetTerminatedThreshold(double terminatedThreshold) {
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		config.put(LEARN_TERMINATED_THRESHOLD_FIELD, terminatedThreshold);
		return this;
	}
	
	
	
	/**
	 * Getting re-sampling rate.
	 * @return re-sampling rate.
	 */
	public double paramGetResampleRate() {
		double resampleRate = config.getAsReal(RESAMPLE_RATE_FILED);
		return Double.isNaN(resampleRate) || resampleRate <= 0 || resampleRate > 1 ? RESAMPLE_RATE_DEFAULT : resampleRate;
	}

	
	/**
	 * Getting maximum iteration.
	 * @return maximum iteration.
	 */
	public int paramGetMaxIteration() {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		return maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
	}

	
	/**
	 * Setting maximum iteration.
	 * @param maxIteration maximum iteration.
	 * @return this network.
	 */
	public NetworkAbstract paramSetMaxIteration(int maxIteration) {
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		config.put(LEARN_MAX_ITERATION_FIELD, maxIteration);
		return this;
	}
	
	
	/**
	 * Getting pseudo-epochs.
	 * @return pseudo-epochs.
	 */
	public int paramGetPseudoEpochs() {
		int pseudoEpochs = config.getAsInt(EPOCHS_PSEUDO_FILED);
		return pseudoEpochs > 0 ? pseudoEpochs : EPOCHS_PSEUDO_DEFAULT;
	}

	
	/**
	 * Setting maximum iteration.
	 * @param maxIteration maximum iteration.
	 * @return this network.
	 */
	public NetworkAbstract paramSetPseudoEpochs(int pseudoEpochs) {
		pseudoEpochs = pseudoEpochs > 0 ? pseudoEpochs :  EPOCHS_PSEUDO_DEFAULT;
		config.put(EPOCHS_PSEUDO_FILED, pseudoEpochs);
		return this;
	}

	
	/**
	 * Getting batches.
	 * @return batches.
	 */
	public int paramGetBatches() {
		return paramGetMaxIteration();
	}

	
	/**
	 * Setting batches.
	 * @param batches batches.
	 * @return this network.
	 */
	public NetworkAbstract paramSetBatches(int batches) {
		return paramSetMaxIteration(batches);
	}
	
	
	/**
	 * Including parameters from other network.
	 * @param other other network.
	 * @return this network.
	 */
	public NetworkAbstract paramSetInclude(NetworkAbstract other) {
		this.config.putAll(other.config);
		return this;
	}
	
	
	@Override
	public void addListener(NetworkListener listener) throws RemoteException {
		synchronized (listenerList) {
			listenerList.add(NetworkListener.class, listener);
		}
	}


	@Override
	public void removeListener(NetworkListener listener) throws RemoteException {
		synchronized (listenerList) {
			listenerList.remove(NetworkListener.class, listener);
		}
	}
	
	
	/**
	 * Getting an array of listeners.
	 * @return array of listeners.
	 */
	protected NetworkListener[] getListeners() {
		if (listenerList == null) return new NetworkListener[] {};
		synchronized (listenerList) {
			return listenerList.getListeners(NetworkListener.class);
		}

	}
	
	
	/**
	 * Firing information event.
	 * @param evt information event.
	 */
	protected void fireInfoEvent(NetworkInfoEvent evt) {
		if (listenerList == null) return;
		
		NetworkListener[] listeners = getListeners();
		for (NetworkListener listener : listeners) {
			try {
				listener.receivedInfo(evt);
			}
			catch (Throwable e) { 
				Util.trace(e);
			}
		}
	}

	
	/**
	 * Firing learning event.
	 * @param evt learning event.
	 */
	protected void fireDoEvent(NetworkDoEvent evt) {
		if (listenerList == null) return;
		
		NetworkListener[] listeners = getListeners();
		for (NetworkListener listener : listeners) {
			try {
				listener.receivedDo(evt);
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
	}


	@Override
	public boolean doPause() throws RemoteException {
		if (!isDoRunning()) return false;
		
		doPaused  = true;
		
		try {
			wait();
		} 
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return true;
	}


	@Override
	public boolean doResume() throws RemoteException {
		if (!isDoPaused()) return false;
		
		doPaused = false;
		notifyAll();
		
		return true;
	}


	@Override
	public boolean doStop() throws RemoteException {
		if (!isDoStarted()) return false;
		
		doStarted = false;
		
		if (doPaused) {
			doPaused = false;
			notifyAll();
		}
		
		try {
			wait();
		} 
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return true;
	}


	@Override
	public boolean isDoStarted() throws RemoteException {
		return doStarted;
	}


	@Override
	public boolean isDoPaused() throws RemoteException {
		return doStarted && doPaused;
	}


	@Override
	public boolean isDoRunning() throws RemoteException {
		return doStarted && !doPaused;
	}

	
	@Override
	public NetworkConfig getConfig() throws RemoteException {
		return config;
	}


	@Override
	public void setConfig(NetworkConfig config) throws RemoteException {
		if (config != null) this.config.putAll(config);
	}


	@Override
	public synchronized Remote export(int serverPort) throws RemoteException {
		if (exported) return null;
		
		Remote stub = null;
		try {
			stub = UnicastRemoteObject.exportObject(this, serverPort);
		}
		catch (Exception e) {
			try {
				if (stub != null) UnicastRemoteObject.unexportObject(this, true);
			}
			catch (Exception e2) {}
			stub = null;
		}
	
		exported = stub != null;
		return stub;
	}


	@Override
	public synchronized void unexport() throws RemoteException {
		if (!exported) return;

		try {
        	UnicastRemoteObject.unexportObject(this, true);
			exported = false;
		}
		catch (NoSuchObjectException e) {
			exported = false;
			Util.trace(e);
		}
		catch (Throwable e) {
			Util.trace(e);
		}
	}

	
	@Override
	public void close() throws Exception {
		try {
			unexport();
		}
		catch (Throwable e) {
			Util.trace(e);
		}
	}


}
