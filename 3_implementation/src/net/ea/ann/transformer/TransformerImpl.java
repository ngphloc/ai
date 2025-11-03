/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.awt.Dimension;
import java.rmi.RemoteException;
import java.util.Collections;
import java.util.List;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerExt;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.TaskTrainer;
import net.ea.ann.raster.Raster;
import net.ea.ann.transformer.TransformerImpl.Attention0;
import net.ea.ann.transformer.TransformerImpl.TransformerBasic;

/**
 * This class implements extensive transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerImpl extends TransformerAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef ID reference.
	 */
	public TransformerImpl(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public TransformerImpl(int neuronChannel) {
		this(neuronChannel, null);
	}


	@Override
	protected TransformerBasic createEncoder() {
		Encoder encoder = new Encoder(this.neuronChannel, this.idRef);
		encoder.paramSetNorm(this.paramIsNorm());
		encoder.paramSetVectorized(this.paramIsVectorized());
		return encoder;
	}
	
	
	@Override
	protected TransformerBasic createDecoder() {
		Decoder decoder = new Decoder(this.neuronChannel, this.idRef);
		decoder.paramSetNorm(this.paramIsNorm());
		decoder.paramSetVectorized(this.paramIsVectorized());
		return decoder;
	}

	
	/**
	 * This class represents transformer encoder.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public class Encoder extends TransformerBasic {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel and ID reference.
		 * @param neuronChannel neuron channel.
		 * @param idRef ID reference.
		 */
		Encoder(int neuronChannel, Id idRef) {
			super(neuronChannel, idRef);
		}

		/**
		 * Default constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		Encoder(int neuronChannel) {
			this(neuronChannel, null);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, int ffnDepth, int nBlocks, int XBlockIndex) {
			return super.initialize(h, n, dm, dk, dv, 0, 0, ffnDepth, nBlocks, -1);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
			return super.initialize(h, n, dm, dk, dv, 0, 0, ffnDepth, nBlocks, -1);
		}
		
	}
	
	
	/**
	 * This class represents transformer decoder.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public class Decoder extends TransformerBasic {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel and ID reference.
		 * @param neuronChannel neuron channel.
		 * @param idRef ID reference.
		 */
		Decoder(int neuronChannel, Id idRef) {
			super(neuronChannel, idRef);
		}

		/**
		 * Default constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		Decoder(int neuronChannel) {
			this(neuronChannel, null);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, int ffnDepth, int nBlocks, int XBlockIndex) {
			return super.initialize(h, n, dm, dk, dv, m, d, ffnDepth, nBlocks, XBlockIndex);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
			return super.initialize(h, n, dm, dk, dv, n, dm, ffnDepth, nBlocks, 1);
		}
		
	}

	
	/**
	 * This class implements extensive basic transformer.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public class TransformerBasic extends net.ea.ann.transformer.TransformerBasic {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel and ID reference.
		 * @param neuronChannel neuron channel.
		 * @param idRef ID reference.
		 */
		TransformerBasic(int neuronChannel, Id idRef) {
			super(neuronChannel, idRef);
		}

		/**
		 * Default constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		TransformerBasic(int neuronChannel) {
			this(neuronChannel, null);
		}

		@Override
		protected net.ea.ann.transformer.TransformerBlock createBlock() {
			TransformerBlock block = new TransformerBlock(this.neuronChannel, idRef);
			block.paramSetNorm(this.paramIsNorm());
			block.paramSetVectorized(this.paramIsVectorized());
			return block;
		}
		
	}
	
	
	/**
	 * This class represents extensive transformer block.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public class TransformerBlock extends net.ea.ann.transformer.TransformerBlock {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel, vectorization flag, and ID reference.
		 * @param neuronChannel neuron channel.
		 * @param idRef ID reference.
		 */
		TransformerBlock(int neuronChannel, Id idRef) {
			super(neuronChannel, idRef);
		}

		/**
		 * Default constructor with neuron channel and vectorization flag.
		 * @param neuronChannel neuron channel.
		 */
		TransformerBlock(int neuronChannel) {
			this(neuronChannel, null);
		}

		@Override
		protected net.ea.ann.transformer.Attention createAttention() {return new Attention();}
		
	}
	
	
	/**
	 * This class represents extensive attention.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	class Attention extends net.ea.ann.transformer.Attention {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Default constructor.
		 */
		Attention() {
			super();
		}

		@Override
		protected net.ea.ann.transformer.Attention0 creatHead() {return new Attention0();}

	}
	
	
	/**
	 * This class represents extensive attention head.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	class Attention0 extends net.ea.ann.transformer.Attention0 {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Y input data with regard to query weight matrix Q and key weight matrix K.
		 */
		protected Matrix Yqk = null;
		
		/**
		 * Assigning input matrices.
		 * @param Y Y input data.
		 * @param X X input data.
		 * @param M masked matrix.
		 * @param Yqk Y input data with regard to query weight matrix Q and key weight matrix K.
		 */
		void assignInputs(Matrix Y, Matrix X, boolean[][] M, Matrix Yqk) {
			super.assignInputs(Y, X, M);
			if (this.Yqk != null && Yqk != null) this.Yqk = Yqk;
		}

		/**
		 * Initializing attention with sample size, model dimension, key dimension, value dimension, other sample size, other model dimension, query-key Y data flag, and zero value.
		 * @param n sample size.
		 * @param dm model dimension. Default model dimension is {@link #MODEL_DIMENSION_DEFAULT}.
		 * @param dk key dimension. Default key dimension is {@link #KEY_DIMENSION_DEFAULT}.
		 * @param dv value dimension. Default value dimension is {@link #VALUE_DIMENSION_DEFAULT}.
		 * @param m other sample size.
		 * @param d other model dimension. Default other model dimension is {@link #MODEL_DIMENSION_DEFAULT}.
		 * @param containsYqk flag to indicate whether to have Y data with regard to query matrix and key matrix.
		 * @param zero zero value.
		 * @return true if initialization is successful.
		 */
		boolean initialize(int n, int dm, int dk, int dv, int m, int d, boolean containsYqk, NeuronValue zero) {
			if (!initialize(n, dm, dk, dv, m, d, zero)) return false;
			if (!containsYqk) return validate();
			
			this.Y = Matrix.create(n, dm, zero);
			return validate();
		}

			
		@Override
		public boolean validate() {
			if (!super.validate()) return false;
			if ((Yqk != null) && (Yqk.rows() != n() || Yqk.columns() != dm())) return false;
			return true;
		}

		/**
		 * Getting Y input data with regard to query weight matrix Q and key weight matrix K.
		 * @return Y input data with regard to query weight matrix Q and key weight matrix K.
		 */
		public Matrix Yqk() {return Yqk;};
		
		@Override
		public Matrix calcQ() {
			Matrix transposedX = calcTransposedX();
			return transposedX != null || Yqk == null ? super.calcQ() : Yqk.multiply(WQ);
		}
		
		@Override
		public Matrix calcK() {
			Matrix transposedX = calcTransposedX();
			return transposedX != null || Yqk == null ? super.calcK() : Yqk.multiply(WK);
		}

		/**
		 * Default constructor.
		 */
		Attention0() {
			super();
		}
		
	}
	
	
}



/**
 * This class implements abstract transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class TransformerAbstract extends NetworkAbstract implements Transformer, MatrixLayerExt {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Transformer encoder.
	 */
	protected TransformerBasic encoder = null;
	
	
	/**
	 * Transformer decoder.
	 */
	protected TransformerBasic decoder = null;

	
	/**
	 * Previous layer.
	 */
	protected MatrixLayer prevLayer = null;
	
	
	/**
	 * Next layer.
	 */
	protected MatrixLayer nextLayer = null;

	
	/**
	 * List of trainers.
	 */
	protected List<TaskTrainer> trainers = Util.newList(0);

	
	/**
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef ID reference.
	 */
	public TransformerAbstract(int neuronChannel, Id idRef) {
		super(idRef);
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, MatrixNetworkAbstract.VECTORIZED_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public TransformerAbstract(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	public Id getIdRef() {
		return idRef;
	}


	@Override
	public int id() {
		return idRef.get();
	}

	
	/**
	 * Validating transformer block.
	 * @return true if transformer block is valid.
	 */
	public boolean validate() {
		return encoder != null || decoder != null;
	}

	
	/**
	 * Resetting transformer.
	 */
	public void reset() {
		encoder = null;
		decoder = null;
	}
	

	/**
	 * Creating encoder.
	 * @return encoder.
	 */
	protected abstract TransformerBasic createEncoder();
	
	
	/**
	 * Creating decoder.
	 * @return decoder.
	 */
	protected abstract TransformerBasic createDecoder();

	
	/**
	 * Initializing transformer.
	 * @param he number of heads (encoder). Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param ne sample size (encoder).
	 * @param dme model dimension (encoder). Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dke key dimension (encoder). Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dve value dimension (encoder). Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param hd number of heads (decoder). Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param nd sample size (decoder).
	 * @param dmd model dimension (decoder). Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dkd key dimension (decoder). Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dvd value dimension (decoder). Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link #BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	private boolean initialize(int he, int ne, int dme, int dke, int dve, int hd, int nd, int dmd, int dkd, int dvd, int ffnDepth, int nBlocks) {
		nBlocks = nBlocks > 0 ? nBlocks : TransformerBasic.BLOCKS_NUMBER_DEFAULT;
		int XBlockIndex = nd > 0 && dmd > 0 && nBlocks > 1 ? 1 : -1;
		this.encoder = null;
		this.decoder = null;
		
		if (XBlockIndex < 0) {
			this.encoder = createEncoder();
			if (!this.encoder.initialize(he, ne, dme, dke, dve, ffnDepth, nBlocks)) return false;
		}
		else {
			this.encoder = createEncoder();
			if (!this.encoder.initialize(he, ne, dme, dke, dve, ffnDepth, nBlocks)) return false;
			Dimension es = this.encoder.getOutputLayer().getSize();
			if (es.height != nd || es.width != dmd) {
				if (!this.encoder.setOutputAdapter(new Dimension(dmd, nd), ffnDepth)) return false;
				this.decoder = createDecoder();
				if (!this.decoder.initialize(hd, nd, dmd, dkd, dvd, nd, dmd, ffnDepth, nBlocks, XBlockIndex)) return false;
			}
			else {
				this.decoder = createDecoder();
				if (!this.decoder.initialize(hd, nd, dmd, dkd, dvd, ne, dme, ffnDepth, nBlocks, XBlockIndex)) return false;
			}
			if (!this.decoder.setInputAttach(XBlockIndex, this.encoder)) return false;
		}
		return validate();
	}
	
	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link #BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
		return initialize(h, n, dm, dk, dv, h, n, dm, dk, dv, ffnDepth, nBlocks);
	}
	
	
	/**
	 * Initializing transformer.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link #BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initializeOnlyEncoder(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
		return initialize(h, n, dm, dk, dv, 0, 0, 0, 0, 0, ffnDepth, nBlocks);
	}

	
	/**
	 * Getting encoder.
	 * @return encoder.
	 */
	public TransformerBasic encoder() {return encoder;}
	
	
	/**
	 * Getting decoder.
	 * @return decoder.
	 */
	public TransformerBasic decoder() {return decoder;}

	
	@Override
	public MatrixLayer getPrevLayer() {
		return prevLayer;
	}


	@Override
	public MatrixLayer getNextLayer() {
		return nextLayer;
	}

	
	@Override
	public Matrix getInput() {
		return validate() ? (decoder != null ? decoder.getInput() : encoder.getInput()) : null;
	}
	
	/**
	 * Setting Y input data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	public void enterInputs(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		if (!validate())
			return;
		else if (encoder != null && decoder != null) {
			encoder.enterInputs(inputX, null);
			decoder.enterInputs(inputY, null, inputMask);
		}
		else if (encoder != null && decoder == null) {
			encoder.enterInputs(inputX != null ? inputX : inputY, inputMask);
		}
		else if (encoder == null && decoder != null) {
			decoder.enterInputs(inputY != null ? inputY : inputX, inputMask);
		}
	}
	
	
	/**
	 * Setting input data and input mask.
	 * @param input input data.
	 * @param inputMask input mask.
	 */
	public void enterInputs(Matrix input, boolean[][] inputMask) {
		if (!validate())
			return;
		else if (encoder != null && decoder != null) {
			decoder.enterInputs(input, inputMask);
		}
		else if (encoder != null && decoder == null) {
			encoder.enterInputs(input, inputMask);
		}
		else if (encoder == null && decoder != null) {
			decoder.enterInputs(input, inputMask);
		}
	}

	
	/**
	 * Setting input data.
	 * @param input input data.
	 */
	public void enterInputs(Matrix input) {
		enterInputs(input, null);
	}

	
	/**
	 * Setting input mask.
	 * @param inputMask input mask.
	 */
	public void enterInputs(boolean[][] inputMask) {
		enterInputs(null, inputMask);
	}

	
	@Override
	public void enterInputs(net.ea.ann.mane.Record record) {
		Matrix inputY = record.input();
		Matrix inputX = record.input2();
		Object extraInput = record.extraInput();
		boolean[][] inputMask = (extraInput != null) && (extraInput instanceof boolean[][]) ? (boolean[][])extraInput : null;
		enterInputs(inputY, inputX, inputMask);
	}

	
	@Override
	public Matrix getOutput() {
		return validate() ? (decoder != null ? decoder.getOutput() : encoder.getOutput()) : null;
	}


	@Override
	public MatrixLayerAbstract getOutputLayer() {
		return validate() ? (decoder != null ? decoder.getOutputLayer() : encoder.getOutputLayer()) : null;
	}
	
	
//	@Override
//	public Function getOutputActivateRef() {
//		MatrixLayerAbstract outputLayer = getOutputLayer();
//		return outputLayer != null ? outputLayer.getActivateRef() : null;
//	}

	
	/**
	 * Getting output adapter.
	 * @return output adapter.
	 */
	public MatrixNetworkImpl getOutputAdapter() {
		if (!validate())
			return null;
		else if (encoder != null && decoder != null) {
			return decoder.getOutputAdapter();
		}
		else if (encoder != null && decoder == null) {
			return encoder.getOutputAdapter();
		}
		else if (encoder == null && decoder != null) {
			return decoder.getOutputAdapter();
		}
		else
			return null;
	}

	
	/**
	 * Setting output adapter.
	 * @param adapter output adapter.
	 */
	public boolean setOutputAdapter(MatrixNetworkImpl adapter) {
		if (!validate())
			return false;
		else if (encoder != null && decoder != null) {
			return decoder.setOutputAdapter(adapter);
		}
		else if (encoder != null && decoder == null) {
			return encoder.setOutputAdapter(adapter);
		}
		else if (encoder == null && decoder != null) {
			return decoder.setOutputAdapter(adapter);
		}
		else
			return false;
	}

	
	/**
	 * Setting output adapter.
	 * @param outputAdapterOutputSize output size of output adapter.
	 * @param outputAdapterDepth output adapter  depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(Dimension outputAdapterOutputSize, int outputAdapterDepth) {
		if (!validate())
			return false;
		else if (encoder != null && decoder != null) {
			return decoder.setOutputAdapter(outputAdapterOutputSize, outputAdapterDepth);
		}
		else if (encoder != null && decoder == null) {
			return encoder.setOutputAdapter(outputAdapterOutputSize, outputAdapterDepth);
		}
		else if (encoder == null && decoder != null) {
			return decoder.setOutputAdapter(outputAdapterOutputSize, outputAdapterDepth);
		}
		else
			return false;
	}

	
	/**
	 * Setting output adapter.
	 * @param middleSize middle size.
	 * @param middleFilter middle filter.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(Dimension middleSize, Filter2D middleFilter, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		if (!validate())
			return false;
		else if (encoder != null && decoder != null) {
			return decoder.setOutputAdapter(middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth);
		}
		else if (encoder != null && decoder == null) {
			return encoder.setOutputAdapter(middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth);
		}
		else if (encoder == null && decoder != null) {
			return decoder.setOutputAdapter(middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth);
		}
		else
			return false;
	}
	

	/**
	 * Setting output adapter.
	 * @param middleSize middle size.
	 * @param middleFilterStride middle filter stride.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(Dimension middleSize, Dimension middleFilterStride, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		if (!validate())
			return false;
		else if (encoder != null && decoder != null) {
			return decoder.setOutputAdapter(middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth);
		}
		else if (encoder != null && decoder == null) {
			return encoder.setOutputAdapter(middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth);
		}
		else if (encoder == null && decoder != null) {
			return decoder.setOutputAdapter(middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth);
		}
		else
			return false;
	}

	
	/**
	 * Removing output adapter.
	 */
	public TransformerAbstract removeOutputAdapter() {
		if (!validate()) return this;
		if (encoder != null && decoder != null) {
			decoder.removeOutputAdapter();
		}
		else if (encoder != null && decoder == null) {
			encoder.removeOutputAdapter();
		}
		else if (encoder == null && decoder != null) {
			decoder.removeOutputAdapter();
		}
		return this;
	}
	
	
	/**
	 * Getting feed-forward network.
	 * @return feed-forward network.
	 */
	public MatrixNetworkImpl getOutputFFN() {
		if (!validate())
			return null;
		else if (encoder != null && decoder != null) {
			return decoder.getOutputFFN();
		}
		else if (encoder != null && decoder == null) {
			return encoder.getOutputFFN();
		}
		else if (encoder == null && decoder != null) {
			return decoder.getOutputFFN();
		}
		else
			return null;
	}

	
	/**
	 * Setting output feed-forward network.
	 * @param ffn feed-forward network.
	 * @return true if setting is successful.
	 */
	public boolean setOutputFFN(MatrixNetworkImpl ffn) {
		if (!validate())
			return false;
		else if (encoder != null && decoder != null) {
			return decoder.setOutputFFN(ffn);
		}
		else if (encoder != null && decoder == null) {
			return encoder.setOutputFFN(ffn);
		}
		else if (encoder == null && decoder != null) {
			return decoder.setOutputFFN(ffn);
		}
		else
			return false;
	}

	
	/**
	 * Getting size of trainers.
	 * @return size of trainers.
	 */
	int getTrainerSize() {
		return trainers.size();
	}
	
	
	/**
	 * Getting trainer at specified index.
	 * @param index specified index.
	 * @return trainer at specified index.
	 */
	TaskTrainer getTrainer(int index) {
		return trainers.get(index);
	}
	
	
	/**
	 * Getting trainer.
	 * @return the first trainer.
	 */
	public TaskTrainer getTrainer() {
		return trainers.size() > 0 ? trainers.get(0) : null;
	}
	
	
	/**
	 * Adding trainer.
	 * @param trainer specified trainer.
	 * @return adding is successful.
	 */
	boolean addTrainer(TaskTrainer trainer) {
		return trainers.add(trainer);
	}
	
	
	/**
	 * Removing trainer.
	 * @param trainer specified trainer.
	 * @return removal is successful.
	 */
	boolean removeTrainer(TaskTrainer trainer) {
		return trainers.remove(trainer);
	}
	
	
	/**
	 * Clearing trainer.
	 */
	void clearTrainers() {
		trainers.clear();
	}
	
	
	/**
	 * Setting trainer.
	 * @param trainer specified trainer.
	 * @return this network.
	 */
	public TransformerAbstract setTrainer(TaskTrainer trainer) {
		trainers.clear();
		if (trainer != null) trainers.add(trainer);
		return this;
	}


	@Override
	public Matrix forward(net.ea.ann.mane.Record...inputs) {
		Matrix inputY = inputs != null && inputs.length > 0 ? inputs[0].input() : null;
		Matrix inputX = inputs != null && inputs.length > 0 ? inputs[0].input2() : null;
		Object extraInput = inputs != null && inputs.length > 0 ? inputs[0].extraInput() : null;
		boolean[][] inputMask = (extraInput != null) && (extraInput instanceof boolean[][]) ? (boolean[][])extraInput : null;
		Matrix result = evaluate(inputY, inputX, inputMask, new Object[] {});
		if (result == null) return result;
		
		MatrixLayer nextLayer = null;
		while ((nextLayer = this.getNextLayer()) != null) {
			Matrix.copy(result, nextLayer.getInput());
			result = nextLayer.evaluate();
		}
		return result;
	}


	/**
	 * Evaluating transformer.
	 * @param inputY Y input.
	 * @param inputX X input.
	 * @param inputMask mask input. 
	 * @param params additional parameters.
	 * @return matrix as output.
	 */
	protected Matrix evaluate(Matrix inputY, Matrix inputX, boolean[][] inputMask, Object...params) {
		if (!validate()) return null;
		Matrix A = null;
		if (encoder != null && decoder != null) {
			A = encoder.evaluate(inputX, null, params);
			A = decoder.evaluate(inputY, A, inputMask, params);
		}
		else if (encoder != null && decoder == null) {
			A = encoder.evaluate(inputY, inputX, inputMask, params);
		}
		else if (encoder == null && decoder != null) {
			A = decoder.evaluate(inputY, inputX, inputMask, params);
		}
		
		if (params != null && params.length > 0 && params[0] != null && params[0] instanceof Error) {
			((Error)params[0]).addLayerOInput(this);
		}
		return A;
	}
	
	
	/**
	 * Evaluating transformer.
	 * @param input input.
	 * @param inputMask mask input.
	 * @param params additional parameters.
	 * @return matrix as output.
	 */
	public Matrix evaluate(Matrix input, boolean[][] inputMask, Object...params) {
		if (!validate()) return null;
		Matrix A = null;
		if (encoder != null && decoder != null) {
			A = encoder.evaluate(params);
			A = decoder.evaluate(input, A, inputMask, params);
		}
		else if (encoder != null && decoder == null) {
			A = encoder.evaluate(input, inputMask, params);
		}
		else if (encoder == null && decoder != null) {
			A = decoder.evaluate(input, inputMask, params);
		}
		
		if (params != null && params.length > 0 && params[0] != null && params[0] instanceof Error) {
			((Error)params[0]).addLayerOInput(this);
		}
		return A;
	}

	
	/**
	 * Evaluating transformer.
	 * @param input input.
	 * @param params additional parameters
	 * @return matrix as output.
	 */
	public Matrix evaluate(Matrix input, Object...params) {
		return evaluate(input, null, params);
	}
	
	
	@Override
	public Matrix evaluate(Record record) throws RemoteException {
		return evaluate(record.inputY(), record.inputX(), record.inputMask(), new Object[] {});
	}

	
	@Override
	public Matrix evaluate(Object...params) {
		return evaluate(null, null, params);
	}

	
	/**
	 * Back-warding transformer block by errors.
	 * @param errors specified errors.
	 * @param learningRate learning rate.
	 * @return learning errors. The first element is main error and the second element is attached error\\.
	 */
	protected Error[][] backward(Error[] errors, double learningRate) {
		if (!validate())
			return null;
		else if (encoder != null && decoder != null) {
			return decoder.backward(errors, learningRate);
		}
		else if (encoder != null && decoder == null) {
			return encoder.backward(errors, learningRate);
		}
		else if (encoder == null && decoder != null) {
			return decoder.backward(errors, learningRate);
		}
		else
			return null;
	}
	
	
	@Override
	public net.ea.ann.mane.Error[] backward(net.ea.ann.mane.Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (!learning) throw new IllegalArgumentException("Method Transformer::backward(Matrix[], MatrixLayer, boolean, double) does not support learning = false");
		Error[] errors = Error.create(outputErrors);
		Error[][] learnedErrors = backward(errors, learningRate);
		if (learnedErrors == null || learnedErrors.length == 0 || learnedErrors[0] == null) return null;
		
		net.ea.ann.mane.Error[] backwardErrors = Error.extract(learnedErrors[0]);
		if (this.prevLayer == null || this == focus)
			return backwardErrors;
		else
			return this.prevLayer.backward(backwardErrors, focus, learning, learningRate);
	}


	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	public net.ea.ann.mane.Error[] backward(net.ea.ann.mane.Error[] outputErrors, double learningRate) {
		return backward(outputErrors, null, true, learningRate);
	}
	
	
	@Override
	public Error[][] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = paramGetMaxIteration();
		double terminatedThreshold = paramGetTerminatedThreshold();
		double learningRate = paramGetLearningRate();
		int epochs = paramGetPseudoEpochs();

		Error[][] outputErrors = null;
		Iterable<Record> newsample = sample;
		for (int epoch = 0; epoch < epochs; epoch++) {
			double lr = calcLearningRate(learningRate, epoch+1);
			if (epoch > 0) {
				if (!(newsample instanceof List<?>)) newsample = net.ea.ann.core.Record.listOf(newsample);
				Collections.shuffle((List<?>)newsample);
			}
			outputErrors = learn(newsample, lr, terminatedThreshold, maxIteration);
		}
		return outputErrors;
	}

	
	/**
	 * Learning transformer.
	 * @param sample sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learning errors. The first element is main error and the second element is attached error\\.
	 */
	protected Error[][] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		if (!validate()) return null;
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Error[][] outputErrors = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			if (trainers.size() == 0) {
				List<Error> errorList = Util.newList(0);
				for (Record record : subsample) {
					Error error = new Error((Matrix)null);
					Matrix A = evaluate(record.inputY(), record.inputX(), record.inputMask(), error);
					Matrix err = record.outputA().subtract(A);
					if (err != null) {
						error.errorSet(err);
						errorList.add(error);
					}
				}
				outputErrors = backward(errorList.toArray(new Error[] {}), lr);
			}
			else {
				List<net.ea.ann.mane.Record> subinouts = Util.newList(0);
				for (Record record : subsample) {
					net.ea.ann.mane.Record mr = null;
					if (record.inputX() == null)
						mr = new net.ea.ann.mane.Record(record.inputY(), record.outputA());
					else
						mr = new net.ea.ann.mane.Record(record.inputY(), record.outputA(), record.inputX(), null);
					subinouts.add(mr);
				}
				net.ea.ann.mane.Error[] errors = null;
				for (TaskTrainer trainer : trainers) {
					errors = trainer.train(this, subinouts, false, learningRate);
				}
				outputErrors = new Error[][] {Error.create(errors)};
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "transformer_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (outputErrors == null || outputErrors.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = Matrix.normMean(Error.create(outputErrors[0]));
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "transformer_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return outputErrors;
	}


	/**
	 * Checking normalization mode.
	 * @return normalization mode in rang [0, 1].
	 */
	public boolean paramIsNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Setting normalization mode.
	 * @param isNorm normalization mode in rang [0, 1]..
	 * @return this transformer.
	 */
	public TransformerAbstract paramSetNorm(boolean isNorm) {
		config.put(Raster.NORM_FIELD, isNorm);
		return this;
	}

	
	/**
	 * Checking vectorization mode.
	 * @return vectorization mode.
	 */
	public boolean paramIsVectorized() {
		if (config.containsKey(MatrixNetworkAbstract.VECTORIZED_FIELD))
			return config.getAsBoolean(MatrixNetworkAbstract.VECTORIZED_FIELD);
		else
			return MatrixNetworkAbstract.VECTORIZED_DEFAULT;
	}


	/**
	 * Setting vectorization mode.
	 * @param vectorized vectorization mode.
	 * @return this network.
	 */
	public TransformerAbstract paramSetVectorized(boolean vectorized) {
		config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, vectorized);
		return this;
	}


}
