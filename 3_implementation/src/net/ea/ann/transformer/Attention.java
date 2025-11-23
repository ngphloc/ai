/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents standard attention (multi-head attention).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Attention implements AddNorm, Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of attentions.
	 */
	protected Attention0[] heads = null;
	
	
	/**
	 * Entire weight matrix.
	 */
	protected Matrix WO = null;
	
	
	/**
	 * Attention output data.
	 */
	protected Matrix A = null;

	
	/**
	 * Add & norm layer.
	 */
	protected AddNormNetwork addNorm = null;
	
	
	/**
	 * Default constructor.
	 */
	protected Attention() {
		super();
	}

	
	/**
	 * Resetting attention.
	 */
	public void reset() {
		heads = null;
		WO = null;
		A = null;
		addNorm = null;
	}
	
	
	/**
	 * Creating head.
	 * @return head.
	 */
	protected Attention0 creatHead() {return new Attention0();}
	
	
	/**
	 * Initializing attention with number of heads, sample size, model dimension, key dimension, value dimension, other sample size, other model dimension, and zero value.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param m other sample size.
	 * @param d other model dimension. Default other model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param zero zero value.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, NeuronValue zero) {
		if (h <= 0) return false;
		
		this.heads = new Attention0[h];
		for (int i = 0; i < h; i++) {
			try {
				Attention0 head = creatHead();
				if (head.initialize(n, dm, dk, dv, m, d, zero))
					this.heads[i] = head;
				else {
					reset();
					return false;
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		
		Matrix Y = this.heads[0].Y;
		Matrix X = this.heads[0].X;
		boolean[][] M = this.heads[0].M;
		for (int i = 1; i < h; i++) {
			Attention0 head = this.heads[i];
			head.assignInputs(Y, X, M);
		}
		
		this.WO = h*dv != dm ? Matrix.create(h*dv, dm, zero) : null;
		this.A = Matrix.create(n, dm, zero);
		
		return validate();
	}

	
	/**
	 * Initializing attention with number of heads, sample size, model dimension, key dimension, value dimension, and zero value.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param zero zero value.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, NeuronValue zero) {
		return initialize(h, n, dm, dk, dv, 0, 0, zero);
	}
	
	
	/**
	 * Checking whether this attention is valid.
	 * @return whether this attention is valid.
	 */
	public boolean validate() {
		if (heads == null || h() <= 0) return false;
		for (int i = 0; i < heads.length; i++) {
			if (!heads[i].validate()) return false;
		}
		
		if (WO != null) {
			if (WO.rows() != h()*dv() || WO.columns() != dm()) return false;
		}
		if (A == null || A.rows() != n() || A.columns() != dm()) return false;

		//Add & Norm can be null.
		
		return true;
	}
	
	
	/**
	 * Getting number of attentions (heads).
	 * @return number of attentions (heads).
	 */
	public int h() {
		return heads != null ? heads.length : 0;
	}
	
	
	/**
	 * Getting head at specified index.
	 * @param index specified index.
	 * @return head at specified index.
	 */
	public Attention0 head(int index) {
		return heads[index];
	}
	
	
	/**
	 * Getting other sample size.
	 * @return other sample size.
	 */
	public int m() {
		return heads.length > 0 ? heads[0].m() : 0;
	}

	
	/**
	 * Getting model dimension of other sample.
	 * @return model dimension of other sample.
	 */
	public int d() {
		return heads.length > 0 ? heads[0].d() : 0;
	}

	
	/**
	 * Getting sample size.
	 * @return sample size.
	 */
	public int n() {
		return heads.length > 0 ? heads[0].n() : 0;
	}
	
	
	/**
	 * Getting model dimension.
	 * @return model dimension.
	 */
	public int dm() {
		return heads.length > 0 ? heads[0].dm() : 0;
	}
	
	
	/**
	 * Getting key dimension.
	 * @return key dimension.
	 */
	public int dk() {
		return heads.length > 0 ? heads[0].dk() : 0;
	}
	
	
	/**
	 * Getting value dimension.
	 * @return value dimension.
	 */
	public int dv() {
		return heads.length > 0 ? heads[0].dv() : 0;
	}

	
	/**
	 * Getting X input data.
	 * @return X input data.
	 */
	public Matrix X() {
		return heads.length > 0 ? heads[0].X : null;
	}
	
	
	/**
	 * Getting Y input data.
	 * @return Y input data.
	 */
	public Matrix Y() {
		return heads.length > 0 ? heads[0].Y : null;
	}

	
	/**
	 * Getting masked matrix.
	 * If an element of this masked matrix is true, the corresponding value is ignored (masked).
	 * @return masked matrix.
	 */
	public boolean[][] M() {
		return heads.length > 0 ? heads[0].M : null;
	}

	
	/**
	 * Getting entire weight matrix.
	 * @return entire weight matrix.
	 */
	public Matrix WO() {
		return WO;
	}

	
	/**
	 * Getting attention output data.
	 * @return attention output data.
	 */
	public Matrix A() {
		return A;
	}

	
	/**
	 * Getting attentions of heads.
	 * @return attentions of heads.
	 */
	private Matrix[] headsA() {
		Matrix[] As = new Matrix[heads.length];
		for (int i = 0; i < heads.length; i++) As[i] = heads[i].A;
		return As;
	}
	
	
	/**
	 * Setting mask.
	 * @param inputMask input mask.
	 */
	protected void setMask(boolean[][] inputMask) {
		boolean[][] M = M();
		if (M != null && inputMask != null) NeuronValue.copy(inputMask, M);
	}

	
	/**
	 * Setting mask over range.
	 * @param row row index.
	 * @param column column index.
	 * @param range range.
	 * @param masked masked flag.
	 */
	protected void setMask(int row, int column, int range, boolean masked) {
		boolean[][] M = M();
		if (M == null) return;
		int n = n();
		if (n <= 0 || row < 0 || row >= n || column < 0 || column >= n) return;
		range = column + range <= n ? range : n - column;
		for (int j = 0; j < range; j++) M[row][j+column] = masked;
	}

	
	/**
	 * Setting mask at specified row and column.
	 * @param row row index.
	 * @param column column index.
	 * @param masked masked flag.
	 */
	protected void setMask(int row, int column, boolean masked) {
		boolean[][] M = M();
		if (M != null) M[row][column] = masked;
	}

		
	/**
	 * Setting mask at specified row.
	 * @param row row index.
	 * @param masked masked flag.
	 */
	protected void setMask(int row, boolean masked) {
		boolean[][] M = M();
		if (M == null) return;
		int n = n();
		if (row < 0 || row >= n) return;
		for (int j = 0; j < n; j++) M[row][j] = masked;
	}
	
	
	/**
	 * Setting mask over all mask matrix.
	 * @param masked masked flag.
	 */
	protected void setMask(boolean masked) {
		boolean[][] M = M();
		if (M == null) return;
		int n = n();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) M[i][j] = masked;
		}
	}
	
	
	/**
	 * Setting Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	protected void enterInputs(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		if (Y() != null && inputY != null) Matrix.copy(inputY, Y());
		if (X() != null && inputX != null) Matrix.copy(inputX, X());
		if (M() != null && inputMask != null) NeuronValue.copy(inputMask, M());
	}

		
	/**
	 * Evaluating attention given Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 * @param params additional parameters.
	 * @return evaluated attention.
	 */
	protected Matrix evaluate(Matrix inputY, Matrix inputX, boolean[][] inputMask, Object...params) {
		if (!validate()) return null;
		enterInputs(inputY, inputX, inputMask);
		
		Matrix[] aList = new Matrix[heads.length];
		for (int i = 0; i < heads.length; i++) aList[i] = heads[i].evaluate();
		
		Matrix eval = WO != null ? Matrix.concatV(aList).multiply(WO) : Matrix.concatV(aList);
		Matrix.copy(eval, A);
		
		if (params != null && params.length > 0 && params[0] != null && params[0] instanceof Error) {
			((Error)params[0]).addLayerOInput(this, A);
		}
		return A;
	}

	
	/**
	 * Back-warding attention by errors.
	 * @param errors specified errors.
	 * @param learningRate learning rate.
	 * @return learning errors.
	 */
	protected Error[] backward(Error[] errors, double learningRate) {
		if (!validate() || errors == null || errors.length == 0) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? NetworkAbstract.LEARN_RATE_DEFAULT : learningRate;
		List<List<Matrix>> headErrsList = Util.newList(this.heads.length);
		for (int i = 0; i < this.heads.length; i++) {
			List<Matrix> headErrors = Util.newList(0);
			headErrsList.add(headErrors);
		}

		int count = 0;
		Matrix dWO = null;
		Matrix[] headsA = headsA();
		Matrix As = Matrix.concatV(headsA).transpose();
		for (Error error : errors) {
			if (error == null) continue;
			Matrix err = error.error();
			if (err == null || err.rows() != this.A.rows() || err.columns() != this.A.columns()) continue;
			count++;
			
			//Calculating errors of WO.
			if (this.WO != null) {
				Matrix d = As.multiply(err);
				dWO = dWO != null ? dWO.add(d) : d;
			}
			
			//Calculating attention errors.
			Matrix ERROR = this.WO != null ? err.multiply(this.WO.transpose()) : err;
			int index = 0;
			for (int i = 0; i < this.heads.length; i++) {
				List<Matrix> headErrors = headErrsList.get(i);
				int n = this.heads[i].A.columns();
				Matrix ERRORi = ERROR.getColumns(index, n);
				headErrors.add(ERRORi);
				index += n;
			}
		}
		if (count == 0) return null;
		
		//Training entire weight matrix WO.
		if (this.WO != null) {
			dWO = dWO.divide0(count);
			this.WO = this.WO.add(dWO.multiply0(learningRate));
		}
		
		//Training every attention head.
		List<Error[]> headOutputErrorsList = Util.newList(0);
		for (int i = 0; i < this.heads.length; i++) {
			List<Matrix> headErrs = headErrsList.get(i);
			Error[] headErrors = Error.create(headErrs.toArray(new Matrix[] {}));
			headErrors = this.heads[i].backward(headErrors, learningRate);
			headOutputErrorsList.add(headErrors);
		}
		
		//Accumulating errors.
		Error[] outputErrors = headOutputErrorsList.get(0);
		for (int i = 1; i < headOutputErrorsList.size(); i++) {
			Error.accum(outputErrors, headOutputErrorsList.get(i));
		}
		
		//Restoring layer inputs.
		for (int i = 0; i < errors.length; i++) errors[i].errorSet(outputErrors[i].error());
		return errors;
	}
	
	
	/**
	 * Initializing attention parameters.
	 * @param attention attention.
	 * @param rnd randomizer.
	 */
	static void initParams(Attention attention, Random rnd) {
		if (attention.heads != null) {
			for (Attention0 head : attention.heads) {
				if (head != null) Attention0.initParams(head, rnd);
			}
		}
		if (attention.WO != null) Matrix.fill(attention.WO, rnd);
	}


}



/**
 * This class represents attention head.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class Attention0 implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default value of model dimension which is dm.
	 */
	public static final int MODEL_DIMENSION_DEFAULT = 512;
	
	
	/**
	 * Number of heads.
	 */
	public static final int HEADS_NUMBER_DEFAULT = 8;
	
	
	/**
	 * Default value of key dimension which is dk.
	 */
	public static final int KEY_DIMENSION_DEFAULT = MODEL_DIMENSION_DEFAULT/HEADS_NUMBER_DEFAULT;

	
	/**
	 * Default value of value dimension which is dv.
	 */
	public static final int VALUE_DIMENSION_DEFAULT = KEY_DIMENSION_DEFAULT;

	
	/**
	 * X input data.
	 */
	protected Matrix X = null;

	
	/**
	 * The first transposition weight matrix transposes X input data.
	 */
	protected Matrix T1 = null;

	
	/**
	 * The second transposition weight matrix transposes X input data.
	 */
	protected Matrix T2 = null;

	
	/**
	 * Y input data.
	 */
	protected Matrix Y = null;
	
	
	/**
	 * Masked matrix.
	 * If an element of this masked matrix is true, the corresponding value is ignored (masked).
	 */
	protected boolean[][] M = null;
	
	
	/**
	 * Query weight matrix.
	 */
	protected Matrix WQ = null;
	
	
	/**
	 * Key weight matrix.
	 */
	protected Matrix WK = null;
	
	
	/**
	 * Value weight matrix.
	 */
	protected Matrix WV = null;
	
	
	/**
	 * Attention output data.
	 */
	protected Matrix A = null;

	
	/**
	 * Default constructor.
	 */
	Attention0() {
		super();
	}

	
	/**
	 * Resetting attention.
	 */
	void reset() {
		X = T1 = Y = WQ = WK = WV = A = null;
		M = null;
	}
	
	
	/**
	 * Initializing attention with sample size, model dimension, key dimension, value dimension, other sample size, other model dimension, and zero value.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link #MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link #KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link #VALUE_DIMENSION_DEFAULT}.
	 * @param m other sample size.
	 * @param d other model dimension. Default other model dimension is {@link #MODEL_DIMENSION_DEFAULT}.
	 * @param zero zero value.
	 * @return true if initialization is successful.
	 */
	boolean initialize(int n, int dm, int dk, int dv, int m, int d, NeuronValue zero) {
		if (n <= 0 || dm <= 0 || dv <= 0 || zero == null) return false;
		if (m <= 0 || d <= 0) {
			m = 0;
			d = 0;
		}
		
		this.X = this.T1 = this.T2 = null;
		if (m > 0 && m != n && d != dm) {
			this.T1 = Matrix.create(n, m, zero);
			this.X = Matrix.create(m, d, zero);
			this.T2 = Matrix.create(d, dm, zero);
		}
		else if (m > 0 && m != n && d == dm) {
			this.T1 = Matrix.create(n, m, zero);
			this.X = Matrix.create(m, dm, zero);
		}
		else if (m > 0 && m == n && d != dm) {
			this.X = Matrix.create(m, d, zero);
			this.T2 = Matrix.create(d, dm, zero);
		}
		else if (m > 0 && m == n && d == dm) {
			this.X = Matrix.create(m, d, zero);
		}
		
		this.Y = Matrix.create(n, dm, zero);
		this.M = new boolean[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) this.M[i][j] = false;
		}
		
		this.WQ = Matrix.create(dm, dk, zero);
		this.WK = Matrix.create(dm, dk, zero);
		this.WV = Matrix.create(dm, dv, zero);
		this.A = Matrix.create(n, dv, zero);
		
		return validate();
	}
	
	
	/**
	 * Assigning input matrices.
	 * @param Y Y input data.
	 * @param X X input data.
	 * @param M masked matrix.
	 */
	void assignInputs(Matrix Y, Matrix X, boolean[][] M) {
		if (this.Y != null && Y != null) this.Y = Y;
		if (this.X != null && X != null) this.X = X;
		if (this.M != null && M != null) this.M = M;
	}
	
	
	/**
	 * Checking whether this attention is valid.
	 * @return whether this attention is valid.
	 */
	public boolean validate() {
		if (Y == null || A == null || M == null || WQ == null || WK == null || WV == null) return false;
		int n = n();
		int dm = dm();
		int dk = dk();
		int dv = dv();
		if (n <= 0 || dm <= 0 || dk <= 0 || dv <= 0) return false;
		
		if (X != null) {
			if ((T1 != null) && (T1.rows() <= 0 || T1.rows() != n || T1.columns() <= 0)) return false;
			if ((T2 != null) && (T2.rows() <= 0 || T2.columns() <= 0 || T2.columns() != dm)) return false;
			if ((T1 == null && T2 == null) && (X.rows() != Y.rows() || X.columns() != Y.columns())) return false;
			if ((T1 != null && T2 == null) && (X.columns() != Y.columns())) return false;
			if ((T1 == null && T2 != null) && (X.rows() != Y.rows())) return false;
		}
		else {
			if (T1 != null || T2 != null) return false;
		}
		
		if (Y.rows() != n || Y.columns() != dm) return false;
		if (A.rows() != n || A.columns() != dv) return false;
		if (M.length != n || M[0].length != n) return false;
		
		if (WQ.rows() != dm || WQ.columns() != dk) return false;
		if (WK.rows() != dm || WK.columns() != dk) return false;
		if (WV.rows() != dm || WV.columns() != dv) return false;
		
		return true;
	}
	
	
	/**
	 * Getting other sample size.
	 * @return other sample size.
	 */
	public int m() {
		return X != null ? X.rows() : 0;
	}

	
	/**
	 * Getting model dimension of other sample.
	 * @return model dimension of other sample.
	 */
	public int d() {
		return X != null ? X.columns() : 0;
	}
	
	
	/**
	 * Getting sample size.
	 * @return sample size.
	 */
	public int n() {
		return Y != null ? Y.rows() : 0;
	}
	
	
	/**
	 * Getting model dimension.
	 * @return model dimension.
	 */
	public int dm() {
		return WQ != null ? WQ.rows() : 0;
	}
	
	
	/**
	 * Getting key dimension.
	 * @return key dimension.
	 */
	public int dk() {
		return WK != null ? WK.columns() : 0;
	}
	
	
	/**
	 * Getting value dimension.
	 * @return value dimension.
	 */
	public int dv() {
		return WV != null ? WV.columns() : 0;
	}

	
	/**
	 * Getting X input data.
	 * @return X input data.
	 */
	public Matrix X() {
		return X;
	}
	
	
	/**
	 * Getting Y input data.
	 * @return Y input data.
	 */
	public Matrix Y() {
		return Y;
	}

	
	/**
	 * Getting the first transposition weight matrix.
	 * @return the first transposition weight matrix.
	 */
	public Matrix T1() {
		return T1;
	}
	
	
	/**
	 * Getting the second transposition weight matrix.
	 * @return the second transposition weight matrix.
	 */
	public Matrix T2() {
		return T2;
	}
	
	
	/**
	 * Getting masked matrix.
	 * If an element of this masked matrix is true, the corresponding value is ignored (masked).
	 * @return masked matrix.
	 */
	public boolean[][] M() {
		return M;
	}

	
	/**
	 * Getting the query weight matrix.
	 * @return the query weight matrix.
	 */
	public Matrix WQ() {
		return WQ;
	}

	
	/**
	 * Getting the key weight matrix.
	 * @return the key weight matrix.
	 */
	public Matrix WK() {
		return WK;
	}

	
	/**
	 * Getting the value weight matrix.
	 * @return the value weight matrix.
	 */
	public Matrix WV() {
		return WV;
	}

	
	/**
	 * Getting attention output data.
	 * @return attention output data.
	 */
	public Matrix A() {
		return A;
	}

	
	/**
	 * Calculating transposed input matrix X.
	 * @return transposed input matrix X.
	 */
	public Matrix calcTransposedX() {
		if (X == null)
			return null;
		else if (T1 != null && T2 != null)
			return T1.multiply(X).multiply(T2);
		else if (T1 != null && T2 == null)
			return T1.multiply(X);
		else if (T1 == null && T2 != null)
			return X.multiply(T2);
		else
			return X;
	}
	
	
	/**
	 * Calculating query matrix Q.
	 * @return query matrix Q.
	 */
	public Matrix calcQ() {
		Matrix transposedX = calcTransposedX();
		return transposedX != null ? transposedX.multiply(WQ) : Y.multiply(WQ);
	}
	
	
	/**
	 * Calculating key matrix K.
	 * @return key matrix K.
	 */
	public Matrix calcK() {
		Matrix transposedX = calcTransposedX();
		return transposedX != null ? transposedX.multiply(WK) : Y.multiply(WK);
	}

	
	/**
	 * Calculating value matrix V.
	 * @return value matrix V.
	 */
	public Matrix calcV() {
		return Y.multiply(WV);
	}

	
	/**
	 * Calculating product of query matrix and key matrix.
	 * @return product of query matrix and key matrix.
	 */
	public Matrix calcQK() {
		Matrix Q = calcQ();
		Matrix K = calcK();
		double factor = Math.sqrt(dk());
		return Q.multiply(K.transpose()).divide0(factor);
	}
	
	
	/**
	 * Evaluating soft-max function of query matrix and key matrix.
	 * @return soft-max matrix.
	 */
	public Matrix calcQKSoftmax() {
		Matrix QK = calcQK();
		
		int n = n();
		NeuronValue zero = QK.get(0, 0).zero();
		Matrix softmax = Matrix.create(n, n, zero);
		for (int i = 0; i < n; i++) {
			NeuronValue sum = zero;
			for (int j = 0; j < n; j++) {
				NeuronValue value = this.M[i][j] ? zero : QK.get(i, j).exp();
				softmax.set(i, j, value);
				sum = sum.add(value);
			}
			
			if (sum.canInvert()) {
				for (int j = 0; j < n; j++) {
					NeuronValue value = softmax.get(i, j);
					value = value.divide(sum);
					softmax.set(i, j, value);
				}
			}
			else {
				NeuronValue prob = zero.valueOf(1.0 / (double)n);
				for (int j = 0; j < n; j++) softmax.set(i, j, prob);
			}
		}
		
		return softmax;
	}
	
	
	/**
	 * Calculating derivative of soft-max function of query matrix and key matrix.
	 * @param row row index.
	 * @return derivative of soft-max matrix.
	 */
	public Matrix calcQKSoftmaxGradient(int row) {
		Matrix softmax = calcQKSoftmax();
		
		int n = softmax.rows();
		NeuronValue zero = softmax.get(0, 0).zero();
		NeuronValue unit = zero.unit();
		Matrix softmaxGrad = Matrix.create(n, n, zero);
		for (int i = 0; i < n; i++) {
			NeuronValue value1 = softmax.get(row, i);
			for (int j = 0; j < n; j++) {
				NeuronValue value2 = softmax.get(row, j);
				NeuronValue prob = i == j ? value1.multiply(unit.subtract(value2)) : value1.multiply(value2.negative());
				softmaxGrad.set(i, j, prob);
			}
		}
		
		double factor = Math.sqrt(dk());
		return softmaxGrad.divide0(factor);
	}
	
	
	/**
	 * Setting Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	void enterInputs(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		if (Y != null && inputY != null) Matrix.copy(inputY, Y);
		if (X != null && inputX != null) Matrix.copy(inputX, X);
		if (M != null && inputMask != null) NeuronValue.copy(inputMask, M);
	}

	
	/**
	 * Evaluating attention given Y input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @return evaluated attention.
	 */
	Matrix evaluate() {
		Matrix V = calcV();
		Matrix softmax = calcQKSoftmax();
		Matrix eval = softmax.multiply(V);
		Matrix.copy(eval, A);
		return A;
	}
	
	
	/**
	 * Learning attention by errors.
	 * @param errors specified errors.
	 * @param learningRate learning rate.
	 * @return learning errors.
	 */
	Error[] backward(Error[] errors, double learningRate) {
		if (errors == null || errors.length == 0) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? NetworkAbstract.LEARN_RATE_DEFAULT : learningRate;
		
		Matrix dW = null, dWV = null, dT1 = null, dT2 = null;
		int count = 0;
		int n = n();
		Matrix V = calcV();
		List<Matrix> YBiasList = Util.newList(0);
		List<Matrix> XBiasList = Util.newList(0);
		for (Error error : errors) {
			if (error == null) continue;
			Matrix err = error.error();
			if (err == null || err.rows() != n || err.columns() != dv()) continue;
			count++;

			Matrix errv = err.multiply(V.transpose());
			Matrix softmax = calcQKSoftmax();
			Matrix[] softmaxGrads = new Matrix[n];
			for (int i = 0; i < n; i++) softmaxGrads[i] = calcQKSoftmaxGradient(i);
			
			//Training weight query matrix and weight key matrix.
			for (int i = 0; i < n; i++) {
				Matrix errvi = errv.getRow(i).multiply(softmaxGrads[i]);
				Matrix d = Y().getRow(i).transpose().multiply(errvi);
				dW = dW != null ? dW.add(d) : d;
			}
			
			//Training weight value matrix.
			Matrix dWVTemp = Y().transpose().multiply(softmax.transpose()).multiply(err); //Please pay attention to original error.
			dWV = dWV != null ? dWV.add(dWVTemp) : dWVTemp;
			
			//Calculating QK mean.
			Matrix QKMean = calcK().multiply(this.WQ.transpose());
			QKMean = QKMean.add(calcQ().multiply(this.WK.transpose()));
//			QKMean = QKMean.multiply0(0.5);

			//Calculating Y bias with regard to Q and K.
			Matrix YBiasQK = null;
			Matrix[] YBiasQKs = new Matrix[n];
			for (int i = 0; i < n; i++) {
				Matrix errvi = errv.getRow(i).multiply(softmaxGrads[i]);
				YBiasQKs[i] = errvi.multiply(QKMean);
			}
			YBiasQK = Matrix.concatH(YBiasQKs);
			
			//Calculating Y bias with regard to V.
			Matrix YBiasV = softmax.transpose().multiply(err).multiply(this.WV.transpose());
			Matrix YBias = YBiasQK.add(YBiasV);
			YBiasList.add(YBias);
			
			//Updating X bias.
			Matrix XBias = this.X != null ? YBias : null;
			if (this.T1 == null && this.T2 == null) {
				if (XBias != null) XBiasList.add(XBias);
				continue;
			}

			//Training the first transposition matrix T1.
			if (this.T1 != null) {
				Matrix[] t1s = new Matrix[n];
				for (int i = 0; i < n; i++) {
					if (this.T2 != null)
						t1s[i] = YBiasQKs[i].multiply(this.T2.transpose()).multiply(X().transpose());
					else
						t1s[i] = YBiasQKs[i].multiply(X().transpose());
				}
				Matrix dT1Temp = Matrix.concatH(t1s);
				dT1 = dT1 != null ? dT1.add(dT1Temp) : dT1Temp;
				
				//Updating X bias with regard to T1.
				if (XBias != null) XBias = this.T1.transpose().multiply(XBias);
			}
			
			//Training the second transposition matrix T2.
			if (this.T2 != null) {
				Matrix dT2Temp = null;
				for (int i = 0; i < n; i++) {
					Matrix errvi = errv.getRow(i).multiply(softmaxGrads[i]);
					Matrix d = this.T1 != null ? this.T1.getRow(i).transpose().multiply(errvi) : errvi;
					dT2Temp = dT2Temp != null ? dT2Temp.add(d) : d;
				}
				dT2Temp = X().transpose().multiply(dT2Temp).multiply(QKMean);
				dT2 = dT2 != null ? dT2.add(dT2Temp) : dT2Temp;
				
				//Updating X bias with regard to T2.
				if (XBias != null) XBias = XBias.multiply(this.T2.transpose());
			}
			
			XBiasList.add(XBias);
		}
		if (count == 0) return null;
		
		if (dW != null) dW = dW.divide0(count);
		if (dWV != null) dWV = dWV.divide0(count);
		if (dT1 != null) dT1 = dT1.divide0(count);
		if (dT2 != null) dT2 = dT2.divide0(count);
		
		//Updating weight query matrix and weight key matrix.
		Matrix Q = calcQ();
		Matrix K = calcK();
		Matrix WQ = this.WQ.add(dW.multiply(K).multiply0(learningRate));
		Matrix.copy(WQ, this.WQ);
		Matrix WK = this.WK.add(dW.multiply(Q).multiply0(learningRate));
		Matrix.copy(WK, this.WK);
		
		//Updating weight value matrix.
		Matrix WV = this.WV.add(dWV.multiply0(learningRate));
		Matrix.copy(WV, this.WV);

		//Updating the first transposition matrix T1.
		if (this.T1 != null) {
			Matrix T1 = this.T1.add(dT1.multiply0(learningRate));
			Matrix.copy(T1, this.T1);
		}

		//Updating the first transposition matrix T2.
		if (this.T2 != null) {
			Matrix T2 = this.T2.add(dT2.multiply0(learningRate));
			Matrix.copy(T2, this.T2);
		}
		
		//Calculating backward errors.
		int N = Math.max(YBiasList.size(), XBiasList.size());
		if (N == 0) return null;
		Error[] outputErrors = new Error[N];
		for (int i = 0; i < N; i++) {
			Matrix errorY = null, errorX = null;
			if (i < YBiasList.size()) errorY = YBiasList.get(i); 
			if (i < XBiasList.size()) errorX = XBiasList.get(i);
			outputErrors[i] = new Error(errorY, errorX); 
		}
		return outputErrors;
	}
	
	
	/**
	 * Initializing attention parameters.
	 * @param attention attention.
	 * @param rnd randomizer.
	 */
	static void initParams(Attention0 attention, Random rnd) {
		if (attention.T1 != null) Matrix.fill(attention.T1, rnd);
		if (attention.T2 != null) Matrix.fill(attention.T2, rnd);
		if (attention.WQ != null) Matrix.fill(attention.WQ, rnd);
		if (attention.WK != null) Matrix.fill(attention.WK, rnd);
		if (attention.WV != null) Matrix.fill(attention.WV, rnd);
	}
	
	
}


