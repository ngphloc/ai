/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.io.Serializable;

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
public class Attention implements Cloneable, Serializable {

	
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
	protected AddNorm addNorm = null;
	
	
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
	 * Initializing attention with sample size, model dimension, value dimension, zero value, other sample size.
	 * @param h number of heads.
	 * @param n sample size.
	 * @param dm model dimension.
	 * @param dk key dimension.
	 * @param dv value dimension.
	 * @param m other sample size.
	 * @param d other model dimension.
	 * @param zero zero value.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, NeuronValue zero, int m, int d) {
		if (h <= 0) return false;
		
		heads = new Attention0[h];
		for (int i = 0; i < h; i++) {
			try {
				Attention0 head = new Attention0();
				if (head.initialize(n, dm, dk, dv, zero, m, d))
					heads[i] = head;
				else {
					reset();
					return false;
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		
		Matrix X = heads[0].X;
		Matrix Y = heads[0].Y;
		boolean[][] M = heads[0].M;
		for (int i = 1; i < h; i++) {
			Attention0 head = heads[i];
			head.assignInputs(M, Y, X);
		}
		
		WO = Matrix.create(h*dv, dm, zero);
		A = Matrix.create(n, dm, zero);
		
		return validate();
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
		
		if (WO == null || A == null) return false;
		if (WO.rows() != h()*dv() || WO.columns() != dm()) return false;
		if (A.rows() != n() || A.columns() != dm()) return false;

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
	 * Setting Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 */
	public void enterInputs(Matrix inputY, Matrix inputX) {
		if (Y() != null && inputY != null) Matrix.copy(inputY, Y());
		if (X() != null && inputX != null) Matrix.copy(inputX, X());
	}

		
	/**
	 * Evaluating attention given Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @return evaluated attention.
	 */
	public Matrix evaluate(Matrix inputY, Matrix inputX) {
		if (!validate()) return null;
		enterInputs(inputY, inputX);
		
		Matrix[] aList = new Matrix[heads.length];
		for (int i = 0; i < heads.length; i++) aList[i] = heads[i].evaluate();
		
		Matrix eval = Matrix.concatV(aList).multiply(WO);
		int n = A.rows();
		int dm = A.columns();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dm; j++) A.set(i, j, eval.get(i, j));
		}
		return A;
	}

	
	/**
	 * Evaluating attention given Y input data.
	 * @param inputY Y input data.
	 * @return evaluated attention.
	 */
	public Matrix evaluate(Matrix inputY) {
		return evaluate(inputY, null);
	}
	
	
	/**
	 * Learning attention by error.
	 * @param error specified error.
	 * @param learningRate learning rate.
	 * @param maxIteration maximum iterations.
	 */
	private void learn(Matrix error, double learningRate) {
		if (!validate()) return;
		if (error == null) return;
		if (error.rows() != A.rows() || error.columns() != A.columns()) return;
		
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? NetworkAbstract.LEARN_RATE_DEFAULT : learningRate;

		//Training entire weight matrix WO.
		Matrix[] headsA = headsA();
		Matrix As = Matrix.concatV(headsA);
		As = As.transpose().multiply(error).multiply0(learningRate);
		WO = WO.add(As);
		
		//Training every attention head with error and entire weight matrix WO.
		Matrix ERROR = error.multiply(WO.transpose());
		int index = 0;
		for (int i = 0; i < heads.length; i++) {
			Attention0 head = heads[i];
			int n = head.A.columns();
			Matrix ERRORi = ERROR.extractVertical(index, n);
			head.learn(ERRORi, learningRate);
			index += n;
		}
	}
	
	
	/**
	 * Learning by input and output.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold  terminated threshold.
	 * @param maxIteration maximum iterations.
	 * @return learning error.
	 */
	public Matrix learn(Matrix inputY, Matrix outputA, Matrix inputX, double learningRate, double terminatedThreshold, int maxIteration) {
		if (outputA == null) return null;
		
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? NetworkAbstract.LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		maxIteration = maxIteration >= 0 ? maxIteration :  NetworkAbstract.LEARN_MAX_ITERATION_DEFAULT;

		Matrix error = null;
		int iteration = 0;
		while (maxIteration <= 0 || iteration < maxIteration) {
			double lr = NetworkAbstract.calcLearningRate(learningRate, iteration, false);
			Matrix A = evaluate(inputY, inputY);
			error = outputA.subtract(A);
			learn(error, lr);
			
			iteration ++;
			
			if (error == null || error.rows() == 0 || error.columns() == 0 || (iteration >= maxIteration && maxIteration == 1))
				break;
			else if (terminatedThreshold > 0) {
				double errorMean = Matrix.normMean(error);
				if (errorMean < terminatedThreshold) break; 
			}
			
		}

		return error;
	}

	
}



/**
 * This class represents simplest attention.
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
	 * Default value of key dimension which is dk.
	 */
	public static final int KEY_DIMENSION_DEFAULT = MODEL_DIMENSION_DEFAULT/8;

	
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
	public Attention0() {
		super();
	}

	
	/**
	 * Resetting attention.
	 */
	public void reset() {
		X = T1 = Y = WQ = WK = WV = A = null;
		M = null;
	}
	
	
	/**
	 * Initializing attention with sample size, model dimension, value dimension, zero value, other sample size.
	 * @param n sample size.
	 * @param dm model dimension.
	 * @param dk key dimension.
	 * @param dv value dimension.
	 * @param m other sample size.
	 * @param d other model dimension.
	 * @param zero zero value.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int n, int dm, int dk, int dv, NeuronValue zero, int m, int d) {
		if (n <= 0 || dm <= 0 || dv <= 0 || zero == null) return false;
		if (m <= 0 && d > 0) return false;
		if (m > 0 && m == n && (d <= 0 || d == dm)) {
			m = 0;
			d = 0;
		}
		if (m > 0 && m != n && d <= 0 ) d = dm;
		
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
	 * @param M masked matrix.
	 * @param Y Y input data.
	 * @param X X input data.
	 */
	protected void assignInputs(boolean[][] M, Matrix Y, Matrix X) {
		if (X != null) this.X = X;
		if (Y != null) this.Y = Y;
		if (M != null) this.M = M;
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
			return null;
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
	 */
	public void enterInputs(Matrix inputY, Matrix inputX) {
		Matrix.copy(inputY, Y);
		if (X != null && inputX != null) Matrix.copy(inputX, X);
	}

	
	/**
	 * Setting mask over range.
	 * @param row row index.
	 * @param column column index.
	 * @param range range.
	 * @param masked masked flag.
	 */
	public void setMask(int row, int column, int range, boolean masked) {
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
	public void setMask(int row, int column, boolean masked) {
		M[row][column] = masked;
	}

		
	/**
	 * Setting mask at specified row.
	 * @param row row index.
	 * @param masked masked flag.
	 */
	public void setMask(int row, boolean masked) {
		int n = n();
		if (row < 0 || row >= n) return;
		for (int j = 0; j < n; j++) M[row][j] = masked;
	}
	
	
	/**
	 * Setting mask over all mask matrix.
	 * @param masked masked flag.
	 */
	public void setMask(boolean masked) {
		int n = n();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) M[i][j] = masked;
		}
	}
	
	
	/**
	 * Evaluating attention given Y input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @return evaluated attention.
	 */
	public Matrix evaluate() {
		Matrix V = calcV();
		Matrix softmax = calcQKSoftmax();
		Matrix eval = softmax.multiply(V);
		
		int n = n();
		int dv = dv();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dv; j++) A.set(i, j, eval.get(i, j));
		}
		return A;
	}
	
	
	/**
	 * Learning attention by error.
	 * @param error specified error.
	 * @param learningRate learning rate.
	 */
	public void learn(Matrix error, double learningRate) {
		if (error == null) return;
		int n = n();
		if (error.rows() != n || error.columns() != dv()) return;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? NetworkAbstract.LEARN_RATE_DEFAULT : learningRate;

		Matrix Q = calcQ();
		Matrix K = calcK();
		Matrix V = calcV();
		Matrix errv = error.multiply(V.transpose());
		Matrix softmax = calcQKSoftmax();

		Matrix[] probs = new Matrix[n];
		for (int i = 0; i < n; i++) probs[i] = calcQKSoftmaxGradient(i);
		
		//Training weight query matrix and weight key matrix.
		Matrix dW = null;
		for (int i = 0; i < n; i++) {
			Matrix errvi = errv.getRow(i).multiply(probs[i]);
			Matrix d = Y().getRow(i).transpose().multiply(errvi);
			dW = dW != null ? dW.add(d) : d;
		}
		Matrix WQ = this.WQ.add(dW.multiply(K).multiply0(learningRate));
		Matrix.copy(WQ, this.WQ);
		Matrix WK = this.WK.add(dW.multiply(Q).multiply0(learningRate));
		Matrix.copy(WK, this.WK);
		
		//Training weight value matrix.
		Matrix dWV = Y().transpose().multiply(softmax.transpose()).multiply(errv);
		Matrix WV = this.WV.add(dWV.multiply0(learningRate));
		Matrix.copy(WV, this.WV);
		
		if (T1 == null && T2 == null) return;
		
		Matrix QKMean = calcK().multiply(WQ.transpose());
		QKMean = QKMean.add(calcQ().multiply(WK.transpose()));
		QKMean = QKMean.multiply0(0.5);

		//Training the first transposition matrix T1.
		if (T1 != null) {
			Matrix[] t1s = new Matrix[n];
			for (int i = 0; i < n; i++) {
				Matrix errvi = errv.getRow(i).multiply(probs[i]);
				t1s[i] = errvi.multiply(QKMean);
				if (T2 != null)
					t1s[i] = t1s[i].multiply(T2.transpose()).multiply(X().transpose());
				else
					t1s[i] = t1s[i].multiply(X().transpose());
				t1s[i] = t1s[i].multiply0(learningRate);
			}
			Matrix T1 = Matrix.concatH(t1s);
			Matrix.copy(T1, this.T1);
		}
		
		//Training the first transposition matrix T2.
		if (T2 != null) {
			Matrix dT2 = null;
			for (int i = 0; i < n; i++) {
				Matrix errvi = errv.getRow(i).multiply(probs[i]);
				Matrix d = T1 != null ? T1.getRow(i).transpose().multiply(errvi) : errvi;
				dT2 = dT2 != null ? dT2.add(d) : d;
			}
			dT2 = X().transpose().multiply(dT2).multiply(QKMean);
			Matrix T2 = this.T2.add(dT2.multiply0(learningRate));
			Matrix.copy(T2, this.T2);
		}
		
	}
	
	
}


