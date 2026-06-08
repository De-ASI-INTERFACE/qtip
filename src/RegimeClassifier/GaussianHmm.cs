// =============================================================================
// OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
// FILE:    src/RegimeClassifier/GaussianHmm.cs
// PURPOSE: Gaussian HMM — Baum-Welch EM training + Viterbi decoding
//          Self-contained, zero external ML dependencies
// =============================================================================

using System;

namespace QTIP.RegimeClassifier;

/// <summary>
/// Multivariate Gaussian HMM with full covariance.
/// Trains via Baum-Welch EM. Decodes via Viterbi.
/// </summary>
public sealed class GaussianHmm
{
    // Model parameters
    public int      NStates  { get; private set; }
    public int      NDims    { get; private set; }
    public double[] Pi       { get; private set; } = Array.Empty<double>(); // initial state probs
    public double[,] A       { get; private set; } = new double[0, 0];      // transition matrix
    public double[,] Means   { get; private set; } = new double[0, 0];      // (K, D)
    public double[,,] Covs   { get; private set; } = new double[0, 0, 0];   // (K, D, D)
    public double   LogLikelihood { get; private set; }
    public bool     IsFitted { get; private set; }

    private readonly int    _maxIter;
    private readonly double _tol;
    private readonly Random _rng;

    public GaussianHmm(int nStates, int maxIter = 200, double tol = 1e-5, int seed = 42)
    {
        NStates  = nStates;
        _maxIter = maxIter;
        _tol     = tol;
        _rng     = new Random(seed);
    }

    // -------------------------------------------------------------------------
    // Fit — Baum-Welch EM
    // -------------------------------------------------------------------------
    public void Fit(double[,] obs)
    {
        int T = obs.GetLength(0);
        NDims = obs.GetLength(1);
        Initialise(obs);

        double prevLL = double.NegativeInfinity;
        for (int iter = 0; iter < _maxIter; iter++)
        {
            var (alpha, scale) = Forward(obs);
            var beta           = Backward(obs, scale);
            var (gamma, xi)    = ComputeGammaXi(obs, alpha, beta);
            UpdateParams(obs, gamma, xi);

            double ll = 0;
            for (int t = 0; t < T; t++) ll += Math.Log(scale[t] + 1e-300);
            LogLikelihood = ll;

            if (Math.Abs(ll - prevLL) < _tol) break;
            prevLL = ll;
        }
        IsFitted = true;
    }

    // -------------------------------------------------------------------------
    // Viterbi decode
    // -------------------------------------------------------------------------
    public int[] Predict(double[,] obs)
    {
        AssertFitted();
        int T = obs.GetLength(0);
        var delta = new double[T, NStates];
        var psi   = new int[T, NStates];

        // Initialise
        for (int k = 0; k < NStates; k++)
            delta[0, k] = Math.Log(Pi[k] + 1e-300) + LogGaussian(obs, 0, k);

        // Recursion
        for (int t = 1; t < T; t++)
            for (int j = 0; j < NStates; j++)
            {
                double best = double.NegativeInfinity; int bestK = 0;
                for (int k = 0; k < NStates; k++)
                {
                    double v = delta[t - 1, k] + Math.Log(A[k, j] + 1e-300);
                    if (v > best) { best = v; bestK = k; }
                }
                delta[t, j] = best + LogGaussian(obs, t, j);
                psi[t, j]   = bestK;
            }

        // Backtrack
        var path = new int[T];
        double maxVal = double.NegativeInfinity;
        for (int k = 0; k < NStates; k++)
            if (delta[T - 1, k] > maxVal) { maxVal = delta[T - 1, k]; path[T - 1] = k; }
        for (int t = T - 2; t >= 0; t--)
            path[t] = psi[t + 1, path[t + 1]];
        return path;
    }

    // -------------------------------------------------------------------------
    // Posterior state probabilities
    // -------------------------------------------------------------------------
    public double[,] PredictProba(double[,] obs)
    {
        AssertFitted();
        int T = obs.GetLength(0);
        var (alpha, scale) = Forward(obs);
        var beta           = Backward(obs, scale);
        var proba          = new double[T, NStates];
        for (int t = 0; t < T; t++)
        {
            double sum = 0;
            for (int k = 0; k < NStates; k++) sum += alpha[t, k] * beta[t, k];
            for (int k = 0; k < NStates; k++)
                proba[t, k] = sum > 0 ? alpha[t, k] * beta[t, k] / sum : 1.0 / NStates;
        }
        return proba;
    }

    // -------------------------------------------------------------------------
    // Forward algorithm (scaled)
    // -------------------------------------------------------------------------
    private (double[,] alpha, double[] scale) Forward(double[,] obs)
    {
        int T     = obs.GetLength(0);
        var alpha = new double[T, NStates];
        var scale = new double[T];

        for (int k = 0; k < NStates; k++)
            alpha[0, k] = Pi[k] * Gaussian(obs, 0, k);
        scale[0] = Normalise(alpha, 0);

        for (int t = 1; t < T; t++)
        {
            for (int j = 0; j < NStates; j++)
            {
                double sum = 0;
                for (int k = 0; k < NStates; k++) sum += alpha[t - 1, k] * A[k, j];
                alpha[t, j] = sum * Gaussian(obs, t, j);
            }
            scale[t] = Normalise(alpha, t);
        }
        return (alpha, scale);
    }

    // -------------------------------------------------------------------------
    // Backward algorithm (scaled)
    // -------------------------------------------------------------------------
    private double[,] Backward(double[,] obs, double[] scale)
    {
        int T    = obs.GetLength(0);
        var beta = new double[T, NStates];
        for (int k = 0; k < NStates; k++) beta[T - 1, k] = 1.0;

        for (int t = T - 2; t >= 0; t--)
            for (int k = 0; k < NStates; k++)
            {
                double sum = 0;
                for (int j = 0; j < NStates; j++)
                    sum += A[k, j] * Gaussian(obs, t + 1, j) * beta[t + 1, j];
                beta[t, k] = sum / (scale[t + 1] + 1e-300);
            }
        return beta;
    }

    // -------------------------------------------------------------------------
    // Gamma / Xi (E-step)
    // -------------------------------------------------------------------------
    private (double[,] gamma, double[,,] xi) ComputeGammaXi(
        double[,] obs, double[,] alpha, double[,] beta)
    {
        int T     = obs.GetLength(0);
        var gamma = new double[T, NStates];
        var xi    = new double[T - 1, NStates, NStates];

        for (int t = 0; t < T; t++)
        {
            double sum = 0;
            for (int k = 0; k < NStates; k++) sum += alpha[t, k] * beta[t, k];
            for (int k = 0; k < NStates; k++)
                gamma[t, k] = sum > 0 ? alpha[t, k] * beta[t, k] / sum : 1.0 / NStates;
        }

        for (int t = 0; t < T - 1; t++)
        {
            double sum = 0;
            for (int k = 0; k < NStates; k++)
                for (int j = 0; j < NStates; j++)
                    sum += alpha[t, k] * A[k, j] * Gaussian(obs, t + 1, j) * beta[t + 1, j];
            for (int k = 0; k < NStates; k++)
                for (int j = 0; j < NStates; j++)
                    xi[t, k, j] = sum > 0
                        ? alpha[t, k] * A[k, j] * Gaussian(obs, t + 1, j) * beta[t + 1, j] / sum
                        : 1.0 / (NStates * NStates);
        }
        return (gamma, xi);
    }

    // -------------------------------------------------------------------------
    // M-step parameter update
    // -------------------------------------------------------------------------
    private void UpdateParams(double[,] obs, double[,] gamma, double[,,] xi)
    {
        int T = obs.GetLength(0);

        // Pi
        for (int k = 0; k < NStates; k++) Pi[k] = gamma[0, k];
        NormaliseVector(Pi);

        // A
        for (int k = 0; k < NStates; k++)
        {
            double denom = 0;
            for (int t = 0; t < T - 1; t++) denom += gamma[t, k];
            for (int j = 0; j < NStates; j++)
            {
                double num = 0;
                for (int t = 0; t < T - 1; t++) num += xi[t, k, j];
                A[k, j] = denom > 0 ? num / denom : 1.0 / NStates;
            }
        }

        // Means
        for (int k = 0; k < NStates; k++)
        {
            double wSum = 0;
            for (int t = 0; t < T; t++) wSum += gamma[t, k];
            for (int d = 0; d < NDims; d++)
            {
                double num = 0;
                for (int t = 0; t < T; t++) num += gamma[t, k] * obs[t, d];
                Means[k, d] = wSum > 0 ? num / wSum : 0;
            }
        }

        // Covariances (full)
        for (int k = 0; k < NStates; k++)
        {
            double wSum = 0;
            for (int t = 0; t < T; t++) wSum += gamma[t, k];
            for (int d1 = 0; d1 < NDims; d1++)
                for (int d2 = 0; d2 < NDims; d2++)
                {
                    double num = 0;
                    for (int t = 0; t < T; t++)
                        num += gamma[t, k] * (obs[t, d1] - Means[k, d1]) * (obs[t, d2] - Means[k, d2]);
                    Covs[k, d1, d2] = wSum > 0 ? num / wSum : (d1 == d2 ? 1.0 : 0.0);
                }
            // Regularise diagonal
            for (int d = 0; d < NDims; d++) Covs[k, d, d] += 1e-6;
        }
    }

    // -------------------------------------------------------------------------
    // Gaussian density
    // -------------------------------------------------------------------------
    private double Gaussian(double[,] obs, int t, int k)
        => Math.Exp(LogGaussian(obs, t, k));

    private double LogGaussian(double[,] obs, int t, int k)
    {
        var x    = new double[NDims];
        var mu   = new double[NDims];
        var cov  = new double[NDims, NDims];
        for (int d = 0; d < NDims; d++) { x[d] = obs[t, d]; mu[d] = Means[k, d]; }
        for (int d1 = 0; d1 < NDims; d1++)
            for (int d2 = 0; d2 < NDims; d2++)
                cov[d1, d2] = Covs[k, d1, d2];

        var (invCov, logDet) = InvertSymmetric(cov);
        var diff = new double[NDims];
        for (int d = 0; d < NDims; d++) diff[d] = x[d] - mu[d];

        double mahal = 0;
        for (int d1 = 0; d1 < NDims; d1++)
            for (int d2 = 0; d2 < NDims; d2++)
                mahal += diff[d1] * invCov[d1, d2] * diff[d2];

        return -0.5 * (NDims * Math.Log(2 * Math.PI) + logDet + mahal);
    }

    // -------------------------------------------------------------------------
    // Matrix inversion (Cholesky for symmetric positive definite)
    // -------------------------------------------------------------------------
    private static (double[,] inv, double logDet) InvertSymmetric(double[,] m)
    {
        int n = m.GetLength(0);
        var L = new double[n, n];

        // Cholesky decomposition
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double sum = m[i, j];
                for (int k = 0; k < j; k++) sum -= L[i, k] * L[j, k];
                L[i, j] = j == i ? Math.Sqrt(Math.Max(sum, 1e-12)) : sum / L[j, j];
            }

        // Log determinant
        double logDet = 0;
        for (int i = 0; i < n; i++) logDet += 2 * Math.Log(L[i, i] + 1e-300);

        // Inverse via forward/back substitution
        var inv = new double[n, n];
        for (int col = 0; col < n; col++)
        {
            var e = new double[n];
            e[col] = 1;
            var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                double s = e[i];
                for (int k = 0; k < i; k++) s -= L[i, k] * y[k];
                y[i] = s / L[i, i];
            }
            var x = new double[n];
            for (int i = n - 1; i >= 0; i--)
            {
                double s = y[i];
                for (int k = i + 1; k < n; k++) s -= L[k, i] * x[k];
                x[i] = s / L[i, i];
            }
            for (int i = 0; i < n; i++) inv[i, col] = x[i];
        }
        return (inv, logDet);
    }

    // -------------------------------------------------------------------------
    // Initialisation — K-means style random seeding
    // -------------------------------------------------------------------------
    private void Initialise(double[,] obs)
    {
        int T = obs.GetLength(0);
        Pi    = new double[NStates];
        A     = new double[NStates, NStates];
        Means = new double[NStates, NDims];
        Covs  = new double[NStates, NDims, NDims];

        // Uniform init
        for (int k = 0; k < NStates; k++) Pi[k] = 1.0 / NStates;
        for (int k = 0; k < NStates; k++)
            for (int j = 0; j < NStates; j++)
                A[k, j] = 1.0 / NStates;

        // Random mean seeds from observations
        var indices = new HashSet<int>();
        while (indices.Count < NStates) indices.Add(_rng.Next(T));
        int idx = 0;
        foreach (int i in indices)
        {
            for (int d = 0; d < NDims; d++) Means[idx, d] = obs[i, d];
            idx++;
        }

        // Identity covariances
        for (int k = 0; k < NStates; k++)
            for (int d = 0; d < NDims; d++)
                Covs[k, d, d] = 1.0;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    private static double Normalise(double[,] mat, int row)
    {
        int cols = mat.GetLength(1);
        double sum = 0;
        for (int k = 0; k < cols; k++) sum += mat[row, k];
        if (sum < 1e-300) sum = 1e-300;
        for (int k = 0; k < cols; k++) mat[row, k] /= sum;
        return sum;
    }

    private static void NormaliseVector(double[] v)
    {
        double sum = 0;
        foreach (var x in v) sum += x;
        if (sum < 1e-300) sum = 1e-300;
        for (int i = 0; i < v.Length; i++) v[i] /= sum;
    }

    private void AssertFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Call Fit() before predicting.");
    }
}
