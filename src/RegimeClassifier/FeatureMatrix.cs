// =============================================================================
// OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
// FILE:    src/RegimeClassifier/FeatureMatrix.cs
// PURPOSE: Feature engineering — log returns, realised vol, momentum, trend strength
// =============================================================================

using System;

namespace QTIP.RegimeClassifier;

public static class FeatureMatrix
{
    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------
    /// <summary>
    /// Builds a z-scored (T, 4) feature matrix from price array.
    /// Features: log return | realised vol | momentum | trend strength
    /// </summary>
    public static double[,] Build(double[] prices, RegimeConfig cfg)
    {
        if (prices.Length < cfg.MinObservations)
            throw new ArgumentException(
                $"Need >= {cfg.MinObservations} price bars, got {prices.Length}");

        var returns       = LogReturns(prices);
        var vol           = RealisedVol(returns, cfg.VolLookback);
        var momentum      = Momentum(prices, cfg.MomentumLookback);
        var trendStrength = TrendStrength(returns, cfg.ZScoreWindow);

        // Align all series to returns length (T-1), skip NaN warm-up
        int warmup = Math.Max(cfg.VolLookback, Math.Max(cfg.MomentumLookback, cfg.ZScoreWindow));
        int n      = returns.Length - warmup;

        if (n <= 0)
            throw new InvalidOperationException(
                "Not enough data after warm-up period.");

        var raw = new double[n, 4];
        for (int i = 0; i < n; i++)
        {
            int src = i + warmup;
            raw[i, 0] = returns[src];
            raw[i, 1] = vol[src];
            raw[i, 2] = momentum[src + 1]; // momentum is on prices (T), shift by 1
            raw[i, 3] = trendStrength[src];
        }

        return ZScore(raw);
    }

    // -------------------------------------------------------------------------
    // Feature helpers
    // -------------------------------------------------------------------------
    internal static double[] LogReturns(double[] prices)
    {
        var ret = new double[prices.Length - 1];
        for (int i = 0; i < ret.Length; i++)
            ret[i] = Math.Log((prices[i + 1] + 1e-12) / (prices[i] + 1e-12));
        return ret;
    }

    internal static double[] RealisedVol(double[] returns, int window)
    {
        var out_ = new double[returns.Length];
        for (int i = window - 1; i < returns.Length; i++)
        {
            double sum = 0, sumSq = 0;
            for (int j = i - window + 1; j <= i; j++) { sum += returns[j]; sumSq += returns[j] * returns[j]; }
            double mean = sum / window;
            out_[i] = Math.Sqrt((sumSq / window) - (mean * mean));
        }
        return out_;
    }

    internal static double[] Momentum(double[] prices, int window)
    {
        var out_ = new double[prices.Length];
        for (int i = window; i < prices.Length; i++)
            out_[i] = (prices[i] - prices[i - window]) / (prices[i - window] + 1e-12);
        return out_;
    }

    internal static double[] TrendStrength(double[] returns, int window)
    {
        var out_ = new double[returns.Length];
        for (int i = window - 1; i < returns.Length; i++)
        {
            double sum = 0;
            for (int j = i - window + 1; j <= i; j++) sum += returns[j];
            out_[i] = sum;
        }
        return out_;
    }

    internal static double[,] ZScore(double[,] mat)
    {
        int rows = mat.GetLength(0);
        int cols = mat.GetLength(1);
        var result = new double[rows, cols];

        for (int c = 0; c < cols; c++)
        {
            double sum = 0;
            for (int r = 0; r < rows; r++) sum += mat[r, c];
            double mean = sum / rows;

            double varSum = 0;
            for (int r = 0; r < rows; r++) varSum += Math.Pow(mat[r, c] - mean, 2);
            double std = Math.Sqrt(varSum / (rows - 1));
            if (std < 1e-12) std = 1e-12;

            for (int r = 0; r < rows; r++)
                result[r, c] = (mat[r, c] - mean) / std;
        }
        return result;
    }
}
