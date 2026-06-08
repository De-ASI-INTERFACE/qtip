// =============================================================================
// OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
// FILE:    tests/csharp/RegimeClassifierTests.cs
// PURPOSE: Full hardened xUnit test suite for C# RegimeClassifierV2
// =============================================================================

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using QTIP.RegimeClassifier;

namespace QTIP.RegimeClassifier.Tests;

public class RegimeClassifierTests
{
    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    private static double[] GbmPrices(int n, double drift = 0.0001, double vol = 0.02, int seed = 42)
    {
        var rng = new Random(seed);
        var prices = new double[n];
        prices[0] = 100.0;
        for (int i = 1; i < n; i++)
        {
            double z = NormalSample(rng);
            prices[i] = prices[i - 1] * Math.Exp(drift + vol * z);
        }
        return prices;
    }

    private static double[] SplicedPrices(int perSegment = 150)
    {
        var bull  = GbmPrices(perSegment, drift: +0.003, vol: 0.010, seed: 1);
        var bear  = GbmPrices(perSegment, drift: -0.003, vol: 0.010, seed: 2);
        var range = GbmPrices(perSegment, drift:  0.000, vol: 0.003, seed: 3);
        var hvol  = GbmPrices(perSegment, drift: +0.001, vol: 0.040, seed: 4);
        var all   = new List<double[]> { bull, bear, range, hvol };
        var result = new List<double>();
        result.AddRange(bull);
        double lastPrice = bull[^1];
        foreach (var seg in new[] { bear, range, hvol })
        {
            double ratio = lastPrice / seg[0];
            result.AddRange(seg.Select(p => p * ratio));
            lastPrice = result[^1];
        }
        return result.ToArray();
    }

    private static double NormalSample(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    // -------------------------------------------------------------------------
    // Feature matrix tests
    // -------------------------------------------------------------------------
    [Fact]
    public void FeatureMatrix_Shape_IsCorrect()
    {
        var prices = GbmPrices(300);
        var cfg    = new RegimeConfig();
        var mat    = FeatureMatrix.Build(prices, cfg);
        Assert.Equal(4, mat.GetLength(1));
        Assert.True(mat.GetLength(0) > 0);
    }

    [Fact]
    public void FeatureMatrix_TooFewBars_Throws()
    {
        var prices = GbmPrices(10);
        var cfg    = new RegimeConfig { MinObservations = 60 };
        Assert.Throws<ArgumentException>(() => FeatureMatrix.Build(prices, cfg));
    }

    [Fact]
    public void FeatureMatrix_NoNaN()
    {
        var prices = GbmPrices(300);
        var cfg    = new RegimeConfig();
        var mat    = FeatureMatrix.Build(prices, cfg);
        int rows   = mat.GetLength(0);
        int cols   = mat.GetLength(1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.False(double.IsNaN(mat[r, c]));
    }

    [Fact]
    public void FeatureMatrix_ZScore_NearZeroMean()
    {
        var prices = GbmPrices(300);
        var cfg    = new RegimeConfig();
        var mat    = FeatureMatrix.Build(prices, cfg);
        int rows   = mat.GetLength(0);
        int cols   = mat.GetLength(1);
        for (int c = 0; c < cols; c++)
        {
            double sum = 0;
            for (int r = 0; r < rows; r++) sum += mat[r, c];
            Assert.True(Math.Abs(sum / rows) < 1e-9, $"Column {c} mean not zero");
        }
    }

    // -------------------------------------------------------------------------
    // Fit tests
    // -------------------------------------------------------------------------
    [Fact]
    public void Fit_ReturnsSelf()
    {
        var prices = GbmPrices(300);
        var clf    = new RegimeClassifierV2();
        var result = clf.Fit(prices);
        Assert.Same(clf, result);
    }

    [Fact]
    public void Fit_SetsIsFitted()
    {
        var prices = GbmPrices(300);
        var clf    = new RegimeClassifierV2();
        Assert.False(clf.IsFitted);
        clf.Fit(prices);
        Assert.True(clf.IsFitted);
    }

    [Fact]
    public void Fit_LabelMap_AllFourRegimes_Assigned()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var regimes = clf.Predict(prices).Distinct().ToList();
        Assert.True(regimes.Count >= 3, $"Only {regimes.Count} distinct regimes found");
    }

    // -------------------------------------------------------------------------
    // Predict tests
    // -------------------------------------------------------------------------
    [Fact]
    public void Predict_BeforeFit_Throws()
    {
        var clf = new RegimeClassifierV2();
        Assert.Throws<InvalidOperationException>(() => clf.Predict(GbmPrices(300)));
    }

    [Fact]
    public void Predict_ReturnsListOfRegimes()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var preds = clf.Predict(prices);
        Assert.NotEmpty(preds);
        Assert.All(preds, r => Assert.IsType<Regime>(r));
    }

    [Fact]
    public void Predict_Length_LessThanInputPrices()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var preds = clf.Predict(prices);
        Assert.True(preds.Count < prices.Length);
    }

    [Fact]
    public void CurrentRegime_IsRegimeType()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var current = clf.CurrentRegime(prices);
        Assert.IsType<Regime>(current);
    }

    // -------------------------------------------------------------------------
    // Probability tests
    // -------------------------------------------------------------------------
    [Fact]
    public void PredictProba_RowsSumToOne()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var proba = clf.PredictProba(prices);
        int rows  = proba.GetLength(0);
        int cols  = proba.GetLength(1);
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += proba[r, c];
            Assert.True(Math.Abs(sum - 1.0) < 1e-5, $"Row {r} sums to {sum}");
        }
    }

    [Fact]
    public void PredictProba_NonNegative()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var proba = clf.PredictProba(prices);
        int rows  = proba.GetLength(0);
        int cols  = proba.GetLength(1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.True(proba[r, c] >= 0);
    }

    [Fact]
    public void CurrentProba_SumsToOne()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var cp    = clf.CurrentProba(prices);
        double sum = cp.Values.Sum();
        Assert.True(Math.Abs(sum - 1.0) < 1e-5);
    }

    [Fact]
    public void CurrentProba_HasAllRegimeKeys()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var cp      = clf.CurrentProba(prices);
        var expected = Enum.GetNames<Regime>().ToHashSet();
        Assert.Equal(expected, cp.Keys.ToHashSet());
    }

    // -------------------------------------------------------------------------
    // Transition matrix tests
    // -------------------------------------------------------------------------
    [Fact]
    public void TransitionMatrix_RowsSumToOne()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var tm = clf.TransitionMatrix();
        foreach (var row in tm.Values)
        {
            double sum = row.Values.Sum();
            Assert.True(Math.Abs(sum - 1.0) < 1e-4, $"Row sums to {sum}");
        }
    }

    [Fact]
    public void TransitionMatrix_NonNegative()
    {
        var prices = SplicedPrices();
        var clf    = new RegimeClassifierV2();
        clf.Fit(prices);
        var tm = clf.TransitionMatrix();
        foreach (var row in tm.Values)
            Assert.All(row.Values, v => Assert.True(v >= 0));
    }

    [Fact]
    public void TransitionMatrix_BeforeFit_Throws()
    {
        var clf = new RegimeClassifierV2();
        Assert.Throws<InvalidOperationException>(() => clf.TransitionMatrix());
    }
}
