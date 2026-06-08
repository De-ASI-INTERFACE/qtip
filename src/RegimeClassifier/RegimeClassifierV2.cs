// =============================================================================
// OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
// FILE:    src/RegimeClassifier/RegimeClassifierV2.cs
// PURPOSE: Top-level regime classifier — wraps GaussianHMM, exposes typed API
// =============================================================================

using System;
using System.Collections.Generic;
using System.Linq;

namespace QTIP.RegimeClassifier;

public sealed class RegimeClassifierV2
{
    private readonly RegimeConfig _cfg;
    private GaussianHmm?          _model;
    private Dictionary<int, Regime> _labelMap = new();
    public bool IsFitted { get; private set; }

    public RegimeClassifierV2(RegimeConfig? cfg = null)
        => _cfg = cfg ?? new RegimeConfig();

    // -------------------------------------------------------------------------
    // Fit
    // -------------------------------------------------------------------------
    public RegimeClassifierV2 Fit(double[] prices)
    {
        var features = FeatureMatrix.Build(prices, _cfg);
        _model = new GaussianHmm(_cfg.NStates, _cfg.MaxIterations, _cfg.Tolerance, _cfg.RandomSeed);
        _model.Fit(features);
        _labelMap  = BuildLabelMap(features);
        IsFitted   = true;
        return this;
    }

    // -------------------------------------------------------------------------
    // Predict
    // -------------------------------------------------------------------------
    public List<Regime> Predict(double[] prices)
    {
        AssertFitted();
        var features = FeatureMatrix.Build(prices, _cfg);
        var hidden   = _model!.Predict(features);
        return hidden.Select(h => _labelMap[h]).ToList();
    }

    public double[,] PredictProba(double[] prices)
    {
        AssertFitted();
        var features = FeatureMatrix.Build(prices, _cfg);
        return _model!.PredictProba(features);
    }

    public Regime CurrentRegime(double[] prices)
        => Predict(prices)[^1];

    public Dictionary<string, double> CurrentProba(double[] prices)
    {
        AssertFitted();
        var proba  = PredictProba(prices);
        int last   = proba.GetLength(0) - 1;
        var result = new Dictionary<string, double>();
        for (int k = 0; k < _cfg.NStates; k++)
            result[_labelMap[k].ToString()] = Math.Round(proba[last, k], 6);
        return result;
    }

    public Dictionary<string, Dictionary<string, double>> TransitionMatrix()
    {
        AssertFitted();
        var tm = _model!.A;
        var labels = Enumerable.Range(0, _cfg.NStates)
                               .Select(i => _labelMap[i].ToString())
                               .ToList();
        var result = new Dictionary<string, Dictionary<string, double>>();
        for (int i = 0; i < _cfg.NStates; i++)
        {
            result[labels[i]] = new Dictionary<string, double>();
            for (int j = 0; j < _cfg.NStates; j++)
                result[labels[i]][labels[j]] = Math.Round(tm[i, j], 6);
        }
        return result;
    }

    public double LogLikelihood(double[] prices)
    {
        AssertFitted();
        var features = FeatureMatrix.Build(prices, _cfg);
        _model!.Fit(features); // recompute LL on given data
        return _model.LogLikelihood;
    }

    // -------------------------------------------------------------------------
    // Label map — rank hidden states by return + vol signature
    // -------------------------------------------------------------------------
    private Dictionary<int, Regime> BuildLabelMap(double[,] features)
    {
        var hidden = _model!.Predict(features);
        int T      = features.GetLength(0);

        var meanRet = new double[_cfg.NStates];
        var meanVol = new double[_cfg.NStates];
        var counts  = new int[_cfg.NStates];

        for (int t = 0; t < T; t++)
        {
            int s = hidden[t];
            meanRet[s] += features[t, 0];
            meanVol[s] += features[t, 1];
            counts[s]++;
        }
        for (int s = 0; s < _cfg.NStates; s++)
        {
            if (counts[s] == 0) continue;
            meanRet[s] /= counts[s];
            meanVol[s] /= counts[s];
        }

        var byRet = Enumerable.Range(0, _cfg.NStates).OrderBy(s => meanRet[s]).ToList();
        var byVol = Enumerable.Range(0, _cfg.NStates).OrderByDescending(s => meanVol[s]).ToList();

        var map = new Dictionary<int, Regime>();
        map[byRet[^1]] = Regime.BullTrend;
        map[byRet[0]]  = Regime.BearTrend;

        foreach (int s in byVol)
            if (!map.ContainsKey(s)) { map[s] = Regime.HighVolBreakout; break; }

        for (int s = 0; s < _cfg.NStates; s++)
            if (!map.ContainsKey(s)) map[s] = Regime.Ranging;

        return map;
    }

    private void AssertFitted()
    {
        if (!IsFitted || _model == null)
            throw new InvalidOperationException("Call Fit() before predicting.");
    }
}
