// =============================================================================
// OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
// FILE:    src/RegimeClassifier/RegimeConfig.cs
// PURPOSE: Configuration for RegimeClassifierV2
// =============================================================================

namespace QTIP.RegimeClassifier;

public sealed record RegimeConfig
{
    public int    NStates          { get; init; } = 4;
    public int    MaxIterations    { get; init; } = 200;
    public double Tolerance        { get; init; } = 1e-5;
    public int    RandomSeed       { get; init; } = 42;
    public int    MinObservations  { get; init; } = 60;
    public int    ZScoreWindow     { get; init; } = 20;
    public int    VolLookback      { get; init; } = 14;
    public int    MomentumLookback { get; init; } = 10;
}
