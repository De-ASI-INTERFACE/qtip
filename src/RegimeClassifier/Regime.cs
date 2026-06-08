// =============================================================================
// OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
// FILE:    src/RegimeClassifier/Regime.cs
// PURPOSE: Market regime state enum
// =============================================================================

namespace QTIP.RegimeClassifier;

public enum Regime
{
    BullTrend,
    BearTrend,
    Ranging,
    HighVolBreakout
}
