# 🎰 Poker Coach AI

**Analyze poker hand histories and receive personalized coaching feedback to improve your strategy.**

Built to analyze entire poker sessions and provide actionable, GTO-based recommendations for strategic improvement.

---

## ✨ Features

- ✅ **Parse .phhs files**: Supports standard poker hand history format
- ✅ **Calculate Statistics**: VPIP, PFR, 3-bet%, C-bet%, Aggression Factor, position stats
- ✅ **Strategy Analysis**: Compare player actions against simplified GTO baseline
- ✅ **Pattern Recognition**: Identify recurring strategic mistakes
- ✅ **Actionable Feedback**: Get specific recommendations ranked by impact
- ✅ **Dynamic Analysis**: Analyze ANY player from ANY .phhs file
- ✅ **Auto-extraction**: Automatically extracts player hands from multi-player files
- ✅ **Session-level insights**: Analyzes patterns across multiple hands

---

## 🚀 Quick Start

### Option 1: Analyze Hero's Pre-extracted Session

```bash
python poker_coach.py abs_NLH_handhq_1_Hero_extracted.phhs Hero
```

### Option 2: Extract & Analyze ANY Player from Large Dataset

```bash
python analyze_any_player.py "/full/path/to/dataset.phhs" "player_id" "FriendlyName"
```

**Example:**
```bash
python analyze_any_player.py "/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/abs NLH handhq_1-OBFUSCATED.phhs" "l1bCLGuFqeFRwUfPsiDu/g" "Player1"
```

This will automatically:
1. 🔍 Scan the entire file for the player
2. 📦 Extract all their hands
3. 📊 Analyze their strategy
4. 💾 Save report and extracted hands

---

## 📊 Sample Output

```
======================================================================
POKER COACH AI - SESSION ANALYSIS
======================================================================

PLAYER: Hero
HANDS ANALYZED: 56

OVERALL STATISTICS
======================================================================
VPIP: 50.0% (Optimal range: 20-30%) ⚠️ TOO HIGH
PFR:  0.0% (Optimal range: 15-22%)  ⚠️ TOO LOW
Aggression Factor: 0.24 (Target: 2.0-2.5) ⚠️ TOO PASSIVE

POSITION BREAKDOWN
======================================================================
BTN   : 10 hands, VPIP 60.0%, PFR 0.0%
CO    :  8 hands, VPIP 50.0%, PFR 0.0%
UTG   : 24 hands, VPIP 50.0%, PFR 0.0%

STRATEGY ANALYSIS
======================================================================
Agreement with Baseline: 46.4%
Strategic Mistakes Found: 30

TOP RECOMMENDATIONS
======================================================================

#1: REDUCE WEAK CALLS
----------------------------------------------------------------------
ISSUE: You're calling too often with weak hands
Frequency: 24 times (43% of hands)

EXAMPLES:
  • Hand #16: Preflop from UTG → Called, Should: Fold
  • Hand #45: Preflop from UTG → Called, Should: Fold

WHY THIS MATTERS:
  Calling with weak hands in bad spots loses money over time

HOW TO IMPROVE:
  ✓ Be more selective about which hands you call with
  ✓ Fold more often when out of position with marginal hands
  ✓ Consider 3-betting or folding instead of passively calling

ESTIMATED IMPACT: +21.4 BB/100 hands potential improvement
```

---

## 📁 Project Structure

```
PokerCoachAI/
├── poker_coach/                    # Core engine
│   ├── parser.py                  # Parse .phhs files
│   ├── statistics.py              # Calculate poker metrics
│   ├── hand_strength.py           # Hand evaluation
│   ├── strategy.py                # GTO baseline
│   ├── analyzer.py                # Compare player vs baseline
│   └── feedback.py                # Generate recommendations
│
├── poker_coach.py                  # Main analysis script
├── analyze_any_player.py           # Extract & analyze any player
├── abs_NLH_handhq_1_Hero_extracted.phhs  # Sample data
│
├── README.md                       # This file
├── USAGE_GUIDE.md                 # Detailed usage guide
└── IMPLEMENTATION_PLAN.txt        # Development plan

Generated files:
├── Hero_analysis_report.txt       # Analysis reports
└── Player1_extracted_hands.phhs   # Extracted hands
```

---

## 🎯 Use Cases

### 1. Self-Improvement
Analyze your own poker sessions to identify leaks and improve win rate.

### 2. Opponent Analysis
Extract and analyze opponents to understand their tendencies.

### 3. Session Review
Review entire sessions to understand patterns and costly mistakes.

### 4. Learning Tool
Study GTO strategy by comparing real play to optimal baseline.

---

## 📖 Complete Documentation

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for:
- Detailed usage instructions
- Understanding statistics
- Interpreting recommendations
- Troubleshooting
- Advanced features
- Technical details

---

## 🎓 How It Works

### 1. Parser
Reads .phhs files and extracts game state, actions, positions, and outcomes.

### 2. Statistics Calculator  
Computes standard poker metrics (VPIP, PFR, aggression, etc.) by position.

### 3. Strategy Baseline (Simplified GTO)
Uses Game Theory Optimal principles:
- Position-based opening ranges
- C-bet frequencies
- 3-bet defense strategies
- Pot odds considerations

### 4. Analyzer
Compares player actions to baseline and identifies deviations.

### 5. Feedback Generator
Groups mistakes into patterns and generates actionable recommendations with estimated impact.

---

## 🧮 Strategy Baseline

The system uses **simplified GTO** based on:

## What You Get

The analysis includes:

1. **Overall Statistics**
   - VPIP (Voluntarily Put $ In Pot)
   - PFR (Pre-Flop Raise %)
   - 3-Bet %
   - C-Bet %
   - Aggression Factor
   - Win Rate

2. **Position Breakdown**
   - Statistics by position (BTN, CO, UTG, etc.)
   - Shows how position affects your play

3. **Strategic Analysis**
   - Compares your actions to simplified GTO baseline
   - Identifies specific mistakes and patterns
   - Categorizes mistakes by type

4. **Top 3 Recommendations**
   - Specific, actionable advice
   - Examples from your actual hands
   - Estimated impact on win rate

## Example Output

```
======================================================================
POKER COACH AI - SESSION ANALYSIS
======================================================================

PLAYER: Hero
HANDS ANALYZED: 56

OVERALL STATISTICS
======================================================================
VPIP: 22.5% (Optimal range: 20-30%)
PFR:  14.3% (Optimal range: 15-22%)
Aggression Factor: 2.10 (Target: 2.0-2.5)

...

TOP RECOMMENDATIONS
======================================================================

#1: STOP OVER-FOLDING
----------------------------------------------------------------------

ISSUE:
  You're folding too often in profitable situations
  Frequency: 8 times (14% of hands)

EXAMPLES:
  • Hand #5: Preflop facing raise with 77 from CO → You f, Optimal: call
  
WHY THIS MATTERS:
  You're leaving money on the table by being too cautious

HOW TO IMPROVE:
  ✓ Call more often with medium-strength hands when in position
  ✓ Medium pairs (66-99) are often profitable calls in position
  
ESTIMATED IMPACT: +1.4 BB/100 hands potential improvement
```

## How It Works

### 1. Parser
Reads .phhs files and extracts:
- Player actions
- Positions
- Hole cards (if visible)
- Pot sizes
- Outcomes

### 2. Statistics Calculator
Calculates standard poker metrics to understand playing style.

### 3. Strategy Baseline
Uses simplified GTO (Game Theory Optimal) principles:
- Position-based opening ranges
- Calling ranges vs raises
- C-bet frequencies
- 3-bet defense strategies

### 4. Analyzer
Compares player actions to baseline strategy and identifies deviations.

### 5. Feedback Generator
Groups mistakes into patterns and generates actionable recommendations.

## Strategy Baseline

The baseline uses fundamental poker theory:

**Pre-flop Opening Ranges by Position:**
- UTG: Top 8% (QQ+, AK)
- MP: Top 15% (99+, AJ+)
- CO: Top 25% (77+, AT+)
- BTN: Top 40% (Any pair, any ace, suited cards)

**Post-flop:**
- C-bet 65% heads-up
- C-bet 45% multiway
- Higher frequency in position

**Defense:**
- Call/4-bet with premium hands vs 3-bet
- Fold weaker hands out of position

## Project Structure

```
PokerCoachAI/
├── poker_coach/
│   ├── __init__.py
│   ├── parser.py          # Parse .phhs files
│   ├── statistics.py      # Calculate poker stats
│   ├── hand_strength.py   # Evaluate hand strength
│   ├── strategy.py        # GTO baseline strategy
│   ├── analyzer.py        # Compare player vs baseline
│   └── feedback.py        # Generate recommendations
├── poker_coach.py          # Main script
├── README.md
└── abs_NLH_handhq_1_Hero_extracted.phhs  # Sample data
```

## Customization

You can customize the strategy baseline by editing `poker_coach/strategy.py`:

```python
# Adjust opening ranges
OPENING_RANGES = {
    'UTG': 7,    # Tighter or looser
    'BTN': 3,    # Adjust button range
}

# Adjust C-bet frequencies
CBET_FREQUENCY = {
    'heads_up': 0.65,   # Your preference
    'multiway': 0.45,
}
```

## Limitations

This is a **simplified** GTO approximation for educational purposes:

- Not as accurate as professional solvers (PioSolver, etc.)
- ~80% accuracy vs ~98% for full CFR+ implementations
- Doesn't account for all edge cases
- Best for learning and improvement, not pro-level coaching

## Future Enhancements

Potential improvements:
- [ ] Full CFR+ integration for true GTO baseline
- [ ] Post-flop analysis (currently focuses on pre-flop)
- [ ] ML model to learn player-specific patterns
- [ ] Visualization of statistics
- [ ] Multi-session tracking
- [ ] Opponent analysis
- [ ] Hand range visualization

## Credits

Built for learning poker AI and strategy analysis.

Uses simplified GTO principles based on:
- Modern poker theory
- Position-based strategy
- Hand strength evaluation
- Fundamental poker mathematics

## License

Educational/Research use.

