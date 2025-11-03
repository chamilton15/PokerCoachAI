# Poker Coach AI - Complete Usage Guide

## 🎯 Quick Start

###

 Option 1: Analyze an Existing Player File

If you already have a .phhs file with a specific player's hands:

```bash
python poker_coach.py abs_NLH_handhq_1_Hero_extracted.phhs Hero
```

### Option 2: Extract & Analyze Any Player from Any File

To analyze any player from the full dataset:

```bash
python analyze_any_player.py <path_to_phhs_file> <player_id> <optional_friendly_name>
```

**Example:**
```bash
python analyze_any_player.py /Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/"abs NLH handhq_1-OBFUSCATED.phhs" "l1bCLGuFqeFRwUfPsiDu/g" "Player1"
```

This will:
1. Extract all hands where the player participated
2. Save extracted hands to `Player1_extracted_hands.phhs`
3. Analyze the player's strategy
4. Generate coaching report saved to `Player1_analysis_report.txt`

---

## 📊 What You Get

### 1. Statistical Analysis
- **VPIP** (Voluntarily Put $ In Pot) - Shows how often you play hands
- **PFR** (Pre-Flop Raise) - Shows your aggression level
- **3-Bet %** - How often you re-raise
- **C-Bet %** - Continuation bet frequency
- **Aggression Factor** - Overall betting vs calling ratio
- **Position Stats** - How you play from different positions

### 2. Strategic Analysis
- Compares your play to Game Theory Optimal (GTO) baseline
- Identifies specific mistake patterns
- Shows frequency of each mistake type
- Provides examples from your actual hands

### 3. Personalized Recommendations
- Top 3-5 most important improvements
- Specific, actionable advice
- Examples from your hands showing mistakes
- Estimated impact on win rate (BB/100 hands)

---

## 📁 File Structure

After running analysis, you'll have:

```
PokerCoachAI/
├── poker_coach/              # Core analysis engine
│   ├── parser.py            # Parses .phhs files
│   ├── statistics.py        # Calculates poker stats
│   ├── hand_strength.py     # Evaluates hand strength
│   ├── strategy.py          # GTO baseline strategy
│   ├── analyzer.py          # Compares player vs baseline
│   └── feedback.py          # Generates recommendations
│
├── poker_coach.py            # Main analysis script
├── analyze_any_player.py     # Extract & analyze any player
├── debug_analysis.py         # Debug tool
│
├── abs_NLH_handhq_1_Hero_extracted.phhs  # Your data
├── Hero_analysis_report.txt              # Generated report
│
└── README.md                 # This file
```

---

## 🔧 Advanced Usage

### Analyze Multiple Players

Create a batch script to analyze multiple players:

```python
import os
from analyze_any_player import extract_and_analyze

data_file = "/path/to/poker_data.phhs"

players = [
    ("player_id_1", "Alice"),
    ("player_id_2", "Bob"),
    ("player_id_3", "Charlie"),
]

for player_id, name in players:
    print(f"\n\nAnalyzing {name}...")
    extract_and_analyze(data_file, player_id, name)
```

### Custom Strategy Baseline

Edit `poker_coach/strategy.py` to customize the GTO baseline:

```python
# Make opening ranges tighter/looser
OPENING_RANGES = {
    'UTG': 7,    # Default: 7 (tight)
    'MP': 6,
    'CO': 4,
    'BTN': 3,   # Default: 3 (loose)
}

# Adjust C-bet frequencies
CBET_FREQUENCY = {
    'heads_up': 0.65,   # Default: 65%
    'multiway': 0.45,   # Default: 45%
}
```

### Debug Player Actions

To see exactly what actions a player is taking:

```bash
python debug_analysis.py
```

This shows the first 10 hands with detailed action breakdowns.

---

## 📈 Understanding the Output

### Sample Report Section

```
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

### Interpreting Statistics

| Stat | Good Range | Meaning |
|------|------------|---------|
| VPIP | 20-30% | % of hands you voluntarily put money in |
| PFR | 15-22% | % of hands you raise pre-flop |
| 3-Bet | 6-10% | % of time you re-raise |
| C-Bet | 60-70% | % of time you bet after raising pre-flop |
| Aggression | 2.0-2.5 | (Bets + Raises) / Calls ratio |

**Too Low** = Playing too tight/passive
**Too High** = Playing too loose/aggressive
**In Range** = Playing solid fundamentals

---

## 🎓 Understanding the Strategy Baseline

The system uses a **simplified GTO (Game Theory Optimal)** approach based on:

### Position-Based Opening Ranges

Positions from worst to best:
1. **UTG** (Under the Gun) - Opens top 8% (QQ+, AK)
2. **MP** (Middle Position) - Opens top 15% (99+, AJ+)
3. **HJ** (Hijack) - Opens top 20% (88+, AT+)
4. **CO** (Cut-off) - Opens top 25% (77+, A9+, KJ+)
5. **BTN** (Button) - Opens top 40% (Any pair, any ace, suited cards)

### C-Bet Strategy

- **Heads-up**: C-bet 65% of flops
- **Multiway**: C-bet 45% of flops
- **In position**: C-bet +10% more
- **Dry boards**: C-bet more often
- **Wet boards**: C-bet less often

### Defense Strategy

- **Vs Raises**: Call/raise with top 50% of opening range
- **Vs 3-Bets**: Continue with premium hands (QQ+, AK)
- **Position matters**: Call wider in position

---

## 🚀 Working with Large Datasets

### Extract Hands for a Specific Player

The `analyze_any_player.py` script automatically:
1. Scans the entire .phhs file (can be 1000+ hands)
2. Finds all hands where your target player participated
3. Extracts only those hands
4. Saves to a new file
5. Runs full analysis

### Recommended Minimum

- **20+ hands**: Basic analysis possible
- **30+ hands**: Good pattern recognition
- **50+ hands**: Reliable recommendations
- **100+ hands**: Comprehensive analysis

### Multiple Sessions

To analyze across multiple sessions, combine extracted hands:

```bash
# Extract from session 1
python analyze_any_player.py session1.phhs "PlayerID" "Player"

# This creates Player_extracted_hands.phhs
# Manually combine multiple extracted files if needed
# Then analyze the combined file
python poker_coach.py Player_combined.phhs Player
```

---

## 🐛 Troubleshooting

### "No hands found for player"

**Problem**: Player ID doesn't match exactly

**Solution**: 
1. Look at the actual player IDs in the file
2. Use partial matching (the script matches substrings)
3. Check the players list in the debug output

```bash
# See what players are in the file
python -c "
from poker_coach.parser import parse_hand_history
hands = parse_hand_history('yourfile.phhs')
if hands:
    print('Players in first hand:')
    for p in hands[0].players:
        print(f'  {p}')
"
```

### "Only X hands found"

**Problem**: Player didn't play many hands in that file

**Solution**:
- Try a different .phhs file (some have 1000s of hands)
- Use multiple files and combine extracted hands
- Look for files where the player was more active

### Parser Errors

**Problem**: File format issues

**Solution**:
- Ensure file is valid .phhs format
- Check that file isn't corrupted
- Try a different file from the dataset

### Low Agreement Rate

**Problem**: Model says player makes lots of mistakes

**This might be accurate!** The model is identifying:
- Playing too tight/loose for position
- Not raising enough
- Calling too often
- Missing C-bet opportunities

Review the recommendations to understand what to improve.

---

## 📊 Example Analysis Results

### Hero's Analysis (56 hands)

```
VPIP: 50.0% (Too high! Optimal: 20-30%)
PFR: 0.0% (Too low! Never raising is too passive)
Aggression: 0.24 (Too low! Target: 2.0-2.5)

Top Issue: Over-calling with weak hands (24 mistakes)
→ Playing too many hands, not aggressive enough
→ Estimated impact: +21.4 BB/100 improvement possible
```

**Translation**: Hero is playing way too many hands (50% vs optimal 25%) and playing them passively (calling instead of raising). This is a huge leak!

---

## 🎯 Next Steps

### 1. Run Your First Analysis

```bash
python poker_coach.py abs_NLH_handhq_1_Hero_extracted.phhs Hero
```

### 2. Review the Report

Read through:
- Your statistics vs optimal ranges
- Top 3 recommendations
- Specific examples from your hands

### 3. Extract More Players

```bash
python analyze_any_player.py "/path/to/big/dataset.phhs" "any_player_id" "FriendlyName"
```

### 4. Compare Players

Analyze multiple players and compare their:
- VPIP/PFR
- Mistake patterns
- Win rates
- Positional awareness

### 5. Track Improvement

- Save reports with dates
- Re-analyze after studying
- Compare new vs old statistics

---

## 🔬 Technical Details

### How the Analysis Works

1. **Parser** reads .phhs file and extracts:
   - Player actions (fold, call, raise)
   - Positions (BTN, CO, UTG, etc.)
   - Pot sizes
   - Outcomes

2. **Statistics Calculator** computes:
   - Standard poker metrics (VPIP, PFR, etc.)
   - Position-based stats
   - Win rates

3. **Strategy Baseline** defines optimal play:
   - Based on GTO (Game Theory Optimal) principles
   - Position-dependent ranges
   - Situation-specific frequencies

4. **Analyzer** compares player vs baseline:
   - Identifies deviations from optimal
   - Categorizes mistake types
   - Calculates severity

5. **Feedback Generator** creates recommendations:
   - Groups mistakes into patterns
   - Generates specific advice
   - Estimates impact on win rate

### Limitations

This is a **simplified** GTO approximation:

- **~80% accuracy** vs professional solvers (~98%)
- Best for **learning and improvement**
- Not replacement for pro-level coaching
- **Pre-flop focused** (post-flop is limited)

### Future Enhancements

Potential additions:
- [ ] Full CFR+ integration for true GTO
- [ ] Deep post-flop analysis
- [ ] ML model for pattern learning
- [ ] Visualization of statistics
- [ ] Multi-session tracking
- [ ] Opponent profiling

---

## 📚 Learning Resources

### Poker Strategy Concepts

- **VPIP**: Measure of looseness (how many hands you play)
- **PFR**: Measure of aggression (how often you raise)
- **Position**: Later position = more information = more hands to play
- **GTO**: Game Theory Optimal - unexploitable strategy
- **C-bet**: Continuation bet after raising pre-flop
- **3-bet**: Re-raise (first raise is "bet", second is "3-bet")

### Recommended Reading

- *Modern Poker Theory* by Michael Acevedo
- *Applications of No-Limit Hold'em* by Matthew Janda
- *The Mathematics of Poker* by Bill Chen

### Online Resources

- GTO Poker Simplified (YouTube)
- Run It Once Training
- PokerStrategy.com

---

## 💡 Tips for Best Results

1. **Get enough hands**: 30+ for reliable analysis
2. **Review examples**: Look at specific hands where you made mistakes
3. **Focus on patterns**: Not individual hands, but recurring leaks
4. **Implement gradually**: Fix top 1-2 issues first
5. **Re-analyze**: Track improvement over time
6. **Study positions**: Learn why position matters
7. **Balance your ranges**: Mix strong and weak hands
8. **Be aggressive**: Raising > Calling in most spots

---

## 🤝 Contributing

Found a bug? Have suggestions?

- Adjust strategy baseline in `poker_coach/strategy.py`
- Modify feedback templates in `poker_coach/feedback.py`
- Customize statistics in `poker_coach/statistics.py`

---

## 📝 License

Educational/Research use for learning poker strategy and AI analysis.

---

## 🎓 Project Context

This Poker Coach AI was built to:
- Analyze poker hand histories automatically
- Provide personalized coaching feedback
- Help players identify and fix strategic leaks
- Use GTO principles for objective baseline
- Work dynamically with any player from any file

**Goal**: Give poker players an AI coach that learns from their actual play and provides specific, actionable advice for improvement.

---

## ⚡ Command Reference

```bash
# Basic analysis
python poker_coach.py <file.phhs> <player_name>

# Extract & analyze from big dataset
python analyze_any_player.py <file.phhs> <player_id> <friendly_name>

# Debug player actions
python debug_analysis.py

# Help
python poker_coach.py --help
python analyze_any_player.py --help
```

---

**Happy analyzing! 🎰🤖**


