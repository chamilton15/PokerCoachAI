# 🚀 Poker Coach AI - Quickstart

Get started analyzing poker hands in 2 minutes!

---

## Option 1: Interactive Menu (Easiest)

```bash
python quickstart.py
```

This gives you a menu to:
1. Analyze Hero's session
2. Extract & analyze any player
3. Browse available datasets

---

## Option 2: Command Line

### Analyze Hero (Pre-extracted)

```bash
python poker_coach.py abs_NLH_handhq_1_Hero_extracted.phhs Hero
```

**Output:** `Hero_analysis_report.txt`

### Extract & Analyze Any Player

```bash
python analyze_any_player.py "/path/to/data.phhs" "player_id" "FriendlyName"
```

**Example:**
```bash
python analyze_any_player.py \
  "/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/abs NLH handhq_1-OBFUSCATED.phhs" \
  "l1bCLGuFqeFRwUfPsiDu/g" \
  "Player1"
```

**Output:**
- `Player1_analysis_report.txt` - Full coaching report
- `Player1_extracted_hands.phhs` - Extracted hands for that player

---

## What You Get

### 📊 Statistics
- VPIP, PFR, Aggression Factor
- 3-Bet %, C-Bet %, Fold to 3-bet %
- Stats by position (BTN, CO, UTG, etc.)
- Win rate (BB/hand)

### 🎯 Strategic Analysis
- Comparison to GTO baseline
- Mistake detection and categorization
- Pattern recognition across session
- Agreement rate with optimal play

### 💡 Recommendations
- Top 3-5 biggest leaks ranked by impact
- Specific examples from your hands
- Actionable advice for improvement
- Estimated BB/100 improvement potential

---

## Example Results

```
PLAYER: Hero
HANDS: 56

VPIP: 50.0% ⚠️ (Optimal: 20-30%)
PFR: 0.0% ⚠️ (Optimal: 15-22%)
Aggression: 0.24 ⚠️ (Optimal: 2.0-2.5)

TOP ISSUE: Over-calling with weak hands (24 times)
IMPACT: +21.4 BB/100 potential improvement

RECOMMENDATION:
✓ Be more selective with hand selection
✓ Raise more, call less
✓ Play position-appropriate ranges
```

---

## Finding Player IDs

Not sure what player IDs are in a file?

```bash
python -c "
from poker_coach.parser import parse_hand_history
hands = parse_hand_history('yourfile.phhs')
if hands:
    print('Players in first hand:')
    for p in hands[0].players:
        print(f'  {p}')
"
```

Or just use partial matching - the script will find players whose ID contains your search string!

---

## Tips for Best Results

1. **Minimum 30 hands** for reliable analysis
2. **50+ hands** recommended for comprehensive insights
3. Use `analyze_any_player.py` to extract from large files
4. Review specific hand examples in the report
5. Focus on top 1-2 recommendations first
6. Re-analyze after making adjustments

---

## Troubleshooting

### "File not found"
- Use full absolute paths
- Check file exists with `ls /path/to/file.phhs`

### "No hands found for player"
- Check player ID is correct
- Try partial matching (just part of the ID)
- Use the player finder command above

### "Only X hands found"
- Player didn't play many hands in that file
- Try a different file with more hands
- Combine multiple sessions if needed

---

## Next Steps

1. ✅ Run analysis on Hero's data
2. ✅ Review the report and recommendations
3. ✅ Extract and analyze other players
4. ✅ Compare different playing styles
5. ✅ Track improvement over time

For detailed documentation, see:
- [README.md](README.md) - Overview and features
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Complete usage guide

---

**Happy analyzing! 🎰🤖**


