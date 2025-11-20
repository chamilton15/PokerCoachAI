# 🎰 Poker Coach AI

**Analyze poker hand histories and receive personalized coaching feedback powered by an LSTM neural network trained on winning players.**

Poker Coach AI uses deep learning to learn optimal poker strategies from hand histories of profitable players. The model was trained on 112,431 decision points from the top 20% most profitable players, achieving 63.68% test accuracy in predicting optimal actions (fold, call, raise). The system provides actionable feedback through a modern web interface, helping players identify strategic mistakes and improve their game.

---

## ✨ Features

- 🤖 **LSTM Neural Network**: Trained on 112,431 decision points from top 20% profitable players
- 🎯 **Action Prediction**: Predicts optimal actions (fold, call, raise) with 63.68% test accuracy
- 📊 **Session Analysis**: Analyze entire poker sessions with detailed metrics
- 💡 **AI-Powered Coaching**: GPT-4o-mini integration for human-readable strategy advice
- 🌐 **Web Interface**: Modern React web application for easy analysis
- 📁 **Preloaded Samples**: Test with preloaded sample files from different stake levels
- 🔍 **Player Dropdown**: Automatic player name extraction from session files
- ✅ **CLI Support**: Command-line tools for batch processing and automation

---

## 🚀 Quick Start

### 🌐 Web Application (Recommended)

The easiest way to use Poker Coach AI is through the web interface:

#### 1. Start the Backend

```bash
cd web_app/backend
pip install -r requirements.txt
python app.py
```

The backend will run on `http://localhost:5001`

**Note:** For AI coaching summaries, create a `.env` file in `ml_pipeline/` with:
```
OPENAI_API_KEY=your_api_key_here
```

#### 2. Start the Frontend

In a new terminal:

```bash
cd web_app/frontend
npm install
npm start
```

The frontend will open at `http://localhost:3000`

#### 3. Analyze a Session

1. Select a preloaded sample file or upload your own `.phh/.phhs` file
2. Choose a player from the dropdown (auto-populated from the file)
3. Click "Analyze Session"
4. View your results including:
   - Agreement rate with optimal play
   - Total hands and decisions analyzed
   - High-confidence mistakes
   - AI-generated coaching summary

### 💻 Command Line Interface

For batch processing or automation:

#### Option 1: Analyze a Player Session

```bash
python poker_coach.py <file.phhs> <player_name>
```

#### Option 2: Extract & Analyze from Large Dataset

```bash
python analyze_any_player.py "/path/to/dataset.phhs" "player_id" "FriendlyName"
```

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
├── web_app/                        # Web application
│   ├── backend/                    # Flask API server
│   │   ├── app.py                 # Main API endpoints
│   │   └── requirements.txt       # Python dependencies
│   └── frontend/                   # React frontend
│       ├── src/
│       │   ├── App.js             # Main React component
│       │   └── App.css            # Styling
│       └── package.json            # Node dependencies
│
├── ml_pipeline/                     # Machine learning pipeline
│   ├── extract_training_data.py   # Extract training examples
│   ├── prepare_training_dataset.py # Data preprocessing
│   ├── train_model.py             # Model training script
│   ├── inference.py               # Model inference for predictions
│   ├── tune_hyperparameters.py    # Hyperparameter tuning
│   └── models/                     # Trained models
│       ├── poker_coach_model.pt   # Trained LSTM model
│       └── training_history.json  # Training metrics
│
├── poker_coach/                    # Core poker analysis engine
│   ├── parser.py                  # Parse .phhs files
│   ├── statistics.py              # Calculate poker metrics
│   ├── hand_strength.py           # Hand evaluation
│   └── feedback.py                # Generate recommendations
│
├── poker_coach.py                  # CLI: Main analysis script
├── analyze_any_player.py           # CLI: Extract & analyze any player
│
└── README.md                       # This file
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

### 1. Data Collection
The model was trained on 112,431 decision points extracted from the top 20% most profitable players in a dataset of over 21 million poker hands.

### 2. Model Architecture
- **LSTM Neural Network**: 2-layer LSTM with 64 hidden units to capture sequential action patterns
- **Input Features**: 
  - Current game state (11 dimensions): hand strength, position, pot size, stack size, etc.
  - Action history (20 actions × 5 features): sequence of previous actions in the hand
- **Output**: Predicts optimal action (fold, call, or raise) with 63.68% test accuracy

### 3. Analysis Pipeline
1. **Parser**: Reads .phhs files and extracts game state, actions, positions, and outcomes
2. **Inference**: Model predicts optimal action for each decision point
3. **Comparison**: Compares player's actual actions to model predictions
4. **Statistics**: Calculates agreement rate, identifies mistakes, and highlights patterns
5. **AI Coaching**: GPT-4o-mini generates human-readable strategy advice

### 4. Model Performance
- **Overall Accuracy**: 63.68% test accuracy (vs 33.3% random baseline, 52.9% majority class)
- **Per-Class Accuracy**:
  - Fold: 82.08%
  - Call: 46.44%
  - Raise: 78.35%

---

## 📊 What You Get

The analysis includes:

1. **Agreement Rate**
   - Percentage of decisions that match the model's optimal predictions
   - Higher agreement suggests better strategic play

2. **Decision Breakdown**
   - Total hands and decisions analyzed
   - High-confidence mistakes (where model was very confident)
   - Sample decision comparisons showing your action vs. optimal action

3. **AI Coaching Summary**
   - 3-sentence high-level summary generated by GPT-4o-mini
   - Actionable advice on how to improve your play
   - Focuses on specific strategic adjustments

4. **Detailed Metrics**
   - Per-decision analysis with confidence scores
   - Pattern recognition for recurring mistakes

## 🔧 Setup & Requirements

### Prerequisites

- Python 3.8+
- Node.js 14+ and npm
- Trained model file: `ml_pipeline/models/poker_coach_model.pt`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chamilton15/PokerCoachAI.git
   cd PokerCoachAI
   ```

2. **Install backend dependencies**
   ```bash
   cd web_app/backend
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd web_app/frontend
   npm install
   ```

4. **Optional: Set up OpenAI API for AI coaching**
   ```bash
   # Create .env file in ml_pipeline/
   echo "OPENAI_API_KEY=your_key_here" > ml_pipeline/.env
   ```

### Running the Application

See the [Web Application README](web_app/README.md) for detailed setup instructions.

## 🧪 Training Your Own Model

To train the model from scratch:

1. **Extract training data**
   ```bash
   cd ml_pipeline
   python extract_training_data.py
   ```

2. **Prepare dataset**
   ```bash
   python prepare_training_dataset.py
   ```

3. **Train model**
   ```bash
   python train_model.py
   ```

4. **Tune hyperparameters (optional)**
   ```bash
   python tune_hyperparameters.py
   ```

See [ml_pipeline/README.md](ml_pipeline/README.md) for detailed training instructions.

## 📈 Model Performance

The model achieves:
- **63.68% test accuracy** (vs 33.3% random baseline, 52.9% majority class)
- **82.08% fold accuracy**
- **46.44% call accuracy**
- **78.35% raise accuracy**

## ⚠️ Limitations

- Fixed 20-action history truncation (may lose context in very long sequences)
- No explicit post-flop hand strength evaluation (only preflop strength is evaluated)
- Trained on specific dataset; performance may vary on different game formats
- Model struggles with "call" decisions (46.44% accuracy) compared to fold/raise

## 🚀 Future Enhancements

Potential improvements:
- [ ] Improve call decision accuracy (currently 46.44%)
- [ ] Post-flop hand strength evaluation
- [ ] Longer action sequence support (beyond 20 actions)
- [ ] Multi-session tracking and trend analysis
- [ ] Hand range visualization
- [ ] Real-time analysis during live play
- [ ] Export analysis reports to PDF

## 📚 Documentation

- [Web Application Guide](web_app/README.md) - Detailed web app setup and usage
- [ML Pipeline Guide](ml_pipeline/README.md) - Model training and development
- [Usage Guide](USAGE_GUIDE.md) - CLI usage and advanced features

## 🤝 Contributing

This project is built for educational and research purposes. Contributions welcome!

## 📄 License

Educational/Research use.

## 🙏 Acknowledgments

- Dataset: [Zenodo Poker Hand Histories](https://zenodo.org/records/13997158)
- Built with PyTorch, React, and Flask

