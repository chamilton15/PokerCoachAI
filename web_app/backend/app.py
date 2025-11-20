#!/usr/bin/env python3
"""
Flask API for Poker Coach AI Model Inference
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
from pathlib import Path
import json

# Optional: OpenAI for AI coaching summaries
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Optional: load environment variables from .env
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml_pipeline"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Import model and inference functions
# Add parent directories to path for imports
backend_dir = Path(__file__).parent
project_root = backend_dir.parent.parent
sys.path.insert(0, str(project_root))

# Load .env from common locations if available
if load_dotenv is not None:
    root_env = project_root / ".env"
    backend_env = backend_dir / ".env"
    ml_pipeline_env = project_root / "ml_pipeline" / ".env"
    
    if root_env.exists():
        load_dotenv(dotenv_path=root_env)
        print(f"Loaded environment from {root_env}")
    elif backend_env.exists():
        load_dotenv(dotenv_path=backend_env)
        print(f"Loaded environment from {backend_env}")
    elif ml_pipeline_env.exists():
        load_dotenv(dotenv_path=ml_pipeline_env)
        print(f"Loaded environment from {ml_pipeline_env}")
    else:
        print(f"No .env file found at {root_env}, {backend_env}, or {ml_pipeline_env}, relying on system env vars")

from ml_pipeline.inference import analyze_player_session


def _build_local_summary(results: dict) -> str:
    """
    Fallback, non-OpenAI summary that always returns 3 concise sentences.
    This is used when OPENAI_API_KEY is not set or the OpenAI client
    isn't available, so the UX still includes coaching text.
    """
    agreement = float(results.get("agreement_rate", 0.0) or 0.0)
    top_mistakes = results.get("top_mistakes", []) or []

    if agreement < 40:
        s1 = "Right now your decisions are inconsistent across many situations, so there is a lot of room to tighten up your overall strategy."
    elif agreement < 60:
        s1 = "Your overall decision-making is reasonable, but there are still clear spots where your strategy can be sharpened."
    else:
        s1 = "Overall your decisions are fairly solid, but there are still a few key leaks that, if fixed, could noticeably improve your results."

    if top_mistakes:
        # Use the most frequent mistake
        name, count = top_mistakes[0]
        if "fold_to_raise" in name:
            s2 = "You tend to fold too often when facing raises, so look for spots where you can continue more aggressively with reasonable hands and positions."
        elif "call_to_raise" in name:
            s2 = "You call too frequently when the model prefers raising, so focus on turning more of your marginal calls into well-sized value bets or bluffs."
        elif "showdown_to_raise" in name:
            s2 = "You take too many hands all the way to showdown instead of applying pressure earlier with well-timed raises."
        else:
            s2 = "The most common pattern in your mistakes suggests your default reaction in certain spots is off, so pay close attention to those repeated situations."
    else:
        s2 = "The pattern of mistakes suggests a few recurring situations where your default reaction is off, so focus on understanding why a more disciplined or assertive action would perform better there."

    s3 = "To improve, review these high-frequency spots, think about board texture and position before acting, and practice choosing the stronger aggressive or disciplined option instead of autopiloting."
    return " ".join([s1, s2, s3])


def generate_ai_feedback(results: dict) -> str:
    """
    Generate a concise 3-sentence, high-level improvement summary.

    If OpenAI is configured (OPENAI_API_KEY + library installed), we call
    an OpenAI model; otherwise we fall back to a local heuristic summary.
    """
    # Always have a local summary available
    local_summary = _build_local_summary(results)

    # If OpenAI client isn't available or API key is missing, use local version
    if OpenAI is None:
        return local_summary

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return local_summary

    # Try OpenAI; on any error, fall back to local summary
    try:
        print("Using OpenAI for AI coaching summary...")
        client = OpenAI(api_key=api_key)

        summary_payload = {
            "player_name": results.get("player_name"),
            "total_hands": results.get("total_hands"),
            "total_decisions": results.get("total_decisions"),
            "agreement_rate": results.get("agreement_rate"),
            "high_confidence_mistakes": results.get("high_confidence_mistakes"),
            "top_mistakes": results.get("top_mistakes", []),
        }

        prompt = (
            "You are a poker coach AI. Based on the analysis results below, write exactly "
            "three short, decisive sentences with specific, actionable poker strategy advice. "
            "Focus on concrete actions the player should take based on their mistake patterns. "
            "DO NOT use generic phrases like 'strive for consistency' or 'work on improving' - "
            "instead give specific, tactical advice like 'raise 3x the big blind from late position "
            "with suited connectors instead of calling' or 'fold weak pairs when facing a raise on "
            "a high-card flop'. Be direct and specific about what to do differently.\n\n"
            f"Analysis JSON:\n{json.dumps(summary_payload)}\n\n"
            "Write three sentences that are specific, tactical, and actionable - no generic advice."
        )

        # New-style OpenAI client (chat.completions.create)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a direct, tactical poker coach. Give specific, actionable advice. Avoid generic phrases."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
            temperature=0.7,
        )

        text = response.choices[0].message.content.strip()
        return text or local_summary
    except Exception as e:
        print(f"AI coaching summary generation failed, using local summary instead: {e}")
        return local_summary

# Preloaded sample files
# Use absolute path to Poker_Data_Set (more reliable)
DATA_DIR = Path("/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq")

print(f"Initializing sample files from: {DATA_DIR}")
print(f"DATA_DIR exists: {DATA_DIR.exists()}")

# Find actual .phhs files
SAMPLE_FILES = []
if DATA_DIR.exists():
    # Sample 1 - 50NLH
    sample1 = DATA_DIR / "ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5"
    if sample1.exists():
        phhs_files = list(sample1.glob("*.phhs"))
        if phhs_files:
            SAMPLE_FILES.append({
                "name": "Sample 1 - 50NLH",
                "path": str(phhs_files[0])
            })
            print(f"✓ Loaded Sample 1: {phhs_files[0].name}")
    
    # Sample 2 - 100NLH
    sample2 = DATA_DIR / "ABS-2009-07-01_2009-07-23_100NLH_OBFU/1"
    if sample2.exists():
        phhs_files = list(sample2.glob("*.phhs"))
        if phhs_files:
            SAMPLE_FILES.append({
                "name": "Sample 2 - 100NLH",
                "path": str(phhs_files[0])
            })
            print(f"✓ Loaded Sample 2: {phhs_files[0].name}")
    
    # Sample 3 - 200NLH
    sample3 = DATA_DIR / "ABS-2009-07-01_2009-07-23_200NLH_OBFU/2"
    if sample3.exists():
        phhs_files = list(sample3.glob("*.phhs"))
        if phhs_files:
            SAMPLE_FILES.append({
                "name": "Sample 3 - 200NLH",
                "path": str(phhs_files[0])
            })
            print(f"✓ Loaded Sample 3: {phhs_files[0].name}")

print(f"Total sample files loaded: {len(SAMPLE_FILES)}")

# Fallback if files not found - use relative paths
if not SAMPLE_FILES:
    DATA_DIR_REL = Path("../../../Poker_Data_Set/data/handhq")
    SAMPLE_FILES = [
        {
            "name": "Sample 1 - 50NLH",
            "path": str(DATA_DIR_REL / "ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/abs NLH handhq_1-OBFUSCATED.phhs")
        },
        {
            "name": "Sample 2 - 100NLH",
            "path": str(DATA_DIR_REL / "ABS-2009-07-01_2009-07-23_100NLH_OBFU/1/abs NLH handhq_1-OBFUSCATED.phhs")
        },
        {
            "name": "Sample 3 - 200NLH",
            "path": str(DATA_DIR_REL / "ABS-2009-07-01_2009-07-23_200NLH_OBFU/2/abs NLH handhq_1-OBFUSCATED.phhs")
        }
    ]

@app.route('/api/sample-files', methods=['GET'])
def get_sample_files():
    """Get list of preloaded sample files"""
    return jsonify({
        "samples": SAMPLE_FILES,
        "count": len(SAMPLE_FILES)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze a poker session file"""
    try:
        data = request.json
        file_path = data.get('file_path')
        player_name = data.get('player_name', 'Hero')
        use_sample = data.get('use_sample', False)
        sample_index = data.get('sample_index', 0)
        
        if use_sample and sample_index < len(SAMPLE_FILES):
            # Use pre-configured absolute path
            file_path = Path(SAMPLE_FILES[sample_index]['path'])
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        # Run analysis
        results = analyze_player_session(str(file_path), player_name)

        # Generate optional AI coaching summary
        ai_summary = generate_ai_feedback(results) if isinstance(results, dict) else None
        if ai_summary:
            results["ai_summary"] = ai_summary
        
        return jsonify({
            "success": True,
            "results": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/preview', methods=['POST'])
def preview():
    """Preview first few hands from a file"""
    try:
        data = request.json
        use_sample = data.get('use_sample', False)
        sample_index = data.get('sample_index', 0)
        player_name = data.get('player_name', '')
        
        if use_sample and sample_index < len(SAMPLE_FILES):
            file_path = Path(SAMPLE_FILES[sample_index]['path'])
            if not file_path.is_absolute():
                file_path = backend_dir.parent.parent.parent.parent / file_path
        else:
            return jsonify({"error": "Preview only available for sample files"}), 400
        
        if not file_path.exists():
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        # Parse and get first few hands - with error handling
        from poker_coach.parser import parse_hand_history
        try:
            hands = parse_hand_history(str(file_path))
        except Exception as parse_error:
            print(f"Warning: Error parsing some hands: {parse_error}")
            # Try to get players from file directly as fallback
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Extract players using regex as fallback
                import re
                player_pattern = r"players\s*=\s*\[(.*?)\]"
                matches = re.findall(player_pattern, content, re.DOTALL)
                all_players = []
                for match in matches[:10]:  # First 10 matches
                    # Extract quoted strings
                    player_names = re.findall(r"'([^']+)'", match)
                    all_players.extend(player_names)
                unique_players = sorted(list(set(all_players)))
                return jsonify({
                    "success": True,
                    "total_hands": 0,
                    "preview_hands": [],
                    "player_found": None,
                    "sample_players": unique_players[:5],
                    "all_players": unique_players,
                    "warning": "Some parsing errors occurred, but players extracted"
                })
            except Exception as fallback_error:
                return jsonify({
                    "success": False,
                    "error": f"Failed to parse file: {str(parse_error)}"
                }), 500
        
        if not hands:
            return jsonify({
                "success": False,
                "error": "No hands found in file"
            }), 400
        
        # Get first 5 hands
        preview_hands = []
        for hand in hands[:5]:
            try:
                players_list = hand.players[:5] if len(hand.players) > 5 else hand.players
                preview_hands.append({
                    'hand_id': hand.hand_id,
                    'players': players_list,
                    'total_players': len(hand.players),
                    'actions_count': len(hand.actions) if hasattr(hand, 'actions') else 0,
                    'date': f"{hand.date.get('year', '')}-{hand.date.get('month', ''):02d}-{hand.date.get('day', ''):02d}" if hand.date else "Unknown"
                })
            except Exception as e:
                print(f"Warning: Error processing hand {hand.hand_id if hasattr(hand, 'hand_id') else 'unknown'}: {e}")
                continue
        
        # If player name provided, check if they're in the file
        player_found = False
        if player_name:
            for hand in hands[:10]:  # Check first 10 hands
                try:
                    if any(player_name.lower() in p.lower() or p.lower() in player_name.lower() for p in hand.players):
                        player_found = True
                        break
                except:
                    continue
        
        # Get all unique players from the file - with error handling
        all_players = []
        for hand in hands:
            try:
                if hasattr(hand, 'players') and hand.players:
                    all_players.extend(hand.players)
            except Exception as e:
                print(f"Warning: Error extracting players from hand: {e}")
                continue
        
        unique_players = sorted(list(set(all_players)))  # Sort for consistency
        
        return jsonify({
            "success": True,
            "total_hands": len(hands),
            "preview_hands": preview_hands,
            "player_found": player_found if player_name else None,
            "sample_players": list(set([p for hand in hands[:10] for p in (hand.players[:3] if hasattr(hand, 'players') and hand.players else [])]))[:5],  # First few unique players for display
            "all_players": unique_players  # All unique players for dropdown
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Preview error: {error_trace}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/players', methods=['POST'])
def get_players():
    """Get all players from a sample file"""
    try:
        data = request.json
        use_sample = data.get('use_sample', False)
        sample_index = data.get('sample_index', 0)
        
        if not use_sample or sample_index >= len(SAMPLE_FILES):
            return jsonify({"error": "Invalid sample index"}), 400
        
        file_path = Path(SAMPLE_FILES[sample_index]['path'])
        if not file_path.is_absolute():
            file_path = backend_dir.parent.parent.parent.parent / file_path
        
        if not file_path.exists():
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        # Try parsing first
        from poker_coach.parser import parse_hand_history
        all_players = []
        
        try:
            hands = parse_hand_history(str(file_path))
            for hand in hands:
                if hasattr(hand, 'players') and hand.players:
                    all_players.extend(hand.players)
        except Exception as parse_error:
            print(f"Parsing error, trying fallback: {parse_error}")
            # Fallback: extract players directly from file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                import re
                # Look for players= lines
                player_lines = re.findall(r'players\s*=\s*\[(.*?)\]', content, re.DOTALL)
                for line in player_lines[:20]:  # Check first 20 occurrences
                    players = re.findall(r"'([^']+)'", line)
                    all_players.extend(players)
            except Exception as e:
                print(f"Fallback extraction also failed: {e}")
        
        unique_players = sorted(list(set(all_players)))
        
        return jsonify({
            "success": True,
            "players": unique_players,
            "count": len(unique_players)
        })
    
    except Exception as e:
        import traceback
        print(f"Get players error: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Poker Coach AI API"})

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')

