import os
import threading
from flask import Flask, render_template, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAST_DIR = os.path.join(BASE_DIR, "static", "past_calls")

os.makedirs(PAST_DIR, exist_ok=True)

app = Flask(__name__)

running = False
thread = None

# FIXED EMOTION LIST (Always show all in dashboard)
ALL_EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "surprised", "disgust"
]

# ---------- GET LAST 10 CALLS ----------
def get_calls():
    try:
        files = [f for f in os.listdir(PAST_DIR) if f.endswith(".wav")]
        files.sort(reverse=True)
        return files[:10]
    except:
        return []

# ---------- DASHBOARD ----------
@app.route("/")
def home():
    return render_template("index.html", calls=get_calls())

# ---------- LIVE DATA ----------
@app.route("/live")
def live():
    try:
        import realtime

        # Ensure ALL emotions exist (fix pie missing labels)
        pie_counts = {e: 0 for e in ALL_EMOTIONS}
        pie_counts.update(dict(realtime.emotion_counts))

        return jsonify({
            "emotion": realtime.current_emotion,
            "confidence": realtime.current_confidence,
            "accuracy": realtime.current_accuracy,
            "time": realtime.call_time,
            "trend_emotions": list(realtime.emotion_history),
            "trend_time": list(realtime.time_history),
            "pie": pie_counts
        })

    except Exception as e:
        print("LIVE ERROR:", e, flush=True)
        return jsonify({
            "emotion": "Listening...",
            "confidence": 0,
            "accuracy": 0,
            "time": 0,
            "trend_emotions": [],
            "trend_time": [],
            "pie": {e: 0 for e in ALL_EMOTIONS}
        })

# ---------- THREAD RUN ----------
def run_emotion():
    import realtime
    realtime.main_loop()

# ---------- START ----------
@app.route("/start")
def start():
    global running, thread

    if running:
        return "Already Running"

    try:
        import realtime
        realtime.reset_live()
        realtime.stop_flag = False

        running = True
        thread = threading.Thread(target=run_emotion, daemon=True)
        thread.start()

        print("🎤 Realtime started", flush=True)
        return "OK"

    except Exception as e:
        print("START ERROR:", e, flush=True)
        running = False
        return "ERROR"

# ---------- STOP ----------
@app.route("/stop")
def stop():
    global running

    try:
        import realtime
        realtime.stop_flag = True
        running = False

        print("🛑 Recording stopped", flush=True)
        return "OK"

    except Exception as e:
        print("STOP ERROR:", e, flush=True)
        return "ERROR"

# ---------- RUN SERVER ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
