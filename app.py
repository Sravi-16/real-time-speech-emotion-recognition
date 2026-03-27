import os
import uuid
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from backend.audio_utils import EMOTION_TO_ID
from backend.database import close_session, create_session, get_predictions, init_db, insert_prediction, list_sessions
from backend.inference import EmotionPredictor
from backend.streaming import MicrophoneStreamer
from backend.audio_utils import write_wav


st.set_page_config(page_title="Real-Time Speech Emotion Recognition", page_icon="🎙️", layout="wide")
st.title("🎙️ Real-Time Speech Emotion Recognition")
st.caption("Microphone-based continuous emotion detection using CNN+LSTM.")
init_db()

DISPLAY_MAP = {
    "happy": "happy",
    "neutral": "neutral",
    "angry": "angry",
    "sad": "sad",
    "frustrated": "neutral",
    "confused": "neutral",
}
EMOJI_MAP = {
    "happy": "😄",
    "neutral": "😐",
    "angry": "😠",
    "sad": "😢",
    "no_speech": "🔇",
}


def to_display_emotion(label: str) -> str:
    if label == "no_speech":
        return "no_speech"
    return DISPLAY_MAP.get(label, "neutral")


def to_display_probs(raw_probs: Dict[str, float]) -> Dict[str, float]:
    merged = {"happy": 0.0, "neutral": 0.0, "angry": 0.0, "sad": 0.0}
    for k, v in raw_probs.items():
        mapped = DISPLAY_MAP.get(k, "neutral")
        merged[mapped] += float(v)
    total = sum(merged.values())
    if total > 0:
        merged = {k: (v / total) * 100.0 for k, v in merged.items()}
    return merged


def init_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "streamer" not in st.session_state:
        st.session_state.streamer = MicrophoneStreamer(chunk_seconds=1.2, overlap_seconds=0.6)
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "latest" not in st.session_state:
        st.session_state.latest = None
    if "timeline" not in st.session_state:
        st.session_state.timeline = []
    if "recording_chunks" not in st.session_state:
        st.session_state.recording_chunks = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "last_recording_path" not in st.session_state:
        st.session_state.last_recording_path = None
    if "last_call_summary" not in st.session_state:
        st.session_state.last_call_summary = None


def ensure_predictor():
    if st.session_state.predictor is None:
        st.session_state.predictor = EmotionPredictor(
            model_path="models/emotion_cnn_lstm.pt",
            smooth_window=2,
            silence_rms_threshold=0.008,
            ema_alpha=0.72,
        )


init_state()

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("Start Microphone", type="primary", use_container_width=True):
        try:
            ensure_predictor()
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"
            create_session(st.session_state.session_id)
            st.session_state.recording_chunks = []
            st.session_state.streamer.start()
            st.session_state.running = True
        except Exception as e:
            st.error(f"Failed to start microphone: {e}")
            st.session_state.running = False

with col2:
    if st.button("Stop Microphone", use_container_width=True):
        st.session_state.streamer.stop()
        st.session_state.running = False
        rec_path = None
        if st.session_state.recording_chunks:
            merged = np.concatenate(st.session_state.recording_chunks).astype("float32")
            os.makedirs("training/results/recordings", exist_ok=True)
            rec_path = f"training/results/recordings/{st.session_state.session_id}.wav"
            write_wav(rec_path, merged)
            st.session_state.last_recording_path = rec_path
        if st.session_state.session_id:
            close_session(st.session_state.session_id, rec_path)
        if st.session_state.timeline:
            call_df = pd.DataFrame(st.session_state.timeline)
            dominant = call_df["emotion"].value_counts().idxmax()
            avg_conf = float(call_df["confidence"].mean())
            st.session_state.last_call_summary = {
                "dominant_emotion": dominant,
                "avg_confidence": avg_conf,
                "samples": len(call_df),
            }

with col3:
    status = "🟢 Running" if st.session_state.running else "⚪ Stopped"
    st.markdown(f"### Status: {status}")

if st.session_state.running:
    st_autorefresh(interval=600, key="emotion_refresh")

    chunk_result = st.session_state.streamer.get_chunk()
    if chunk_result is not None:
        chunk, ts = chunk_result
        st.session_state.recording_chunks.append(chunk)
        out = st.session_state.predictor.predict_chunk(chunk, timestamp=ts)
        display_probs = to_display_probs(out["probs"])
        # Decide the *display* label from the probability distribution among the 4 UI classes.
        # This reduces cases where UI shows neutral even when happy/angry/sad is higher.
        if out["label"] == "no_speech":
            display_label = "no_speech"
            display_conf = 0.0
        else:
            display_label = max(display_probs.keys(), key=lambda k: display_probs[k])
            display_conf = float(display_probs.get(display_label, 0.0))
        out["display_label"] = display_label
        out["display_probs"] = display_probs
        out["display_confidence"] = display_conf
        st.session_state.latest = out
        if st.session_state.session_id:
            insert_prediction(
                st.session_state.session_id,
                display_label,
                display_conf,
                out.get("rms", 0.0),
            )
        if display_label != "no_speech":
            st.session_state.timeline.append(
                {
                    "timestamp": pd.to_datetime(ts, unit="s"),
                    "emotion": display_label,
                    "confidence": display_conf,
                }
            )
            if len(st.session_state.timeline) > 300:
                st.session_state.timeline = st.session_state.timeline[-300:]

top_left, top_right = st.columns([1, 2])

with top_left:
    st.subheader("Current Emotion")
    if st.session_state.latest is None:
        st.info("Start microphone to begin prediction.")
    else:
        if st.session_state.latest.get("display_label") == "no_speech":
            st.metric("Emotion", "No Speech")
            st.metric("Confidence", "0.00%")
        else:
            label = st.session_state.latest["display_label"]
            emoji = EMOJI_MAP.get(label, "🙂")
            st.metric("Emotion", f"{emoji} {label.title()}")
            st.metric("Confidence", f"{st.session_state.latest['display_confidence']:.2f}%")
        st.caption(f"Mic RMS: {st.session_state.latest.get('rms', 0.0):.4f}")

with top_right:
    st.subheader("Emotion Probabilities")
    if st.session_state.latest is not None:
        probs: Dict[str, float] = st.session_state.latest.get("display_probs", {})
        df_prob = pd.DataFrame({"emotion": list(probs.keys()), "probability": list(probs.values())})
        df_prob["emotion"] = pd.Categorical(
            df_prob["emotion"], categories=["happy", "neutral", "angry", "sad"], ordered=True
        )
        df_prob = df_prob.sort_values("emotion")
        fig_prob = px.bar(df_prob, x="emotion", y="probability", range_y=[0, 100], color="emotion")
        fig_prob.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Probability (%)", height=330)
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.empty()

st.subheader("Emotion Timeline")
if len(st.session_state.timeline) > 0:
    df_timeline = pd.DataFrame(st.session_state.timeline)
    fig_line = px.line(df_timeline, x="timestamp", y="confidence", color="emotion", markers=True)
    fig_line.update_layout(height=360, yaxis_title="Confidence (%)")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Emotion Distribution (Live Pie Chart)")
    pie_df = (
        df_timeline.groupby("emotion", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    fig_pie = px.pie(pie_df, names="emotion", values="count", hole=0.35)
    fig_pie.update_layout(height=360)
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("Timeline will appear after first predictions.")

with st.expander("Troubleshooting"):
    st.markdown(
        """
        - Ensure model exists at `models/emotion_cnn_lstm.pt`.
        - If microphone fails on Windows, close apps currently using the input device.
        - Use a quiet environment for stable predictions.
        - Recorded calls are saved in `training/results/recordings/`.
        - DB logs are saved in `training/results/emotion_app.db`.
        """
    )

if st.session_state.last_recording_path:
    st.success(f"Last call recording saved: {st.session_state.last_recording_path}")
    st.audio(st.session_state.last_recording_path)
    if st.session_state.last_call_summary:
        dominant = st.session_state.last_call_summary["dominant_emotion"]
        emoji = EMOJI_MAP.get(dominant, "🙂")
        st.info(
            f"Recorded Call Summary: {emoji} Dominant Emotion: {dominant.title()} | "
            f"Avg Confidence: {st.session_state.last_call_summary['avg_confidence']:.2f}% | "
            f"Samples: {st.session_state.last_call_summary['samples']}"
        )

st.divider()
st.subheader("Past Calls (Stored Recordings + DB Logs)")
sessions_df = list_sessions(limit=50)
if sessions_df.empty:
    st.info("No stored calls yet. Start and stop microphone to create a session.")
else:
    sessions_df = sessions_df.copy()
    sessions_df["started_at"] = pd.to_datetime(sessions_df["started_at"], errors="coerce")
    sessions_df["ended_at"] = pd.to_datetime(sessions_df["ended_at"], errors="coerce")
    sessions_df["status"] = sessions_df["ended_at"].apply(lambda x: "closed" if pd.notna(x) else "open")
    st.dataframe(
        sessions_df[["session_id", "started_at", "ended_at", "status", "recording_path"]],
        use_container_width=True,
        hide_index=True,
    )

    session_ids = sessions_df["session_id"].tolist()
    selected = st.selectbox("Select a session to view/play", session_ids)
    sel_row = sessions_df[sessions_df["session_id"] == selected].iloc[0]

    st.markdown("### Playback")
    rec_path = sel_row["recording_path"]
    if isinstance(rec_path, str) and rec_path.strip() and os.path.exists(rec_path):
        st.audio(rec_path)
        with open(rec_path, "rb") as f:
            st.download_button(
                "Download Recording (.wav)",
                data=f,
                file_name=os.path.basename(rec_path),
                mime="audio/wav",
                use_container_width=True,
            )
    else:
        st.warning("No recording file found for this session (maybe it was stopped before saving).")

    st.markdown("### Stored Emotion Data (from database)")
    pred_df = get_predictions(selected)
    if pred_df.empty:
        st.info("No prediction rows found for this session.")
    else:
        pred_df = pred_df.copy()
        pred_df["ts_utc"] = pd.to_datetime(pred_df["ts_utc"], errors="coerce")
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        colA, colB = st.columns([2, 1])
        with colA:
            fig_hist = px.line(pred_df, x="ts_utc", y="confidence", color="emotion", markers=True)
            fig_hist.update_layout(height=320, yaxis_title="Confidence (%)", xaxis_title="Time (UTC)")
            st.plotly_chart(fig_hist, use_container_width=True)
        with colB:
            dist = pred_df["emotion"].value_counts().reset_index()
            dist.columns = ["emotion", "count"]
            fig_dist = px.pie(dist, names="emotion", values="count", hole=0.35)
            fig_dist.update_layout(height=320)
            st.plotly_chart(fig_dist, use_container_width=True)
