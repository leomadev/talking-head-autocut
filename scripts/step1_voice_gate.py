#!/usr/bin/env python3
"""
Step 1: Acoustic Voice Gate (Pre-Whisper, Apple Silicon MPS GPU Accelerated)
- Loads or builds the user's permanent voice profile (resources/my_voice_profile.npy).
  * Zero-config auto-enrollment: On the first run, if resources/my_voice_profile.npy does not exist,
    automatically builds and saves the speaker's 256-dim voice fingerprint from their 8 longest
    monologue islands (>= 3.5s).
  * Explicit enrollment: Pass `--enroll <video_or_audio_file>` to rebuild/overwrite the profile anytime.
- Runs VoiceEncoder on Apple Silicon GPU (PyTorch MPS) when available (falls back to CUDA/CPU).
- Extracts 16kHz mono audio locked to video container PTS=0 (`aresample=async=1:first_pts=0`).
- Scans raw 16kHz audio using WebRTC VAD speech islands (tight 90ms padding).
- Evaluates each speech island AND 0.8s sliding sub-windows against the user's voice embedding.
- Surgically mutes any audio interval that does NOT belong to the user (Google Translate TTS,
  phone notifications, system alerts, third-party speech) while keeping natural room ambiance.
- Outputs a 100% speaker-isolated WAV file (step1_clean_voice.wav) and gate report JSON.
"""
import os
import sys
import json
import struct
import subprocess
import numpy as np
import soundfile as sf
import torch
import webrtcvad
from resemblyzer import VoiceEncoder

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROFILE_PATH = os.path.join(SKILL_ROOT, "resources", "my_voice_profile.npy")


def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_raw_16k(video_path, wav_path):
    print(f"[Step 1] Extracting 16kHz mono audio from {video_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-af", "aresample=async=1:first_pts=0",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_speech_islands(wav, sr=16000, frame_ms=30, pad_ms=90):
    vad = webrtcvad.Vad(2)
    frame_len = int(sr * frame_ms / 1000)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

    num_frames = len(pcm16) // frame_len
    is_speech = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        frame = pcm16[i * frame_len : (i + 1) * frame_len]
        raw_bytes = struct.pack(f"{len(frame)}h", *frame)
        is_speech[i] = vad.is_speech(raw_bytes, sr)

    pad_frames = max(1, int(pad_ms / frame_ms))
    dilated = np.copy(is_speech)
    for i in range(num_frames):
        if is_speech[i]:
            dilated[max(0, i - pad_frames) : min(num_frames, i + pad_frames + 1)] = True

    islands = []
    in_island = False
    start_idx = 0
    for i, val in enumerate(dilated):
        if val and not in_island:
            in_island = True
            start_idx = i
        elif not val and in_island:
            in_island = False
            islands.append((start_idx * frame_ms / 1000.0, i * frame_ms / 1000.0))
    if in_island:
        islands.append((start_idx * frame_ms / 1000.0, num_frames * frame_ms / 1000.0))

    return islands


def get_or_create_voice_profile(encoder, wav, sr, islands, profile_path=DEFAULT_PROFILE_PATH, force_enroll=False):
    if os.path.exists(profile_path) and not force_enroll:
        print(f"[Step 1] Loaded permanent speaker profile from {profile_path}")
        return np.load(profile_path)

    print("[Step 1] Building permanent speaker profile from longest monologue islands...")
    long_islands = sorted([isl for isl in islands if (isl[1] - isl[0]) >= 3.5],
                          key=lambda x: x[1] - x[0], reverse=True)[:8]
    if not long_islands:
        long_islands = sorted(islands, key=lambda x: x[1] - x[0], reverse=True)[:5]
    embeds = [encoder.embed_utterance(wav[int(s * sr) : int(e * sr)]) for s, e in long_islands]
    profile = np.mean(embeds, axis=0)
    profile /= np.linalg.norm(profile)
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    np.save(profile_path, profile)
    print(f"[Step 1] Saved permanent speaker profile (256-dim d-vector) to {profile_path}")
    return profile


def enroll_only(media_path, profile_path=DEFAULT_PROFILE_PATH):
    tmp_wav = "/tmp/enroll_16k.wav"
    extract_raw_16k(media_path, tmp_wav)
    wav, sr = sf.read(tmp_wav)
    device = get_best_device()
    encoder = VoiceEncoder(device=device)
    islands = get_speech_islands(wav, sr=sr, frame_ms=30, pad_ms=90)
    get_or_create_voice_profile(encoder, wav, sr, islands, profile_path=profile_path, force_enroll=True)
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)


def run_step1(video_path, work_dir="audio_work", force_enroll=False):
    os.makedirs(work_dir, exist_ok=True)
    raw_wav_path = os.path.join(work_dir, "raw_16k.wav")
    clean_wav_path = os.path.join(work_dir, "step1_clean_voice.wav")
    report_path = os.path.join(work_dir, "step1_gate_report.json")

    extract_raw_16k(video_path, raw_wav_path)
    wav, sr = sf.read(raw_wav_path)
    device = get_best_device()
    encoder = VoiceEncoder(device=device)
    print(f"[Step 1] VoiceEncoder running on device: {device.upper()}")

    print("[Step 1] Detecting voice activity islands across raw audio...")
    islands = get_speech_islands(wav, sr=sr, frame_ms=30, pad_ms=90)
    profile = get_or_create_voice_profile(encoder, wav, sr, islands, force_enroll=force_enroll)

    clean_wav = np.copy(wav)
    kept_islands = []
    muted_islands = []

    for s, e in islands:
        dur = e - s
        s_idx, e_idx = int(s * sr), int(e * sr)
        chunk = wav[s_idx:e_idx]
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if dur < 0.35 or rms < 0.002:
            similarity = 0.0
            is_me = False
        else:
            emb = encoder.embed_utterance(chunk)
            similarity = float(np.dot(profile, emb))
            if rms > 0.045 and similarity < 0.68:
                is_me = False
            else:
                is_me = similarity >= 0.64

        if is_me and dur >= 1.6:
            win_s = 0.80
            step_s = 0.25
            t = s
            while t + win_s <= e + 0.05:
                w_start = max(s, t)
                w_end = min(e, t + win_s)
                if w_end - w_start >= 0.50:
                    sub_chunk = wav[int(w_start * sr) : int(w_end * sr)]
                    sub_rms = float(np.sqrt(np.mean(sub_chunk ** 2)))
                    if sub_rms > 0.003:
                        sub_sim = float(np.dot(profile, encoder.embed_utterance(sub_chunk)))
                        if (sub_rms > 0.045 and sub_sim < 0.68) or sub_sim < 0.58:
                            clean_wav[int(w_start * sr) : int(w_end * sr)] *= 0.001
                            muted_islands.append({
                                "start": round(w_start, 2),
                                "end": round(w_end, 2),
                                "duration": round(w_end - w_start, 2),
                                "similarity": round(sub_sim, 3),
                                "note": "surgical_subwindow_mute"
                            })
                t += step_s

        info = {"start": round(s, 2), "end": round(e, 2), "duration": round(dur, 2), "similarity": round(similarity, 3)}
        if is_me:
            kept_islands.append(info)
        else:
            clean_wav[s_idx:e_idx] = chunk * 0.001
            muted_islands.append(info)

    sf.write(clean_wav_path, clean_wav, sr)
    with open(report_path, "w") as f:
        json.dump({
            "total_islands": len(islands),
            "kept_my_voice": len(kept_islands),
            "muted_foreign_or_noise": len(muted_islands),
            "kept": kept_islands,
            "muted": muted_islands
        }, f, indent=2)

    print(f"[Step 1] Done! Kept {len(kept_islands)} islands (Your Voice), Muted {len(muted_islands)} foreign/TTS/noise intervals.")
    print(f"[Step 1] Output clean audio: {clean_wav_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  step1_voice_gate.py <input_video> [work_dir] [--enroll]\n  step1_voice_gate.py --enroll <sample_video_or_audio>")
        sys.exit(1)
    if sys.argv[1] == "--enroll":
        if len(sys.argv) < 3:
            print("Usage: step1_voice_gate.py --enroll <sample_video_or_audio>")
            sys.exit(1)
        enroll_only(sys.argv[2])
    else:
        video_in = sys.argv[1]
        work_d = sys.argv[2] if (len(sys.argv) > 2 and not sys.argv[2].startswith("--")) else "audio_work"
        force_e = "--enroll" in sys.argv
        run_step1(video_in, work_d, force_enroll=force_e)
