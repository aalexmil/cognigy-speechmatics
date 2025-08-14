##########
# SPEECHMATICS

import os
import json
import time
import asyncio
import logging
from typing import Optional, Dict, Any, List

import boto3
import requests
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("uvicorn.error")
VERBOSE_SM_LOGS = os.getenv("VERBOSE_SM_LOGS", "0") in ("1", "true", "True")
FINAL_ON_END = 1

def _qs(ws: WebSocket, key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        qs = ws.scope.get("query_string", b"").decode("utf-8", "ignore")
        for part in qs.split("&"):
            if part.startswith(key + "="):
                return part.split("=", 1)[1] or default
    except Exception:
        pass
    return default


def _redact_bearer(value: Optional[str]) -> str:
    if not value:
        return "None"
    if value.lower().startswith("bearer "):
        token = value[7:].strip()
        if len(token) <= 6:
            return "Bearer ***"
        return f"Bearer {token[:3]}…{token[-3:]}"
    return "***"


def check_auth(header: Optional[str]) -> bool:
    log.info(f"auth header seen: {_redact_bearer(header)!r}")
    # Accept all for PoC; add real auth if needed
    return True


def get_param(name: str) -> str:
    ssm = boto3.client("ssm")
    param = ssm.get_parameter(Name=name, WithDecryption=True)
    return param["Parameter"]["Value"]


def get_speechmatics_api_key() -> str:
    # Try env first; fall back to AWS SSM param named "speechmatics-api-key"
    return os.environ.get("SPEECHMATICS_API_KEY") or get_param("speechmatics-api-key")


def normalize_lang(lang: str) -> str:
    """Map 'en-US'→'en', also accept common names."""
    if not lang:
        return "en"
    raw = lang.strip().lower()
    common = {
        "english": "en", "german": "de", "spanish": "es", "french": "fr"
    }
    raw = common.get(raw, raw)
    if "-" in raw or "_" in raw:
        raw = raw.split("-", 1)[0].split("_", 1)[0]
    return raw or "en"


def _extract_text_from_results(results: List[Dict[str, Any]]) -> str:
    """Build a string from Speechmatics 'results' array."""
    parts: List[str] = []
    for r in results or []:
        typ = r.get("type")
        if typ not in ("word", "punctuation"):
            continue
        alts = r.get("alternatives") or []
        if not alts:
            continue
        content = (alts[0].get("content") or "").strip()
        if not content:
            continue
        # Insert spacing only for words (punctuation joins)
        if typ == "word":
            if parts and not parts[-1].endswith((" ", "\n")):
                parts.append(" ")
            parts.append(content)
        else:
            parts.append(content)
    return "".join(parts).strip()


def _is_interesting_sm(msg_type: str, data: dict) -> bool:
    """Return True for events worth INFO-level logging."""
    if msg_type in ("RecognitionStarted", "EndOfTranscript", "Error", "Warning"):
        return True
    if msg_type in ("AddTranscript", "AddPartialTranscript"):
        md = data.get("metadata") or {}
        t = (md.get("transcript") or "").strip()
        if not t:
            t = _extract_text_from_results(data.get("results") or [])
        return bool(t.strip())
    return False


# -------- config --------
SPEECHMATICS_REGION = os.environ.get("SPEECHMATICS_REGION", "eu2")  # e.g., eu2/us1
SPEECHMATICS_RT_WSS = f"wss://{SPEECHMATICS_REGION}.rt.speechmatics.com/v2"
DEFAULT_OPERATING_POINT = os.environ.get("SPEECHMATICS_OPERATING_POINT", "standard")  # 'standard'|'enhanced'
DEFAULT_MODEL = os.environ.get("SPEECHMATICS_MODEL", DEFAULT_OPERATING_POINT)  # for qs=model compatibility
DEFAULT_SAMPLE_RATE = int(os.environ.get("DEFAULT_SAMPLE_RATE", "8000"))

app = FastAPI()


@app.websocket("/transcribe/speechmatics")
async def transcribe_speechmatics(ws: WebSocket):
    # ---- connection metadata
    auth = ws.headers.get("authorization")
    proto = ws.headers.get("sec-websocket-protocol")
    ua = ws.headers.get("user-agent")
    xcorr = ws.headers.get("x-correlation-id")
    qs_raw = ws.scope.get("query_string", b"").decode(errors="ignore")

    log.info(
        "WS upgrade from %s; UA=%r; proto=%r; qs=%r; x-correlation-id=%r",
        ws.client, ua, proto, qs_raw, xcorr
    )

    if not check_auth(auth):
        await ws.close(code=1008)
        return

    await ws.accept()
    call_id = xcorr or os.urandom(4).hex()
    t0 = time.time()
    first_audio_at = None

    log.info("[%s] accepted from %s", call_id, ws.client)

    # ---- 1) Expect client's 'start' frame
    try:
        start_msg = await ws.receive_text()
        cfg = json.loads(start_msg)
        if cfg.get("type") != "start":
            log.warning("[%s] first msg not 'start': %r", call_id, cfg)
            await ws.close(code=1002)
            return
    except Exception as e:
        log.exception("[%s] parse start failed: %s", call_id, e)
        await ws.close(code=1002)
        return

    language_in = cfg.get("language", "en-US")
    language = normalize_lang(language_in)
    sample_rate = int(cfg.get("sampleRateHz") or DEFAULT_SAMPLE_RATE)
    interim = bool(cfg.get("interimResults", True))

    model = _qs(ws, "model", DEFAULT_MODEL)
    operating_point = "enhanced" if str(model).lower() == "enhanced" else DEFAULT_OPERATING_POINT

    log.info(
        "[%s] start cfg: language=%s sampleRateHz=%s interim=%s model=%s (operating_point=%s)",
        call_id, language_in, sample_rate, interim, model, operating_point
    )

    # ---- 2) Connect Speechmatics RT WS and send StartRecognition
    try:
        api_key = get_speechmatics_api_key()
        s_ws = await websockets.connect(
            SPEECHMATICS_RT_WSS,
            extra_headers={"Authorization": f"Bearer {api_key}"},
            max_size=10_000_000,
            ping_interval=20,
            ping_timeout=20,
        )

        start_recognition = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
            "transcription_config": {
                "language": language,
                "enable_partials": interim,
                "operating_point": operating_point,
                "max_delay": 4,
                "conversation_config": {                      
                    "end_of_utterance_silence_trigger": 0.75  # ← 0.5–0.8 is good for VG
                },
            },
        }

        log.debug("[%s] speechmatics init: %r", call_id, {**start_recognition, "auth": "***"})

        await s_ws.send(json.dumps(start_recognition))
        log.info("[%s] sent StartRecognition → awaiting RecognitionStarted", call_id)

    except Exception as e:
        log.exception("[%s] speechmatics connect/init failed: %s", call_id, e)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": "speechmatics_connect_failed"}))
        except Exception:
            pass
        await ws.close(code=1011)
        return

    audio_sent = 0
    frames_in = 0
    texts_in = 0
    sm_msgs = 0
    transcripts_sent = 0
    eos_sent = False
    last_log_t = time.time()
    final_committed: List[str] = []
    latest_partial_text = ""
    final_emitted = False   

    recognition_started = asyncio.Event()
    session_reason = "unknown"

    async def _send_eos():
        nonlocal eos_sent
        if eos_sent:
            return
        eos_sent = True
        try:
            # Speechmatics v2 expects EndOfStream with last_seq_no
            await s_ws.send(json.dumps({
                "message": "EndOfStream",
                "last_seq_no": frames_in,  # number of audio chunks we've sent
            }))
            log.info("[%s] EOS sent to Speechmatics (last_seq_no=%d)", call_id, frames_in)
        except Exception as e:
            log.debug("[%s] EOS send suppressed: %s", call_id, e)

    # ---- Pumps
    async def pump_inbound():
        """Client WS → Speechmatics WS."""
        nonlocal audio_sent, frames_in, texts_in, first_audio_at, last_log_t, session_reason
        nonlocal final_committed, latest_partial_text, transcripts_sent, final_emitted
        try:
            # Wait for recognition start before forwarding audio
            await recognition_started.wait()
            log.debug("[%s] recognition started; forwarding audio", call_id)

            while True:
                msg = await ws.receive()

                if "bytes" in msg and msg["bytes"] is not None:
                    buf = msg["bytes"]
                    frames_in += 1
                    audio_sent += len(buf)
                    if first_audio_at is None:
                        first_audio_at = time.time()
                        log.info(
                            "[%s] first audio frame: %d bytes (t+%.3fs from accept)",
                            call_id, len(buf), first_audio_at - t0
                        )

                    if VERBOSE_SM_LOGS:
                        log.debug("[%s] <client #%d> audio frame: %d bytes", call_id, frames_in, len(buf))

                    try:
                        await s_ws.send(buf)  # binary frames = audio in v2
                    except (ConnectionClosed, ConnectionClosedOK):
                        log.info("[%s] Speechmatics ws closed while sending audio; stopping inbound pump", call_id)
                        return

                    # Throughput once per second
                    now = time.time()
                    if now - last_log_t >= 1.0:
                        elapsed = now - (first_audio_at or t0)
                        br = (audio_sent / elapsed) if elapsed > 0 else 0.0
                        log.info("[%s] audio: frames=%d bytes=%d bitrate=%.1f B/s",
                                 call_id, frames_in, audio_sent, br)
                        last_log_t = now

                elif "text" in msg and msg["text"] is not None:
                    texts_in += 1
                    raw_text = msg["text"]
                    log.info("[%s] <client text #%d> %s", call_id, texts_in, raw_text)
                    try:
                        obj = json.loads(raw_text)
                    except Exception:
                        obj = {}
                    if obj.get("type") == "stop":
                        log.info("[%s] CLIENT STOP received (final_emitted=%s)", call_id, final_emitted)
                        session_reason = "client_stop"

                        # --- IMMEDIATE FINAL for Voice Gateway ---
                        # VG ignores transcripts after it sends "stop".
                        # Synthesize a final NOW from committed + latest partial.
                        if FINAL_ON_END and not final_emitted:
                            committed = " ".join(final_committed).strip()
                            fallback_text = " ".join(x for x in (committed, latest_partial_text) if x).strip()
                            if fallback_text:
                                payload = {
                                    "type": "transcription",
                                    "is_final": True,
                                    "alternatives": [{"transcript": fallback_text}],
                                    "channel": 0,
                                    "language": language,
                                }
                                s = json.dumps(payload, ensure_ascii=False)
                                try:
                                    await ws.send_text(s)
                                    log.info("[%s] -> client synthesized FINAL on stop: %s", call_id, s)
                                    transcripts_sent += 1
                                    final_emitted = True
                                except WebSocketDisconnect:
                                    log.info("[%s] client ws closed while sending synthesized final", call_id)

                        await _send_eos()
                        return
                else:
                    log.info("[%s] client sent non-text/non-bytes; treating as EOS", call_id)
                    session_reason = "client_eos"
                    await _send_eos()
                    return

        except WebSocketDisconnect:
            log.info("[%s] client disconnected; sending EOS to Speechmatics", call_id)
            session_reason = "client_disconnect"
            await _send_eos()
        except asyncio.CancelledError:
            log.info("[%s] inbound pump cancelled", call_id)
            return
        except Exception as e:
            log.exception("[%s] inbound pump error: %s", call_id, e)
            session_reason = "inbound_error"
            await _send_eos()

    async def pump_outbound():
        nonlocal sm_msgs, transcripts_sent, final_committed, session_reason, latest_partial_text, final_emitted
        try:
            async for m in s_ws:
                sm_msgs += 1

                # Ignore binary frames unless verbose
                if isinstance(m, (bytes, bytearray)):
                    if VERBOSE_SM_LOGS:
                        log.debug("[%s] <speechmatics raw #%d> binary %d bytes", call_id, sm_msgs, len(m))
                    continue

                # Parse JSON
                try:
                    data = json.loads(m)
                except Exception as e:
                    if VERBOSE_SM_LOGS:
                        log.debug("[%s] could not parse Speechmatics frame #%d: %s; raw=%r", call_id, sm_msgs, e, m)
                    continue

                msg_type = data.get("message")

                # Start gate
                if msg_type == "RecognitionStarted":
                    log.info("[%s] RecognitionStarted; session id=%s", call_id, data.get("id"))
                    recognition_started.set()
                    continue

                # Explicitly ignore super-chatty acks
                if msg_type == "AudioAdded":
                    if VERBOSE_SM_LOGS:
                        log.debug("[%s] AudioAdded #%d", call_id, sm_msgs)
                    continue

                # Warnings/Infos
                if msg_type in ("Warning", "Info"):
                    if _is_interesting_sm(msg_type, data):
                        log.info("[%s] %s: %s", call_id, msg_type, json.dumps(data, ensure_ascii=False))
                    else:
                        if VERBOSE_SM_LOGS:
                            log.debug("[%s] %s: %s", call_id, msg_type, json.dumps(data, ensure_ascii=False))
                    continue
                
                
                if msg_type == "EndOfUtterance":
                    committed = " ".join(final_committed).strip()
                    # Prefer committed; fall back to latest partial text if needed
                    text = committed or latest_partial_text

                    if text and not final_emitted:
                        final_payload = {
                            "type": "transcription",
                            "is_final": True,
                            "alternatives": [{"transcript": text}],
                            "channel": 0,
                            "language": language,
                        }
                        s = json.dumps(final_payload, ensure_ascii=False)
                        log.info("[%s] EOU -> emitting single FINAL to client: %s", call_id, s)
                        await ws.send_text(s)
                        transcripts_sent += 1
                        final_emitted = True

                    # After we’ve emitted the final, stop forwarding any more partials
                    continue
                
                    
                    
                # Errors
                if msg_type == "Error":
                    err = data.get("reason") or data.get("detail") or data
                    log.warning("[%s] Speechmatics error: %r", call_id, err)
                    try:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "provider": "speechmatics",
                            "message": err,
                        }))
                    except WebSocketDisconnect:
                        log.info("[%s] client closed while propagating error", call_id)
                        return
                    continue

                # Partials & segment finals
                if msg_type in ("AddPartialTranscript", "AddTranscript"):
                    if final_emitted:
                            continue  # ignore anything after our final
                    md = data.get("metadata") or {}
                    text = (md.get("transcript") or "").strip() or _extract_text_from_results(data.get("results") or [])
                    is_seg_final = (msg_type == "AddTranscript")

                    # track freshest partial for synth-final on stop
                    if (not is_seg_final) and text:
                        latest_partial_text = text

                    if is_seg_final and text:
                        # Accumulate committed text for the session
                        final_committed.append(text)

                    committed = " ".join(final_committed).strip()
                    transcript = " ".join(x for x in (committed, "" if is_seg_final else text) if x).strip()

                    if transcript:
                        if FINAL_ON_END:
                            # Never mark segment final as utterance final
                            payload = {
                                "type": "transcription",
                                "is_final": False,
                                "alternatives": [{"transcript": transcript}],
                                "channel": 0,
                                "language": language,
                            }
                            s = json.dumps(payload, ensure_ascii=False)
                            log.info("[%s] -> client transcription (final=%s): %s", call_id, False, s)
                            try:
                                await ws.send_text(s)
                                transcripts_sent += 1
                            except WebSocketDisconnect:
                                log.info("[%s] client ws closed while sending transcript; stopping outbound pump", call_id)
                                return
                        else:
                            # Legacy behavior: segment finals marked as final
                            payload = {
                                "type": "transcription",
                                "is_final": is_seg_final,
                                "alternatives": [{"transcript": transcript}],
                                "channel": 0,
                                "language": language,
                            }
                            s = json.dumps(payload, ensure_ascii=False)
                            log.info("[%s] -> client transcription (final=%s): %s", call_id, is_seg_final, s)
                            try:
                                await ws.send_text(s)
                                transcripts_sent += 1
                                if is_seg_final:
                                    final_emitted = True
                            except WebSocketDisconnect:
                                log.info("[%s] client ws closed while sending transcript; stopping outbound pump", call_id)
                                return
                    else:
                        if VERBOSE_SM_LOGS:
                            log.debug("[%s] %s with no text", call_id, msg_type)
                    continue

                # End of transcript → close client WS with a single final
                if msg_type == "EndOfTranscript":
                    committed = " ".join(final_committed).strip()
                    if committed and not final_emitted:
                        final_payload = {
                            "type": "transcription",
                            "is_final": True,
                            "alternatives": [{"transcript": committed}],
                            "channel": 0,
                            "language": language,
                        }
                        s = json.dumps(final_payload, ensure_ascii=False)
                        log.info("[%s] -> client final transcript on EndOfTranscript: %s", call_id, s)
                        try:
                            await ws.send_text(s)
                            transcripts_sent += 1
                            final_emitted = True
                        except WebSocketDisconnect:
                            pass

                    session_reason = "end_of_transcript"
                    log.info("[%s] EndOfTranscript; closing client WS", call_id)
                    try:
                        await ws.close(code=1000)
                    finally:
                        return

                # Unknown message
                if VERBOSE_SM_LOGS:
                    log.debug("[%s] unhandled message type=%r: %s", call_id, msg_type, json.dumps(data, ensure_ascii=False))

        except (ConnectionClosed, ConnectionClosedOK) as e:
            log.info("[%s] Speechmatics ws closed: %s", call_id, repr(e))
        except asyncio.CancelledError:
            log.info("[%s] outbound pump cancelled", call_id)
            return
        except Exception as e:
            log.exception("[%s] Speechmatics read loop error: %s", call_id, e)
        finally:
            try:
                await s_ws.close()
            except Exception:
                pass

    # ---- run both pumps
    task_out = asyncio.create_task(pump_outbound(), name=f"outbound-{call_id}")
    task_in = asyncio.create_task(pump_inbound(), name=f"inbound-{call_id}")

    # Wait until inbound or outbound finishes (whichever first)
    done, pending = await asyncio.wait(
        {task_in, task_out}, return_when=asyncio.FIRST_COMPLETED, timeout=60
    )

    # If inbound finished but outbound still pending, give it a short grace period to flush finals
    if task_in in done and task_out in pending:
        _done2, _pending2 = await asyncio.wait(
            {task_out}, return_when=asyncio.FIRST_COMPLETED, timeout=2.5
        )
        if task_out in _pending2:
            # Fallback: synthesize a final from committed + latest partial (if not yet sent)
            committed = " ".join(final_committed).strip()
            fallback_text = " ".join(x for x in (committed, latest_partial_text) if x).strip()
            if fallback_text and not final_emitted:
                final_payload = {
                    "type": "transcription",
                    "is_final": True,
                    "alternatives": [{"transcript": fallback_text}],
                    "channel": 0,
                    "language": language,
                }
                s = json.dumps(final_payload, ensure_ascii=False)
                log.warning("[%s] Synthesizing final due to missing EndOfTranscript: %s", call_id, s)
                try:
                    await ws.send_text(s)
                    transcripts_sent += 1
                    final_emitted = True
                except WebSocketDisconnect:
                    pass
            pending = set([task_out])

    for t in done:
        try:
            _ = t.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("[%s] task %s raised: %s", call_id, getattr(t, "get_name", lambda: "?")(), e)

    # Cancel whichever is still running; drain quietly
    for t in pending:
        t.cancel()
    if pending:
        _ = await asyncio.gather(*pending, return_exceptions=True)

    dur = max(0.001, time.time() - t0)
    bitrate = audio_sent / dur
    log.info(
        "[%s] finished; bytes=%d frames=%d texts=%d sm_msgs=%d transcripts_sent=%d dur=%.2fs avg_bitrate=%.1f B/s reason=%s",
        call_id, audio_sent, frames_in, texts_in, sm_msgs, transcripts_sent, dur, bitrate, session_reason
    )