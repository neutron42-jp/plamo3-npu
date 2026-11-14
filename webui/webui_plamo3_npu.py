import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import sys
import threading
import time

from aiohttp import web
import onnxruntime_genai as og
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_plamo3_npu as npu_runner

MAX_SEQ_LEN = npu_runner.SAFE_TOTAL_SEQ_LEN
WEBUI_DIR = Path(__file__).resolve().parent

# Supported languages from PLaMo-3 Translate 0.8B README_ja.md + chat_template.jinja (ja-easy).
LANGUAGES: list[dict[str, str]] = [
    {"code": "ja", "name": "日本語 (Japanese)"},
    {"code": "ja-easy", "name": "やさしい日本語 (Easy Japanese)"},
    {"code": "en", "name": "英語 (English)"},
    {"code": "zh-hans", "name": "中国語・簡体字 (Chinese Simplified)"},
    {"code": "zh-hant", "name": "中国語・繁体字 (Chinese Traditional)"},
    {"code": "ko", "name": "韓国語 (Korean)"},
    {"code": "fr", "name": "フランス語 (French)"},
    {"code": "de", "name": "ドイツ語 (German)"},
    {"code": "es", "name": "スペイン語 (Spanish)"},
    {"code": "it", "name": "イタリア語 (Italian)"},
    {"code": "ru", "name": "ロシア語 (Russian)"},
    {"code": "ar", "name": "アラビア語 (Arabic)"},
    {"code": "hi", "name": "ヒンディー語 (Hindi)"},
    {"code": "id", "name": "インドネシア語 (Indonesian)"},
    {"code": "th", "name": "タイ語 (Thai)"},
    {"code": "vi", "name": "ベトナム語 (Vietnamese)"},
    {"code": "nl", "name": "オランダ語 (Dutch)"},
    {"code": "pl", "name": "ポーランド語 (Polish)"},
    {"code": "az", "name": "アゼルバイジャン語 (Azerbaijani)"},
    {"code": "be", "name": "ベラルーシ語 (Belarusian)"},
    {"code": "bg", "name": "ブルガリア語 (Bulgarian)"},
    {"code": "bn", "name": "ベンガル語 (Bengali)"},
    {"code": "ca", "name": "カタルーニャ語 (Catalan)"},
    {"code": "ceb", "name": "セブアノ語 (Cebuano)"},
    {"code": "et", "name": "エストニア語 (Estonian)"},
    {"code": "eu", "name": "バスク語 (Basque)"},
    {"code": "fa", "name": "ペルシャ語 (Persian)"},
    {"code": "gu", "name": "グジャラート語 (Gujarati)"},
    {"code": "he", "name": "ヘブライ語 (Hebrew)"},
    {"code": "hr", "name": "クロアチア語 (Croatian)"},
    {"code": "hu", "name": "ハンガリー語 (Hungarian)"},
    {"code": "hy", "name": "アルメニア語 (Armenian)"},
    {"code": "jv", "name": "ジャワ語 (Javanese)"},
    {"code": "ka", "name": "ジョージア語 (Georgian)"},
    {"code": "kk", "name": "カザフ語 (Kazakh)"},
    {"code": "km", "name": "クメール語 (Khmer)"},
    {"code": "lt", "name": "リトアニア語 (Lithuanian)"},
    {"code": "lv", "name": "ラトビア語 (Latvian)"},
    {"code": "ml", "name": "マラヤーラム語 (Malayalam)"},
    {"code": "mr", "name": "マラーティー語 (Marathi)"},
    {"code": "ms", "name": "マレー語 (Malay)"},
    {"code": "my", "name": "ミャンマー語 (Myanmar)"},
    {"code": "ne", "name": "ネパール語 (Nepali)"},
    {"code": "pa", "name": "パンジャーブ語 (Punjabi)"},
    {"code": "si", "name": "シンハラ語 (Sinhala)"},
    {"code": "sk", "name": "スロバキア語 (Slovak)"},
    {"code": "sl", "name": "スロベニア語 (Slovenian)"},
    {"code": "sr", "name": "セルビア語 (Serbian)"},
    {"code": "sw", "name": "スワヒリ語 (Swahili)"},
    {"code": "ta", "name": "タミル語 (Tamil)"},
    {"code": "te", "name": "テルグ語 (Telugu)"},
    {"code": "tl", "name": "タガログ語 (Tagalog)"},
    {"code": "ur", "name": "ウルドゥー語 (Urdu)"},
    {"code": "uz", "name": "ウズベク語 (Uzbek)"},
]


@dataclass
class TranslationPlan:
    """Validated translation workload: either a single NPU pass or sequence-safe chunks."""

    text: str
    source_lang: str | None
    target_lang: str
    form: str | None
    context: str | None
    glossary: list[dict[str, str]] | None
    requested_max_new_tokens: int
    full_prompt_tokens: int
    single_input_ids: list[int] | None
    chunks: list[tuple[str, str]] = field(default_factory=list)
    trailing_sep: str = ""


class NPUTranslatorEngine:
    """Single-worker NPU inference engine with streaming and cancellation support."""

    def __init__(self, model_dir: Path, tokenizer_dir: Path) -> None:
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        npu_runner.configure_ryzenai_runtime()
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        cfg = og.Config(str(model_dir))
        npu_runner.configure_ryzenai_options(cfg, model_dir)
        self.model = og.Model(cfg)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npu-worker")
        self._state_lock = threading.Lock()
        self._latest_request_id = 0
        self._active_cancel_event: threading.Event | None = None
        self.last_token_attn_len: int | None = None
        self._warmup()

    def _warmup(self) -> None:
        """Warm up Prefill and Token DynamicDispatch kernels so first UI request runs at steady speed."""
        t0 = time.perf_counter()
        warmup_ids = npu_runner.encode_prompt(self.tokenizer, "Hello", "en", "ja")
        res = npu_runner.run_generation(
            self.model,
            self.tokenizer,
            warmup_ids,
            max_new_tokens=8,
            inspect_first_logits=False,
            last_token_attn_len=self.last_token_attn_len,
        )
        self.last_token_attn_len = res["last_token_attn_len"]
        print(
            f"[WebUI] NPU warmup ready in {(time.perf_counter() - t0):.2f}s "
            f"(prefill={res['prefill_ms']:.1f}ms, decode={res['tok_per_s']:.1f} tok/s)",
            flush=True,
        )

    def start_request(self) -> tuple[int, threading.Event]:
        """Register a new translation request and signal any active generation to stop."""
        with self._state_lock:
            self._latest_request_id += 1
            if self._active_cancel_event is not None:
                self._active_cancel_event.set()
            cancel_event = threading.Event()
            self._active_cancel_event = cancel_event
            return self._latest_request_id, cancel_event

    def is_latest(self, req_id: int) -> bool:
        with self._state_lock:
            return req_id == self._latest_request_id

    def prepare_translation_plan(
        self,
        text: str,
        source_lang: str | None,
        target_lang: str,
        form: str | None,
        context: str | None,
        glossary: list[dict[str, str]] | None,
        requested_max_new_tokens: int,
    ) -> TranslationPlan:
        """Encode translation request and plan single-pass vs NPU-safe history-aware chunking."""
        src = None if not source_lang or source_lang == "auto" else source_lang.strip().lower()
        tgt = target_lang.strip().lower()
        frm = form.strip().lower() if form and form.strip() else None
        ctx = context if context and context.strip() else None
        req_max = max(requested_max_new_tokens, 16)

        if not text.strip():
            return TranslationPlan(
                text="",
                source_lang=src,
                target_lang=tgt,
                form=frm,
                context=ctx,
                glossary=glossary,
                requested_max_new_tokens=0,
                full_prompt_tokens=0,
                single_input_ids=[],
            )

        full_ids = npu_runner.encode_prompt(
            self.tokenizer,
            text,
            src,
            tgt,
            form=frm,
            context=ctx,
            glossary=glossary,
        )
        if len(full_ids) <= npu_runner.SAFE_SINGLE_PROMPT_TOKENS:
            return TranslationPlan(
                text=text,
                source_lang=src,
                target_lang=tgt,
                form=frm,
                context=ctx,
                glossary=glossary,
                requested_max_new_tokens=req_max,
                full_prompt_tokens=len(full_ids),
                single_input_ids=full_ids,
            )

        chunks, trailing_sep = npu_runner.split_text_into_chunks(text, self.tokenizer)
        if not chunks:
            chunks = [("", text)]
            trailing_sep = ""
        first_only = npu_runner.encode_prompt(
            self.tokenizer,
            chunks[0][1],
            src,
            tgt,
            form=frm,
            context=ctx,
            glossary=glossary,
        )
        if len(first_only) >= npu_runner.SAFE_TOTAL_SEQ_LEN - 32:
            raise ValueError(
                f"コンテキスト・用語集を含めたプロンプト ({len(first_only)} tok) が上限 ({npu_runner.SAFE_TOTAL_SEQ_LEN}) に近すぎます。"
            )
        return TranslationPlan(
            text=text,
            source_lang=src,
            target_lang=tgt,
            form=frm,
            context=ctx,
            glossary=glossary,
            requested_max_new_tokens=req_max,
            full_prompt_tokens=len(full_ids),
            single_input_ids=None,
            chunks=chunks,
            trailing_sep=trailing_sep,
        )

    def _run_stream_chunk(
        self,
        req_id: int,
        cancel_event: threading.Event,
        chunk_idx: int,
        total_chunks: int,
        leading_sep: str,
        chunk_input_ids: list[int],
        allowed_new: int,
        totals: dict,
        emit_fn,
    ) -> tuple[bool, str, int, float, int]:
        """Run one sequence-safe prefill+decode pass on the NPU and emit SSE token deltas."""
        if self.last_token_attn_len is not None and abs(len(chunk_input_ids) - self.last_token_attn_len) <= 1:
            self.last_token_attn_len = npu_runner.reset_token_kv_state(self.model)

        params = og.GeneratorParams(self.model)
        params.set_search_options(do_sample=False, max_length=len(chunk_input_ids) + allowed_new)
        gen = og.Generator(self.model, params)

        step_ms: list[float] = []
        chunk_tok_ids: list[int] = []
        sent_len = 0
        decoded_full = ""
        rep_trim = 0
        aborted = False
        prefill_ms = 0.0
        t_dec_0 = time.perf_counter()
        t_dec_1 = t_dec_0

        try:
            t_prefill_0 = time.perf_counter()
            gen.append_tokens(chunk_input_ids)
            prefill_ms = (time.perf_counter() - t_prefill_0) * 1000.0
            totals["prompt_tokens"] += len(chunk_input_ids)
            totals["prefill_ms"] += prefill_ms

            emit_fn(
                "meta",
                {
                    "request_id": req_id,
                    "chunk_index": chunk_idx + 1,
                    "chunks_total": total_chunks,
                    "source_tokens": totals["source_tokens"],
                    "prompt_tokens": totals["prompt_tokens"],
                    "prefill_ms": round(totals["prefill_ms"], 2),
                    "output_tokens": totals["output_tokens"],
                },
            )
            if leading_sep:
                cur_dec_s = max(totals["decode_s"], 1e-9)
                steady = totals["steady_step_ms"]
                steady_tps = (1000.0 / (sum(steady) / len(steady))) if steady else 0.0
                emit_fn(
                    "token",
                    {
                        "delta": leading_sep,
                        "output_tokens": totals["output_tokens"],
                        "tok_per_s": round(totals["generated_tokens"] / cur_dec_s, 2) if totals["generated_tokens"] else 0.0,
                        "steady_tok_per_s": round(steady_tps, 2),
                    },
                )

            t_dec_0 = time.perf_counter()
            while not gen.is_done():
                if cancel_event.is_set() or not self.is_latest(req_id):
                    aborted = True
                    break
                ts = time.perf_counter()
                gen.generate_next_token()
                te = time.perf_counter()
                dt_ms = (te - ts) * 1000.0
                step_ms.append(dt_ms)
                if len(step_ms) > 1:
                    totals["steady_step_ms"].append(dt_ms)

                next_token = int(gen.get_next_tokens()[0])
                if next_token in npu_runner.EOS_IDS:
                    break
                chunk_tok_ids.append(next_token)

                if len(chunk_tok_ids) >= 16:
                    rep = npu_runner.has_tail_repetition(chunk_tok_ids)
                    if rep:
                        rep_trim = rep
                        break

                decoded_full = self.tokenizer.decode(chunk_tok_ids, skip_special_tokens=True)
                safe_text = decoded_full.rstrip("\ufffd")
                if len(safe_text) > sent_len:
                    delta = safe_text[sent_len:]
                    sent_len = len(safe_text)
                    cur_dec_s = max(totals["decode_s"] + (te - t_dec_0), 1e-9)
                    cur_gen = totals["generated_tokens"] + len(step_ms)
                    cur_out = totals["output_tokens"] + len(chunk_tok_ids)
                    steady = totals["steady_step_ms"] or step_ms
                    steady_tps = 1000.0 / (sum(steady) / len(steady))
                    emit_fn(
                        "token",
                        {
                            "delta": delta,
                            "output_tokens": cur_out,
                            "tok_per_s": round(cur_gen / cur_dec_s, 2),
                            "steady_tok_per_s": round(steady_tps, 2),
                        },
                    )
            t_dec_1 = time.perf_counter()
        finally:
            if len(step_ms) >= 2:
                self.last_token_attn_len = len(chunk_input_ids) + len(step_ms) - 1
            del gen, params

        totals["decode_s"] += max(t_dec_1 - t_dec_0, 0.0)
        totals["generated_tokens"] += len(step_ms)

        if aborted:
            return True, "", 0, prefill_ms, 0

        if not rep_trim and len(decoded_full) > sent_len:
            cur_dec_s = max(totals["decode_s"], 1e-9)
            steady = totals["steady_step_ms"] or step_ms
            steady_tps = (1000.0 / (sum(steady) / len(steady))) if steady else 0.0
            emit_fn(
                "token",
                {
                    "delta": decoded_full[sent_len:],
                    "output_tokens": totals["output_tokens"] + len(chunk_tok_ids),
                    "tok_per_s": round(totals["generated_tokens"] / cur_dec_s, 2),
                    "steady_tok_per_s": round(steady_tps, 2),
                },
            )

        clean_ids = chunk_tok_ids[:-rep_trim] if rep_trim and len(chunk_tok_ids) > rep_trim else chunk_tok_ids
        clean_text = self.tokenizer.decode(clean_ids, skip_special_tokens=True) if rep_trim else decoded_full
        totals["output_tokens"] += len(clean_ids)
        return False, clean_text, rep_trim, prefill_ms, len(clean_ids)

    def stream_worker(
        self,
        req_id: int,
        cancel_event: threading.Event,
        plan: TranslationPlan,
        emit_fn,
    ) -> None:
        """Execute single-pass or chunked NPU translation in the worker thread and emit SSE events."""
        t_wall_0 = time.perf_counter()
        is_chunked = plan.single_input_ids is None
        total_chunks = len(plan.chunks) if is_chunked else 1
        mode = "chunked" if is_chunked else "single"
        print(
            f"[WebUI req={req_id}] start mode={mode} src={plan.source_lang or 'auto'} tgt={plan.target_lang} "
            f"chars={len(plan.text)} source_tokens={plan.full_prompt_tokens} chunks={total_chunks}",
            flush=True,
        )
        try:
            totals: dict = {
                "source_tokens": plan.full_prompt_tokens,
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "output_tokens": 0,
                "prefill_ms": 0.0,
                "decode_s": 0.0,
                "steady_step_ms": [],
            }
            out_parts: list[str] = []
            history_pair: list[dict[str, str]] | None = None

            work_items = plan.chunks if is_chunked else [("", plan.text)]
            for idx, (sep, body) in enumerate(work_items):
                if cancel_event.is_set() or not self.is_latest(req_id):
                    print(f"[WebUI req={req_id}] cancelled", flush=True)
                    emit_fn("cancelled", {})
                    return

                if is_chunked:
                    chunk_ids = npu_runner.build_chunk_prompt_ids(
                        self.tokenizer,
                        body,
                        history_pair,
                        plan.source_lang,
                        plan.target_lang,
                        form=plan.form,
                        context=plan.context,
                        glossary=plan.glossary,
                    )
                else:
                    assert plan.single_input_ids is not None
                    chunk_ids = plan.single_input_ids

                allowed_new = min(
                    plan.requested_max_new_tokens,
                    max(16, npu_runner.SAFE_TOTAL_SEQ_LEN - len(chunk_ids)),
                )
                aborted, clean_text, rep_trim, prefill_ms, out_len = self._run_stream_chunk(
                    req_id=req_id,
                    cancel_event=cancel_event,
                    chunk_idx=idx,
                    total_chunks=total_chunks,
                    leading_sep=sep,
                    chunk_input_ids=chunk_ids,
                    allowed_new=allowed_new,
                    totals=totals,
                    emit_fn=emit_fn,
                )
                if aborted:
                    print(f"[WebUI req={req_id}] cancelled", flush=True)
                    emit_fn("cancelled", {})
                    return

                if is_chunked or rep_trim:
                    print(
                        f"[WebUI req={req_id}] chunk {idx + 1}/{total_chunks} "
                        f"prompt={len(chunk_ids)} out={out_len} prefill={prefill_ms:.1f}ms rep_trim={rep_trim}",
                        flush=True,
                    )
                out_parts.append(sep + clean_text)
                if is_chunked:
                    history_pair = [{"role": "user", "content": body}, {"role": "assistant", "content": clean_text}]

            decode_s = max(totals["decode_s"], 1e-9)
            steady = totals["steady_step_ms"]
            steady_tps = (1000.0 / (sum(steady) / len(steady))) if steady else (totals["generated_tokens"] / decode_s)
            tok_per_s = totals["generated_tokens"] / decode_s

            if plan.trailing_sep:
                emit_fn(
                    "token",
                    {
                        "delta": plan.trailing_sep,
                        "output_tokens": totals["output_tokens"],
                        "tok_per_s": round(tok_per_s, 2),
                        "steady_tok_per_s": round(steady_tps, 2),
                    },
                )

            full_translation = "".join(out_parts) + plan.trailing_sep
            wall_s = time.perf_counter() - t_wall_0
            print(
                f"[WebUI req={req_id}] done chunks={total_chunks} prompt={totals['prompt_tokens']} "
                f"gen={totals['generated_tokens']} out={totals['output_tokens']} "
                f"prefill={totals['prefill_ms']:.1f}ms decode={decode_s * 1000.0:.1f}ms "
                f"steady={steady_tps:.1f} tok/s wall={wall_s:.2f}s",
                flush=True,
            )
            emit_fn(
                "done",
                {
                    "translation": full_translation,
                    "chunks_total": total_chunks,
                    "source_tokens": plan.full_prompt_tokens,
                    "prompt_tokens": totals["prompt_tokens"],
                    "generated_tokens": totals["generated_tokens"],
                    "output_tokens": totals["output_tokens"],
                    "prefill_ms": round(totals["prefill_ms"], 2),
                    "decode_ms": round(decode_s * 1000.0, 2),
                    "tok_per_s": round(tok_per_s, 2),
                    "steady_tok_per_s": round(steady_tps, 2),
                },
            )
        except Exception as exc:
            print(f"[WebUI req={req_id}] error: {exc}", flush=True)
            emit_fn("error", {"message": str(exc)})
        finally:
            emit_fn(None, None)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        del self.model


def parse_glossary(raw_glossary) -> list[dict[str, str]] | None:
    """Parse structured list or newline-separated 'source -> target' glossary entries."""
    if not raw_glossary:
        return None
    if isinstance(raw_glossary, list):
        items = []
        for entry in raw_glossary:
            if isinstance(entry, dict):
                s = str(entry.get("source", "")).strip()
                t = str(entry.get("target", "")).strip()
                if s and t:
                    items.append({"source": s, "target": t})
        return items or None
    if isinstance(raw_glossary, str):
        items = []
        for line in raw_glossary.splitlines():
            line = line.strip()
            if not line:
                continue
            for sep in ("->", "=>", "\t", "="):
                if sep in line:
                    left, right = line.split(sep, 1)
                    s, t = left.strip(), right.strip()
                    if s and t:
                        items.append({"source": s, "target": t})
                    break
        return items or None
    return None


def sse_format(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEBUI_DIR / "index.html")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "model_loaded": True, "device": "AMD Ryzen AI NPU (NPU Strix)"})


async def handle_info(request: web.Request) -> web.Response:
    engine: NPUTranslatorEngine = request.app["engine"]
    return web.json_response(
        {
            "model": "PLaMo-3 Translate 0.8B (INT4 Full-Fusion)",
            "backend": "AMD Ryzen AI NPU (DynamicDispatch)",
            "model_dir": str(engine.model_dir),
            "max_seq_len": MAX_SEQ_LEN,
            "safe_single_prompt_tokens": npu_runner.SAFE_SINGLE_PROMPT_TOKENS,
            "chunked_long_text": True,
            "languages": LANGUAGES,
        }
    )


def _validate_request_body(engine: NPUTranslatorEngine, data: dict) -> TranslationPlan:
    text = str(data.get("text", ""))
    source_lang = data.get("source_lang", "auto")
    target_lang = str(data.get("target_lang", "ja"))
    form = data.get("form")
    context = data.get("context")
    glossary = parse_glossary(data.get("glossary"))
    max_new_tokens = int(data.get("max_new_tokens", 512))
    return engine.prepare_translation_plan(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        form=form,
        context=context,
        glossary=glossary,
        requested_max_new_tokens=max_new_tokens,
    )


async def handle_translate_stream(request: web.Request) -> web.StreamResponse:
    engine: NPUTranslatorEngine = request.app["engine"]
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    try:
        plan = _validate_request_body(engine, data)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    if not plan.text.strip():
        await response.write(
            sse_format(
                "done",
                {
                    "translation": "",
                    "chunks_total": 0,
                    "source_tokens": 0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "output_tokens": 0,
                    "prefill_ms": 0.0,
                    "decode_ms": 0.0,
                    "tok_per_s": 0.0,
                    "steady_tok_per_s": 0.0,
                },
            )
        )
        await response.write_eof()
        return response

    req_id, cancel_event = engine.start_request()
    loop = asyncio.get_running_loop()
    async_queue: asyncio.Queue = asyncio.Queue()

    def emit(ev_name, payload) -> None:
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(async_queue.put_nowait, (ev_name, payload))
            except RuntimeError:
                pass

    future = loop.run_in_executor(
        engine.executor,
        engine.stream_worker,
        req_id,
        cancel_event,
        plan,
        emit,
    )

    try:
        while True:
            ev_name, payload = await async_queue.get()
            if ev_name is None:
                break
            await response.write(sse_format(ev_name, payload))
            if ev_name in ("done", "error", "cancelled"):
                break
    except (asyncio.CancelledError, ConnectionResetError):
        cancel_event.set()
        return response
    finally:
        if not future.done() and not cancel_event.is_set():
            await asyncio.wrap_future(future)
    try:
        await response.write_eof()
    except Exception:
        pass
    return response


async def handle_translate_sync(request: web.Request) -> web.Response:
    engine: NPUTranslatorEngine = request.app["engine"]
    try:
        data = await request.json()
        plan = _validate_request_body(engine, data)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    if not plan.text.strip():
        return web.json_response({"translation": "", "chunks_total": 0, "prompt_tokens": 0, "output_tokens": 0})

    req_id, cancel_event = engine.start_request()
    ev_queue: queue.Queue = queue.Queue()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        engine.executor,
        engine.stream_worker,
        req_id,
        cancel_event,
        plan,
        lambda ev, pl: ev_queue.put((ev, pl)),
    )
    while not ev_queue.empty():
        ev_name, payload = ev_queue.get()
        if ev_name == "done":
            return web.json_response(payload)
        if ev_name == "error":
            return web.json_response(payload, status=500)
        if ev_name == "cancelled":
            return web.json_response({"error": "Cancelled by newer request"}, status=409)
    return web.json_response({"error": "Unexpected empty worker response"}, status=500)


def create_app(engine: NPUTranslatorEngine) -> web.Application:
    app = web.Application()
    app["engine"] = engine
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/info", handle_info)
    app.router.add_post("/api/translate/stream", handle_translate_stream)
    app.router.add_post("/api/translate", handle_translate_sync)

    async def on_cleanup(_app: web.Application) -> None:
        engine.close()

    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Google-Translate-style WebUI for PLaMo-3 Translate 0.8B on AMD Ryzen AI NPU")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model_dir", type=Path, default=npu_runner.DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer_dir", type=Path, default=npu_runner.DEFAULT_TOKENIZER_DIR)
    args = parser.parse_args()

    engine = NPUTranslatorEngine(args.model_dir, args.tokenizer_dir)
    app = create_app(engine)
    print(f"[WebUI] Listening on http://{args.host}:{args.port}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
