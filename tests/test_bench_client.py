"""Check bench/vllm_bench.py's measurements against a fake streaming server whose timing
is known: 0.2 s to first token, then 9 more tokens at 0.05 s each."""
import asyncio
import json

from aiohttp import web

from bench.vllm_bench import run_level

FIRST, STEP, TOKENS = 0.2, 0.05, 10


def make_app(state):
    async def completions(request):
        body = await request.json()
        if body["prompt"] == "boom":
            return web.Response(status=500, text="CUDA out of memory")
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        resp = web.StreamResponse()
        await resp.prepare(request)
        for i in range(TOKENS):
            await asyncio.sleep(FIRST if i == 0 else STEP)
            chunk = {"choices": [{"text": "x"}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {"choices": [], "usage": {"completion_tokens": TOKENS}}
        await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        state["inflight"] -= 1
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    return app


async def with_server(prompts, concurrency):
    state = {"inflight": 0, "peak": 0}
    runner = web.AppRunner(make_app(state))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        lvl = await run_level(f"http://127.0.0.1:{port}", "m", prompts, concurrency, TOKENS)
    finally:
        await runner.cleanup()
    return lvl, state


def test_known_timings_and_concurrency_cap():
    lvl, state = asyncio.run(with_server(["p"] * 8, concurrency=4))
    assert lvl["succeeded"] == 8 and lvl["failed"] == 0
    assert state["peak"] == 4                       # closed loop never exceeds the cap
    assert abs(lvl["ttft_s"]["mean"] - FIRST) < 0.05
    expected_latency = FIRST + (TOKENS - 1) * STEP
    assert abs(lvl["latency_s"]["mean"] - expected_latency) < 0.1
    assert abs(lvl["tpot_s"]["mean"] - STEP) < 0.02
    # 2 waves of 4 requests, each ~expected_latency long, 10 tokens each
    assert abs(lvl["output_tokens_per_s"] - 80 / (2 * expected_latency)) < 8


def test_http_errors_are_counted_not_hidden():
    lvl, _ = asyncio.run(with_server(["p", "boom", "p"], concurrency=2))
    assert lvl["succeeded"] == 2 and lvl["failed"] == 1
    assert "500" in lvl["first_errors"][0] and "out of memory" in lvl["first_errors"][0]


def test_summary_lists_unserved_variants():
    from bench.summarize import build_table
    lvl = {"concurrency": 1, "requests": 4, "failed": 0, "output_tokens_per_s": 10.0,
           "ttft_s": {"p50": 0.1, "p95": 0.2}, "latency_s": {"p50": 1.0, "p95": 2.0}}
    recs = [{"variant": "a", "levels": [lvl]},
            {"variant": "b", "failed_to_start": True, "server_status": "server exited with code 1",
             "server_log_tail": ["INFO start", "ValueError: unsupported quantization"]}]
    out = build_table(recs)
    assert "| a | 1 |" in out and "**b**" in out and "unsupported quantization" in out
