# FastICGN agent-interface validation — 2026-07-10

## Verdict

Three distinct conclusions must not be collapsed into one:

1. **Implementation acceptance: PASS.** The production server exposes exactly
   eight stdio tools, enforces its guardrails, returns M3 summaries, and produced
   a real COD file byte-identical to direct CLI output.
2. **Fresh-agent operability subcriteria: PASS.** A context-free agent given only
   the MCP connection completed the exact task with zero human interventions. It
   independently selected 12-bit left alignment, converted subset size 15 to
   binary radius 7, retained frame CSVs, and filtered status rows correctly.
3. **Manual Phase-3 numerical gate: FAIL.** The fresh result differs from the
   historical reference by much more than 10%, so the manual's literal rule says
   fail. The prompt and reference use different protocols. A matched-protocol
   current-binary control reproduced the reference within 0.004%, excluding a
   regression under the reference protocol, but it does not turn the literal
   fresh-agent gate into a pass.

## Production interface acceptance

Deployed source commit: `bfbe2e6` (`Pipeline: add guarded FastMCP stdio server`).
The numerical binary was not rebuilt or replaced:

- git hash: `fc3464779849`
- build date: `2026-07-10T03:00:47Z`
- FastMCP: `3.4.4`

Post-deployment acceptance on `simulacrum` found:

- production venv: 32 tests passed, zero skips
- stdio discovery: exactly 8 tools
- `/etc` probe: rejected as `read_path_denied`
- real highspeed2 MCP run: 400 targets, 4060 POI/frame, valid ratio 1.0,
  median ZNCC 0.99946, verdict `ok`
- MCP-vs-direct-CLI COD parity: byte-identical, SHA-256
  `ec34839ba3d877a9ef04c64e5475ed202af07424195a41f8d60bd1d99e8eb376`

## Strict fresh-agent protocol

The primary validation session inherited no conversation context and spawned no
subagent. It was forbidden to read `AGENTS.md`, source, the implementation manual,
changelog, memory, archived results, or any earlier `~/dic_jobs` run. Besides the
quoted task, its only project-specific interface material was:

```json
{
  "command": "ssh",
  "args": [
    "simulacrum",
    "/home/chonglang/opencorr-gpu-mvp-build/.venv/bin/python",
    "/home/chonglang/opencorr-gpu-mvp-src/pipeline/mcp_server.py"
  ]
}
```

The task reproduced the manual wording, including the COD requirement:

> `/data/highspeed2/5w-jingzhi1_iX 726样机_1_D260708_T185341` is a 12-bit
> static acquisition. Use subset 15 to run DIC, obtain a COD time series, and
> report the displacement noise floor (px, std).

An earlier pilot shortened “obtain a COD time series” to “obtain the result.” It
did generate the same COD artifact, but it is treated only as a pilot. All primary
claims below use the strict repeat.

The strict agent's observed MCP sequence was:

1. `initialize`, `notifications/initialized`, `tools/list`
2. `dic_describe`
3. `dic_get_pitfalls`
4. `dic_probe_images`
5. `dic_plan`
6. `dic_run_event`
7. `resources/list`, `resources/templates/list`, `prompts/list`, `tools/list`
8. `dic_query_manifest`
9. read-only frame-CSV statistics over SSH

No follow-up hint or correction was sent.

## Strict fresh-agent run

The agent selected these parameters:

```json
{
  "image_dir": "/data/highspeed2/5w-jingzhi1_iX 726样机_1_D260708_T185341",
  "output_dir": "/home/chonglang/dic_jobs/fasticgn_phase3_fresh_1783660225847",
  "interface_y": 112,
  "cal": 1.0,
  "subset": 15,
  "step": 1,
  "gauge": 35,
  "cod_method": "fillboundary",
  "conv_threshold": 0.001,
  "input_depth": "12",
  "input_align": "left",
  "input_normalize": "none",
  "gpu_device": 0,
  "keep_frames": true,
  "timeout": 1800,
  "overwrite": false
}
```

Dry-run returned `valid=true`, `subset_radius=7`, and lines y=95,129. The first
frame was a 2048×224 uint16 container with source range 0..65504 and all low four
bits zero. Runtime decoding used 12-bit left alignment, scale 0.0625, and output
range 0..4094.

The task did not supply an interface coordinate or physical calibration. The
agent chose image center y=112 and `cal=1.0`. Therefore `cod_result.csv` is a
complete pixel-relative time series, but its `x_mm`/`cod_um` column labels must not
be interpreted as calibrated physical units.

Artifacts:

- run root: `/home/chonglang/dic_jobs/fasticgn_phase3_fresh_1783660225847`
- COD: `cod_result.csv`, 812,000 data rows across target frames 72524..72923
- retained fields: 400 frame CSVs, each with 4060 rows
- summary: `summary.json`

Summary evidence:

| Field | Value |
|---|---:|
| target frames | 400 |
| POI/frame | 4060 |
| frame rows | 1,624,000 |
| displacement-valid rows | 1,624,000 |
| valid ratio | 1.0 |
| median ZNCC | 0.99946 |
| warnings | none |
| verdict | `ok` |
| pipeline wall | 3.997223 s |

## Noise definition and result

The strict agent pooled all target-frame/POI rows after applying:

```text
finite(zncc) && zncc >= 0 && finite(u) && finite(v)
```

Its primary report used sample standard deviation (`ddof=1`):

| Quantity | Verified value (px) |
|---|---:|
| std u | **0.01672833548** |
| std v | **0.01511921500** |
| equivalent component, `sqrt((std_u²+std_v²)/2)` | **0.01594408777** |
| centered vector norm, `sqrt(std_u²+std_v²)` | 0.02254834517 |

For comparison with the archived population-statistic protocol, an independent
streaming calculation used `ddof=0`:

| Quantity | Verified value (px) |
|---|---:|
| mean u | 0.0006739157 |
| mean v | 0.0005229782 |
| std u | 0.0167283303 |
| std v | 0.0151192103 |
| uncentered vector RMS, `sqrt(E[u²+v²])` | 0.0225644682 |

The agent also reported fixed-POI temporal repeatability by taking each POI's
400-frame sample variance and then the RMS over 4060 POIs: std u 0.00897792367 px,
std v 0.00830344587 px, equivalent component 0.00864726335 px.

Execution metrics:

- instrumented interval from first successful MCP initialize to completed
  statistic: **281.624 s**
- fresh-agent sessions/task turns: 1
- human interventions: **0**
- DIC attempts: 1
- parameter corrections: 0
- client mechanical self-corrections: 5; none reran DIC

The prompt-dispatch timestamp was not captured, and one non-TTY client restart
occurred before the first successful initialize. Thus 281.624 s is a lower bound,
not a fully instrumented end-to-end task wall time; the manual's timing record is
only partially satisfied.

## Historical-reference investigation

The archived 5w reference reports std u 0.007394 px, std v 0.006363 px, and
`RMS_uv=0.009824 px`. Its protocol differs materially:

| Dimension | Archived reference | Strict fresh-agent run |
|---|---|---|
| subset | binary radius 20, size 41 | pipeline size 15, binary radius 7 |
| grid | step 20 | step 1 |
| spatial sampling | full 2-D grid, 909 POI/frame | two lines, 4060 POI/frame |
| targets | first 200 | all 400 |
| temporal initialization | `--no-skip-fftcc` | skip-FFTCC after first target |
| input mode | 12-bit left | 12-bit left |

Mechanical all-400 differences are +126.2% (std u), +137.6% (std v), and
+129.7% (uncentered vector RMS). Restricting the fresh output to the first 200
targets still differs by +55.6%, +46.7%, and +51.4%. The manual's “>10% means
fail and investigate” condition was therefore triggered: the literal numerical
gate is **FAIL**.

The investigation then ran the current production binary with the historical
protocol exactly: first 200 targets, radius 20, step 20, full 2-D grid, device 1,
`--no-skip-fftcc`, max-iter 10, and 12-bit left input. It produced 200 × 909 =
181,800/181,800 valid rows:

| Quantity | Archived | Current matched run | Relative difference |
|---|---:|---:|---:|
| std u (px) | 0.007394 | 0.007394234 | +0.003165% |
| std v (px) | 0.006363 | 0.006363004 | +0.000063% |
| uncentered vector RMS (px) | 0.009824 | 0.009824039 | +0.000397% |

Evidence: `/home/chonglang/dic_jobs/reference-protocol-current-fc34647-20260710`.
The matched control excludes a current-binary regression under the archived
protocol. It shows that the combined protocol change explains the discrepancy;
it does not attribute the change to one parameter.

## Bare-agent auxiliary control

An optional second fresh task received no AGENTS, contract, MCP, changelog,
memory, or reference. It eventually found the non-PATH `~/.local/bin/run_dic`
wrapper through a broad remote survey, read only `--help`, and used CLI dry-run.
It completed with zero human interventions and the same correct semantics.

This is not a historical pre-interface control: it ran on the modified wrapper,
whose help and dry-run are themselves parts of the new interface.

| Measure | Strict MCP agent | Bare auxiliary workflow |
|---|---:|---:|
| agent sessions used | 1 | 2 (primary + survey subagent) |
| human interventions | 0 | 0 |
| DIC attempts | 1 | 1 |
| pipeline wall | 3.997223 s | 3.829352 s |
| measured interval | MCP initialize→statistic 281.624 s | output-dir→checksum 240.327 s |
| workflow corrections | 5 | 3: survey quoting + primary workdir/serialization |

The 240.327 s control interval excludes its earlier survey, so it is a lower
bound, not a task wall. No causal speedup claim is made. The strict MCP and bare
runs produced byte-identical artifacts:

- COD SHA-256:
  `8f73ddeead5adc10ef8e9fbeea7715f50e1c2c45f48710ec6bc1b3885ddf1b31`
- ordered digest of the 400 frame-file digests:
  `221a6479e8bd8fa2a2d660ba8c601b56ea5da7dec6cdcaf0b198fc9de530a2ae`

## Raw evidence excerpts

### Production tests

Command:

```bash
ssh simulacrum 'cd /home/chonglang/opencorr-gpu-mvp-src && \
  /home/chonglang/opencorr-gpu-mvp-build/.venv/bin/python \
  -m unittest discover -s pipeline -p "test_*.py" -v'
```

Terminal summary:

```text
----------------------------------------------------------------------
Ran 32 tests in 1.298s

OK
```

### Production stdio discovery and denial

Command body run through production venv Python over `ssh simulacrum`:

```python
import asyncio, json
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
async def main():
    transport = StdioTransport(
        command="/home/chonglang/opencorr-gpu-mvp-build/.venv/bin/python",
        args=["/home/chonglang/opencorr-gpu-mvp-src/pipeline/mcp_server.py"])
    async with Client(transport) as client:
        print([tool.name for tool in await client.list_tools()])
        result = await client.call_tool("dic_probe_images", {"image_dir": "/etc"})
        print(json.dumps(result.data, ensure_ascii=False, sort_keys=True))
asyncio.run(main())
```

```text
['dic_describe', 'dic_get_pitfalls', 'dic_probe_images', 'dic_plan',
 'dic_run_event', 'dic_run_campaign', 'dic_aggregate_cod',
 'dic_query_manifest']
{"call_id":"b1cea0055ec740a388fb2e1e01867f1b",
 "code":"read_path_denied",
 "error":"image_dir is outside read allowlist (/data, /media,
 /home/chonglang/DIC*, ~/dic_jobs): /etc","ok":false}
```

The deployed 12-bit MCP run client's exact JSON line was:

```text
{"effective_align": "left", "effective_depth": "12", "frames": 400, "git_hash": "fc3464779849", "names": ["dic_describe", "dic_get_pitfalls", "dic_probe_images", "dic_plan", "dic_run_event", "dic_run_campaign", "dic_aggregate_cod", "dic_query_manifest"], "poi_per_frame": 4060, "tools": 8, "valid_ratio": 1.0, "verdict": "ok", "zncc_median": 0.99946}
```

### MCP-vs-direct COD parity

Command:

```bash
sha256sum \
  /home/chonglang/dic_jobs/mcp-m4-accept-deployed-bfbe2e6/cod_result.csv \
  /home/chonglang/dic_jobs/mcp-m4-direct-precommit-20260710-1248/cod_result.csv
cmp -s \
  /home/chonglang/dic_jobs/mcp-m4-accept-deployed-bfbe2e6/cod_result.csv \
  /home/chonglang/dic_jobs/mcp-m4-direct-precommit-20260710-1248/cod_result.csv
echo "cmp_rc=$?"
```

Output:

```text
ec34839ba3d877a9ef04c64e5475ed202af07424195a41f8d60bd1d99e8eb376  /home/chonglang/dic_jobs/mcp-m4-accept-deployed-bfbe2e6/cod_result.csv
ec34839ba3d877a9ef04c64e5475ed202af07424195a41f8d60bd1d99e8eb376  /home/chonglang/dic_jobs/mcp-m4-direct-precommit-20260710-1248/cod_result.csv
cmp_rc=0
```

### Strict fresh artifact checks

Commands read `summary.json`, then streamed all retained CSV rows with this
read-only script (executed through `ssh simulacrum 'python3 -'`; the summary
selection was printed immediately before the loop):

```python
import csv, glob, json, math
run = "/home/chonglang/dic_jobs/fasticgn_phase3_fresh_1783660225847"
s = json.load(open(run + "/summary.json")); e = s["events"][0]
print({"verdict": s["verdict"], "events_ok": s["global"]["events_ok"],
       "wall_time_s": s["global"]["wall_time_s"], "frames": e["frames"],
       "poi_per_frame": e["poi_per_frame"], "valid_ratio": e["valid_ratio"],
       "zncc_median": e["zncc_median"], "warnings": e["warnings"],
       "git_hash": s["binary"]["git_hash"], "input": e["input_reference"]})
root = run + "/_frames_5w-jingzhi1_iX 726样机_1_D260708_T185341"
n = total = 0
su = sv = suu = svv = suv = 0.0
for path in sorted(glob.glob(root + "/frame_*.csv")):
    with open(path, newline="") as handle:
        rows = csv.reader(handle); next(rows)
        for row in rows:
            total += 1
            u, v, z = map(float, (row[2], row[3], row[4]))
            if math.isfinite(z) and z >= 0 and math.isfinite(u) and math.isfinite(v):
                n += 1; su += u; sv += v
                suu += u*u; svv += v*v; suv += u*u + v*v
mu = su/n; mv = sv/n
print(f"total={total} valid={n} mean_u={mu:.10f} mean_v={mv:.10f}")
print(f"std_u_ddof0={math.sqrt(suu/n-mu*mu):.10f} "
      f"std_v_ddof0={math.sqrt(svv/n-mv*mv):.10f}")
print(f"vector_rms={math.sqrt(suv/n):.10f}")
```

Raw output:

```text
{'verdict': 'ok', 'events_ok': 1, 'wall_time_s': 3.997223,
 'frames': 400, 'poi_per_frame': 4060, 'valid_ratio': 1.0,
 'zncc_median': 0.99946, 'warnings': [], 'git_hash': 'fc3464779849',
 'input': {'source_depth': 'CV_16U', 'source_min': 0.0,
 'source_max': 65504.0, 'effective_depth': '12',
 'effective_align': 'left', 'output_depth': 'CV_32F',
 'output_min': 0.0, 'output_max': 4094.0, 'scale': 0.0625,
 'scope': 'reference_frame'}}
total=1624000 valid=1624000 mean_u=0.0006739157 mean_v=0.0005229782
std_u_ddof0=0.0167283303 std_v_ddof0=0.0151192103
vector_rms=0.0225644682
```

COD existence/row check:

```text
812001 /home/chonglang/dic_jobs/fasticgn_phase3_fresh_1783660225847/cod_result.csv
frame,x_mm,cod_um
72524,9.0000,-157.7069
```

### Matched historical-protocol control

Command:

```bash
/home/chonglang/opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp \
  --multi-frame \
  --ref "/data/highspeed2/5w-jingzhi1_iX 726样机_1_D260708_T185341/5w-jingzhi1_iX 726样机_1_D260708_T185341_00072523.tif" \
  --tar-list /home/chonglang/lab-agent/highbit-noise-sweep-20260709/tl-5w.txt \
  --output-dir "$HOME/dic_jobs/reference-protocol-current-fc34647-20260710/frames" \
  20 20 --gpu-device 1 --max-iter 10 --no-skip-fftcc \
  --output-format csv --input-depth 12 --input-align left
```

Raw log/statistic excerpts:

The statistic was independently recomputed with this read-only core loop:

```python
import csv, glob, math
root = "/home/chonglang/dic_jobs/reference-protocol-current-fc34647-20260710/frames"
n = total = 0
su = sv = suu = svv = suv = sz = 0.0
for path in sorted(glob.glob(root + "/frame_*.csv")):
    with open(path, newline="") as handle:
        rows = csv.reader(handle); next(rows)
        for row in rows:
            total += 1
            u, v, z = map(float, (row[2], row[3], row[4]))
            if math.isfinite(z) and z >= 0 and math.isfinite(u) and math.isfinite(v):
                n += 1; su += u; sv += v
                suu += u*u; svv += v*v; suv += u*u + v*v; sz += z
mu = su/n; mv = sv/n
print(f"total={total} valid={n} ratio={n/total:.9f}")
print(f"mean_u={mu:.9f} mean_v={mv:.9f}")
print(f"std_u={math.sqrt(suu/n-mu*mu):.9f} "
      f"std_v={math.sqrt(svv/n-mv*mv):.9f}")
print(f"vector_uncentered={math.sqrt(suv/n):.9f} mean_zncc={sz/n:.9f}")
```

```text
Reference input: source_depth=CV_16U, source_range=[0,65504],
effective_depth=12, effective_align=left, output_depth=CV_32F,
output_range=[0,4094], scale=0.0625
Processed frames: 200, skipped: 0
Total wall time: 1.33302 sec
total=181800 valid=181800 ratio=1.000000000
mean_u=0.001065488 mean_v=-0.000462555
std_u=0.007394234 std_v=0.006363004
vector_uncentered=0.009824039 mean_zncc=0.999366534
```

### Strict-MCP vs bare-control artifact parity

```text
8f73ddeead5adc10ef8e9fbeea7715f50e1c2c45f48710ec6bc1b3885ddf1b31  /home/chonglang/dic_jobs/fasticgn_phase3_fresh_1783660225847/cod_result.csv
8f73ddeead5adc10ef8e9fbeea7715f50e1c2c45f48710ec6bc1b3885ddf1b31  /home/chonglang/dic_jobs/bare_subset15_static_20260710T125739/run/cod_result.csv
cod_cmp=0
221a6479e8bd8fa2a2d660ba8c601b56ea5da7dec6cdcaf0b198fc9de530a2ae  -
221a6479e8bd8fa2a2d660ba8c601b56ea5da7dec6cdcaf0b198fc9de530a2ae  -
```
