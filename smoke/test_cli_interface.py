#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""End-to-end tests of the CLI's own interface: --help, --version, the closing
stdout status line, --quiet, the input bit-depth warnings, the resume contract
without scheduling options, the rejection messages and --verify-only.

Synthetic inputs only; all evidence is kept in a unique directory.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import tempfile
import time
import unittest
import zlib

from generate_synthetic_dic import make_base, write_pgm

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--fault-lib', type=Path)
parser.add_argument('--work-dir', type=Path)
parser.add_argument('--expect-version')
parser.add_argument('--expect-backend', choices=('cuda', 'cpu'))
parser.add_argument('--expect-fftw', choices=('0', '1'))
args = parser.parse_args()
args.binary = args.binary.resolve()
if args.fault_lib: args.fault_lib = args.fault_lib.resolve()
ROOT = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix='fasticgn-cli-interface-')).resolve()
if args.work_dir: ROOT.mkdir(parents=True, exist_ok=False)
print('Evidence:', ROOT, flush=True)

SOURCE = Path(__file__).resolve().parents[1] / 'opencorr_cli_cuda_mvp.cpp'
WIDTH, HEIGHT = 192, 128
SCHEDULING_OPTIONS = ('gpu_device', 'threads', 'preload', 'decode_threads')
STATUS_SCHEMA = 'fasticgn.run-status/1'
VERIFY_SCHEMA = 'fasticgn.verify-only/1'
STORED_WARNINGS = 20       # the status line keeps the text of this many warnings
STDOUT_LINE_WAIT_S = 5.0   # kStdoutLineWaitMs in opencorr_cli_cuda_mvp.cpp


def version_fields():
    run = subprocess.run([str(args.binary), '--version'], capture_output=True, text=True, timeout=30)
    return dict(line.split('=', 1) for line in run.stdout.splitlines()[1:])


VERSION = version_fields()


def hashes(folder, suffix='.bin'):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('frame_*' + suffix))}


def stat_snapshot(folder):
    """Every entry with inode, mtime, size and type: any write, even a rewrite of
    identical bytes or a new or removed name, changes it."""
    result = {}
    for path in [folder, *sorted(folder.rglob('*'))]:
        info = path.lstat()
        result[str(path.relative_to(folder))] = (info.st_ino, info.st_mtime_ns, info.st_size, stat.S_IFMT(info.st_mode))
    return result


def shifted(base, shift):
    pixels = bytearray([127]) * (WIDTH * HEIGHT)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            sx = x - shift; ix = math.floor(sx); fraction = sx - ix
            if 0 <= ix < WIDTH - 1:
                pixels[y * WIDTH + x] = round((1 - fraction) * base[y * WIDTH + ix] + fraction * base[y * WIDTH + ix + 1])
    return pixels


def write_pgm16(path, pixels, scale):
    path.write_bytes(f'P5\n{WIDTH} {HEIGHT}\n65535\n'.encode('ascii')
                     + struct.pack(f'>{WIDTH * HEIGHT}H', *(value * scale for value in pixels)))


def write_png(path, rows, bits):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', WIDTH, HEIGHT, bits, 0, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(b''.join(b'\x00' + row for row in rows))) + chunk(b'IEND', b''))


def write_png16(path, pixels, scale):
    write_png(path, [struct.pack(f'>{WIDTH}H', *(value * scale for value in pixels[y * WIDTH:(y + 1) * WIDTH]))
                     for y in range(HEIGHT)], 16)


def write_png4(path, pixels):
    """4-bit grayscale PNG: two pixels per byte, the top four bits of each 8-bit value."""
    write_png(path, [bytes((pixels[y * WIDTH + x] >> 4) << 4 | pixels[y * WIDTH + x + 1] >> 4
                           for x in range(0, WIDTH, 2)) for y in range(HEIGHT)], 4)


def write_tiff16(path, pixels, scale, order, bits_type=3):
    """Uncompressed single-strip 16-bit grayscale TIFF; order is '<' (II) or '>' (MM).
    bits_type is the field type of BitsPerSample: 3 (SHORT, the standard) or 4 (LONG)."""
    data = struct.pack(f'{order}{WIDTH * HEIGHT}H', *(value * scale for value in pixels))
    entries = [(256, 4, WIDTH), (257, 4, HEIGHT), (258, bits_type, 16), (259, 3, 1), (262, 3, 1),
               (273, 4, 8 + 2 + 9 * 12 + 4), (277, 3, 1), (278, 4, HEIGHT), (279, 4, len(data))]
    directory = struct.pack(order + 'H', len(entries))
    for tag, kind, value in entries:
        packed = struct.pack(order + 'HHI', tag, kind, 1)
        packed += struct.pack(order + 'HH', value, 0) if kind == 3 else struct.pack(order + 'I', value)
        directory += packed
    directory += struct.pack(order + 'I', 0)
    path.write_bytes((b'II' if order == '<' else b'MM') + struct.pack(order + 'HI', 42, 8) + directory + data)


class CliInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self._testMethodName
        self.root.mkdir()
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.serial = 0
        self.base = make_base(WIDTH, HEIGHT, 20261005)
        self.frames = [shifted(self.base, shift) for shift in (.25, .65, 1.1, .4)]
        write_pgm(self.inputs / 'ref.pgm', WIDTH, HEIGHT, self.base)
        for i, pixels in enumerate(self.frames):
            write_pgm(self.inputs / f'tar{i}.pgm', WIDTH, HEIGHT, pixels)
        self.reference = self.inputs / 'ref.pgm'
        self.listing = self.root / 'targets.txt'
        self.targets = [self.inputs / f'tar{i}.pgm' for i in range(4)]
        self.write_listing()

    def write_listing(self, targets=None, listing=None):
        targets = self.targets if targets is None else targets
        (listing or self.listing).write_text(''.join(f'{path} frame_{i:06d}.csv\n' for i, path in enumerate(targets)))

    def call(self, argv, env=None, cwd=None):
        run = subprocess.run([str(args.binary)] + [str(a) for a in argv], env=env, cwd=cwd,
                             capture_output=True, text=True, errors='replace', timeout=180)
        self.serial += 1
        (self.root / f'run-{self.serial}.log').write_text(
            json.dumps({'argv': [str(a) for a in argv], 'rc': run.returncode}) + '\n--- stdout\n' + run.stdout
            + '\n--- stderr\n' + run.stderr, errors='backslashreplace')
        return run

    def run_cli(self, out, fmt='raw', extra=(), success=True, fault=None, target='frame_000001.bin', listing=None):
        argv = ['--multi-frame', '--ref', self.reference, '--tar-list', listing or self.listing, '--output-dir', out,
                '7', '5', '--threads', '2', '--gpu-device', '0', '--output-format', fmt] + list(extra)
        env = None
        if fault:
            if not args.fault_lib: self.skipTest('requires the syscall fault library')
            env = dict(os.environ, LD_PRELOAD=str(args.fault_lib), OPENCORR_TEST_IO_FAULT=fault,
                       OPENCORR_TEST_IO_TARGET=target)
        run = self.call(argv, env=env)
        if success: self.assertEqual(run.returncode, 0, run.stderr[-2500:])
        else: self.assertNotEqual(run.returncode, 0)
        return run

    def copy_of(self, source, label):
        out = self.root / label
        subprocess.run(['cp', '-a', str(source), str(out)], check=True)
        return out

    def intent_of(self, out, key):
        """The ledger record of one output; its file name is a digest, so look inside."""
        for path in (out / '.opencorr-state').glob('*.json'):
            if path.name not in ('contract.json', 'contract.pending.json', 'complete.json') \
                    and json.loads(path.read_text()).get('key') == key:
                return path
        self.fail(f'no ledger record for {key}')

    def status(self, run):
        """The closing status line: stdout is exactly that one JSON line."""
        self.assertTrue(run.stdout.endswith('\n'), run.stdout[-200:])
        lines = run.stdout.splitlines()
        self.assertEqual(len(lines), 1, run.stdout[:500])
        line = json.loads(lines[0])
        self.assertEqual(line['schema'], STATUS_SCHEMA)
        self.assertEqual(line['exit_code'], run.returncode)
        self.assertEqual(line['status'], 'ok' if run.returncode == 0 else 'error')
        self.assertEqual(line['error'] is None, run.returncode == 0)
        self.assertEqual(len(line['warnings']), min(line['warning_count'], STORED_WARNINGS))
        return line

    def verify(self, out, expect):
        before = stat_snapshot(out) if out.is_dir() else None
        run = self.call(['--verify-only', out])
        lines = run.stdout.splitlines()
        self.assertEqual(len(lines), 1, run.stdout[:500] + run.stderr[-500:])
        report = json.loads(lines[0])
        self.assertEqual(report['schema'], VERIFY_SCHEMA)
        self.assertEqual(report['status'], expect, json.dumps(report)[:1500])
        self.assertEqual(report['complete'], expect == 'complete')
        self.assertEqual(run.returncode, 0 if expect == 'complete' else 1)
        self.assertEqual(report['exit_code'], run.returncode)
        self.assertIn('nothing was computed or written', run.stderr)
        if before is not None:
            self.assertEqual(stat_snapshot(out), before, 'verify-only changed the directory')
        return report

    # ---- #19: --help and --version
    def test_help_prints_usage_and_every_default_on_stdout(self):
        block = SOURCE.read_text(encoding='utf-8').split('bool parseCommonOption', 1)[1].split('bool loadTarList', 1)[0]
        options = sorted(set(re.findall(r'arg == "(--[a-z0-9-]+)"', block)))
        self.assertIn('--quiet', options)
        for argv in (['--help'], ['-h'], ['--multi-frame', '--help'], ['a', 'b', '-h']):
            run = self.call(argv)
            self.assertEqual(run.returncode, 0, argv)
            self.assertEqual(run.stderr, '')
            self.assertIn('Usage: opencorr_cli_cuda_mvp', run.stdout)
            self.assertIn('Options (default in brackets):', run.stdout)
            table = run.stdout.split('Options (default in brackets):', 1)[1]
            for option in options:
                self.assertRegex(table, r'(?m)^  .*%s\b.*\[.+\]$' % re.escape(option), option)
            self.assertRegex(table, r'--max-iter M .*\[10\]')
            self.assertRegex(table, r'--conv-threshold T .*\[0\.001\]')
            self.assertRegex(table, r'--input-depth 8\|12\|16\|auto .*\[8\]')
            self.assertRegex(table, r'--threads N .*: \d+ here\]')
            self.assertNotIn('"schema"', run.stdout)
            for mode in ('--verify-only <output_dir>', '--version', '--help'):
                self.assertIn(mode, run.stdout)

    def test_usage_errors_stay_nonzero_on_stderr(self):
        cases = [(['--bogus'], 'unknown option: --bogus'),
                 ([], 'no arguments given'),
                 (['--multi-frame'], 'missing: --ref --tar-list --output-dir'),
                 (['--multi-frame', '--tar-list', 'x', '--output-dir', 'y', '7', '5', '--ref'], 'missing value for --ref'),
                 (['a.pgm', 'b.pgm', '7'], 'got 3 positional argument(s)'),
                 (['a.pgm', 'b.pgm', '7', '5', 'out.csv', '--max-iter'], 'missing value for --max-iter')]
        for argv, message in cases:
            run = self.call(argv, cwd=self.root)
            self.assertEqual(run.returncode, 1, argv)
            self.assertIn('Error: ', run.stderr)
            self.assertIn(message, run.stderr)
            self.assertIn('Usage: opencorr_cli_cuda_mvp', run.stderr)
            self.assertIn('--help for every option with its default', run.stderr)
            self.assertNotIn('Options (default in brackets):', run.stderr)
            line = self.status(run)
            self.assertIn(message, line['error'])
            self.assertEqual(line['frames_processed'], 0)

    def test_version_keeps_its_first_four_lines_and_appends(self):
        run = self.call(['--version'])
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stderr, '')
        lines = run.stdout.splitlines()
        self.assertEqual(lines[0], 'FastICGN (opencorr_cli_cuda_mvp)')
        self.assertEqual(lines[1], 'name=opencorr_cli_cuda_mvp')
        self.assertRegex(lines[2], r'^git_hash=([0-9a-f]{12}|unknown)$')
        self.assertRegex(lines[3], r'^build_date=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')
        self.assertEqual([line.split('=', 1)[0] for line in lines[4:]], ['version', 'git_dirty', 'backend', 'fftw'])
        fields = dict(line.split('=', 1) for line in lines[1:])
        self.assertRegex(fields['version'], r'^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$')
        self.assertIn(fields['git_dirty'], ('0', '1', 'unknown'))
        self.assertEqual(fields['git_dirty'] == 'unknown', fields['git_hash'] == 'unknown')
        self.assertIn(fields['backend'], ('cuda', 'cpu'))
        self.assertIn(fields['fftw'], ('0', '1'))
        if args.expect_version: self.assertEqual(fields['version'], args.expect_version)
        if args.expect_backend: self.assertEqual(fields['backend'], args.expect_backend)
        if args.expect_fftw: self.assertEqual(fields['fftw'], args.expect_fftw)
        extra = self.call(['--version', 'more'])
        self.assertEqual((extra.returncode, extra.stdout), (1, ''))
        self.assertIn('--version does not accept additional arguments', extra.stderr)

    # ---- #22: status line and --quiet
    def test_status_line_closes_success_resume_and_failure(self):
        out = self.root / 'out'
        line = self.status(self.run_cli(out))
        self.assertEqual((line['mode'], line['frames_total'], line['frames_processed'], line['frames_skipped']),
                         ('multi-frame', 4, 4, 0))
        self.assertEqual(line['output_dir'], str(out.resolve()))
        self.assertEqual(line['output_format'], 'raw')
        self.assertGreater(line['elapsed_sec'], 0)
        self.assertEqual(line['warnings'], [])
        self.assertEqual((line['version'], line['git_hash'], line['backend']),
                         (VERSION['version'], VERSION['git_hash'], VERSION['backend']))
        self.assertEqual(line['git_dirty'], {'0': False, '1': True, 'unknown': None}[VERSION['git_dirty']])
        self.assertEqual(line['totals']['skip_fftcc_frames'], 3)
        manifest = json.loads((out / 'manifest.opencorr-raw.json').read_text())
        self.assertEqual(manifest['engine']['version'], VERSION['version'])
        self.assertEqual(manifest['engine']['git_dirty'], line['git_dirty'])

        again = self.run_cli(out)
        self.assertIn('Verified complete run', again.stderr)
        line = self.status(again)
        self.assertEqual((line['frames_total'], line['frames_processed'], line['frames_skipped']), (4, 0, 4))

        self.write_listing(self.targets[:2] + [self.inputs / 'absent.pgm'])
        failed = self.run_cli(self.root / 'failed', success=False)
        line = self.status(failed)
        self.assertEqual(line['frames_total'], 3)
        self.assertIn('absent.pgm', line['error'])
        self.assertIn('Error: ', failed.stderr)
        self.assertIn(line['error'], failed.stderr)

        single = self.call([self.reference, self.targets[0], '7', '5', self.root / 'single.csv'])
        self.assertEqual(single.returncode, 0, single.stderr[-1500:])
        line = self.status(single)
        self.assertEqual((line['mode'], line['frames_total'], line['frames_processed']), ('single-frame', 1, 1))
        self.assertEqual(line['output_path'], str((self.root / 'single.csv').resolve()))
        self.assertEqual(line['output_dir'], str(self.root.resolve()))

    def test_status_line_survives_a_reader_that_went_away(self):
        # stdout is a pipe whose read end is closed before the run ends: the run
        # must still exit 0 instead of dying of SIGPIPE on its last line.
        out = self.root / 'out'
        read_end, write_end = os.pipe()
        os.close(read_end)
        try:
            run = subprocess.run([str(args.binary), '--multi-frame', '--ref', str(self.reference), '--tar-list',
                                  str(self.listing), '--output-dir', str(out), '7', '5', '--threads', '2',
                                  '--output-format', 'raw'], stdout=write_end, stderr=subprocess.PIPE, timeout=180)
        finally:
            os.close(write_end)
        self.assertEqual(run.returncode, 0, run.stderr[-1500:])
        self.assertEqual(self.verify(out, 'complete')['frames_verified'], 4)

    def test_quiet_drops_only_the_per_frame_lines(self):
        loud = self.run_cli(self.root / 'loud', extra=['--temporal-extrapolation'])
        quiet = self.run_cli(self.root / 'quiet', extra=['--temporal-extrapolation', '--quiet'])
        self.assertEqual(hashes(self.root / 'loud'), hashes(self.root / 'quiet'))
        self.assertEqual(len(hashes(self.root / 'quiet')), 4)
        progress = [line for line in loud.stderr.splitlines() if re.match(r'^\[\d+/4\] ', line)]
        self.assertEqual(len(progress), 4)
        # The per-frame line keeps the format it had before --quiet existed.
        self.assertRegex(progress[0], r'^\[1/4\] load-visible \S+ sec \(preload I/O \S+ sec\), FFTCC \S+ sec, '
                                      r'(GPU|CPU) ICGN \S+ sec, bad_ratio=\S+, baseline_bad_ratio=\S+, '
                                      r'fallback_threshold=\S+, valid_ratio=\S+, iteration_mean=\S+, iteration_max=\d+, '
                                      r'poi_fallback_count=\d+, poi_fallback_recovered=\d+, temporal_seed_count=\d+, '
                                      r'CSV-wait \S+ sec, CSV-enqueue \S+ sec, raw-wait \S+ sec, raw-enqueue \S+ sec, '
                                      r'skip-fftcc=no, fallback no -> .*/frame_000000\.bin$')
        self.assertIn('[BASELINE]', loud.stderr)
        for marker in ('[BASELINE]', '[FALLBACK]', '[POI-FALLBACK]'):
            self.assertNotIn(marker, quiet.stderr)
        self.assertFalse([line for line in quiet.stderr.splitlines() if re.match(r'^\[\d+/4\] ', line)])
        for kept in ('Multi-frame mode: 4 target frames', 'Total POIs:', 'Processed frames: 4, skipped: 0',
                     'Frames with skipped FFTCC: 3', 'Total wall time:'):
            self.assertIn(kept, loud.stderr)
            self.assertIn(kept, quiet.stderr)
        self.assertEqual(self.status(quiet)['frames_processed'], 4)
        # Warnings and errors are not progress: --quiet keeps them.
        self.write_listing(self.targets[:1] + [self.inputs / 'absent.pgm'])
        failed = self.run_cli(self.root / 'failed', extra=['--quiet'], success=False)
        self.assertIn('Error: ', failed.stderr)

    def test_help_is_not_read_from_an_option_value_or_a_verify_only_directory(self):
        source = SOURCE.read_text(encoding='utf-8')
        block = source.split('bool parseCommonOption', 1)[1].split('bool loadTarList', 1)[0]
        reads_a_value = set(re.findall(r'requireValue\("(--[a-z0-9-]+)"\)', block)) | {'--ref', '--tar-list', '--output-dir'}
        listed = set(re.findall(r'"(--[a-z0-9-]+)"', source.split('kOptionsWithValue[] = {', 1)[1].split('};', 1)[0]))
        self.assertEqual(listed, reads_a_value)

        for option in sorted(listed):
            work = self.root / f'cwd{option}'
            work.mkdir()
            for word in ('-h', '--help'):
                out = work / f'out{word}'
                argv = ['--multi-frame']
                for name, value in (('--ref', self.reference), ('--tar-list', self.listing), ('--output-dir', out)):
                    if name != option: argv += [name, value]
                argv += ['7', '5', '--threads', '2', '--output-format', 'raw', option, word]
                run = self.call(argv, cwd=work)
                line = self.status(run)  # one status line: the word was a value, not a request for help
                self.assertNotIn('Usage:', run.stdout)
                if option == '--output-dir':
                    self.assertEqual((line['status'], line['output_dir']), ('ok', str((work / word).resolve())))
                    self.assertEqual(len(hashes(work / word)), 4)
                elif option == '--diagnostics-dir':
                    self.assertEqual(line['status'], 'ok')
                    self.assertTrue((work / word / 'manifest.opencorr-diagnostics.json').is_file())
                else:
                    self.assertEqual(line['status'], 'error')

        # --verify-only: its argument is a directory whatever it is called.
        runs = self.root / 'cwd--output-dir'
        for word in ('-h', '--help'):
            run = self.call(['--verify-only', word], cwd=runs)
            self.assertEqual(run.returncode, 0, run.stdout[:300] + run.stderr[-300:])
            report = json.loads(run.stdout)
            self.assertEqual((report['schema'], report['status'], report['frames_verified'], report['output_dir']),
                             (VERIFY_SCHEMA, 'complete', 4, str((runs / word).resolve())))
            run = self.call(['--verify-only', word], cwd=self.inputs)  # no such directory here
            self.assertEqual(run.returncode, 1)
            self.assertEqual(json.loads(run.stdout)['status'], 'not_found')
            self.assertNotIn('Usage:', run.stdout + run.stderr)

        # Everywhere else the two words still ask for help.
        for argv in (['--multi-frame', '--ref', self.reference, '-h'], ['--quiet', '--help'],
                     ['--help', '--verify-only'], ['--version', '--help'], ['--output-dir', 'x', '-h']):
            run = self.call(argv)
            self.assertEqual(run.returncode, 0, argv)
            self.assertIn('Options (default in brackets):', run.stdout)

    def test_status_line_counts_every_warning_and_keeps_the_first_twenty(self):
        rows = ','.join(['0'] * 25 + ['60'])  # row 0 is inside the subset margin: one warning per mention
        run = self.run_cli(self.root / 'out', extra=['--line', rows])
        printed = [line for line in run.stderr.splitlines() if line.startswith('Warning: ')]
        self.assertEqual(len(printed), 25, run.stderr[-800:])
        line = self.status(run)
        self.assertEqual((line['warning_count'], len(line['warnings'])), (25, STORED_WARNINGS))
        self.assertEqual(['Warning: ' + text for text in line['warnings']], printed[:STORED_WARNINGS])

    def test_status_line_waits_for_room_in_a_full_non_blocking_pipe(self):
        def start(name):
            read_end, write_end = os.pipe()
            fcntl.fcntl(write_end, fcntl.F_SETFL, fcntl.fcntl(write_end, fcntl.F_GETFL) | os.O_NONBLOCK)
            size = fcntl.fcntl(write_end, 1032)  # F_GETPIPE_SZ
            os.write(write_end, b'x' * (size - 300))  # 300 bytes of room: less than the line needs
            with open(self.root / f'{name}.stderr', 'wb') as log:
                process = subprocess.Popen(
                    [str(args.binary), '--multi-frame', '--ref', str(self.reference), '--tar-list', str(self.listing),
                     '--output-dir', str(self.root / name), '7', '5', '--threads', '2', '--output-format', 'raw'],
                    stdout=write_end, stderr=log)
            os.close(write_end)
            # The closing totals are the last thing on stderr; the status line is the next act.
            deadline = time.monotonic() + 120
            while b'Total wall time' not in (self.root / f'{name}.stderr').read_bytes():
                self.assertIsNone(process.poll(), 'the run ended before its closing totals')
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            return process, read_end, size, time.monotonic()

        def drain(read_end):
            data = b''
            while True:
                chunk = os.read(read_end, 1 << 16)
                if not chunk: break
                data += chunk
            os.close(read_end)
            return data

        # A reader that comes back: the line arrives whole.
        process, read_end, size, _ = start('read-late')
        time.sleep(0.5)
        self.assertIsNone(process.poll(), 'the process did not wait for room in the pipe')
        data = drain(read_end)
        self.assertEqual(process.wait(timeout=60), 0)
        line = data[size - 300:]
        self.assertTrue(line.endswith(b'\n') and line.count(b'\n') == 1, line[-200:])
        status = json.loads(line)
        self.assertEqual((status['schema'], status['status'], status['frames_processed']), (STATUS_SCHEMA, 'ok', 4))

        # A reader that never does: the wait is bounded, the run still exits 0, the line is lost.
        process, read_end, size, since = start('never-read')
        self.assertEqual(process.wait(timeout=STDOUT_LINE_WAIT_S + 60), 0)
        waited = time.monotonic() - since
        self.assertGreater(waited, STDOUT_LINE_WAIT_S - 1.0)
        self.assertEqual(drain(read_end)[size - 300:], b'')
        self.assertEqual(self.verify(self.root / 'never-read', 'complete')['frames_verified'], 4)

    def test_single_frame_non_utf8_output_path_is_not_recoverable_from_the_status_line(self):
        # Documented behaviour, pinned: the file is written where it was asked for,
        # the status line stays valid JSON and shows the undecodable bytes as U+FFFD.
        folder = self.root / os.fsdecode(b'sf-\xe9\xff')
        folder.mkdir()
        target = folder / os.fsdecode(b'out-\xfe.csv')
        run = self.call([self.reference, self.targets[0], '7', '5', target])
        self.assertEqual(run.returncode, 0, run.stderr[-800:])
        self.assertTrue(target.is_file())
        line = self.status(run)
        self.assertEqual(line['output_path'], str(self.root.resolve()) + '/sf-��/out-�.csv')
        self.assertFalse(os.path.exists(line['output_path']))

    # ---- #20: bit depth
    def deep_inputs(self, scale, writer=write_pgm16, suffix='pgm'):
        folder = self.inputs / f'deep-{scale}-{suffix}{writer.__name__}'
        folder.mkdir()
        writer(folder / f'ref.{suffix}', self.base, scale)
        targets = []
        for i, pixels in enumerate(self.frames):
            writer(folder / f'tar{i}.{suffix}', pixels, scale)
            targets.append(folder / f'tar{i}.{suffix}')
        return folder / f'ref.{suffix}', targets

    def test_sixteen_bit_reference_read_as_eight_bit_warns(self):
        # 257 spreads 8-bit values over the whole 16-bit range (low bits vary);
        # 16 leaves them within 12 bits, as right-aligned 12-bit data.
        advice = {257: '--input-depth 16', 16: '--input-depth 12 --input-align right'}
        for scale, expected in advice.items():
            self.reference, targets = self.deep_inputs(scale)
            self.write_listing(targets)
            out = self.root / f'default-{scale}'
            run = self.run_cli(out)
            warnings = [line for line in run.stderr.splitlines() if line.startswith('Warning: ')]
            self.assertEqual(len(warnings), 1, run.stderr[-1500:])
            self.assertIn('the reference image is stored as CV_16U', warnings[0])
            self.assertIn('--input-depth 8', warnings[0])
            self.assertIn(expected, warnings[0])
            self.assertIn(str(self.reference), warnings[0])
            line = self.status(run)
            self.assertEqual(line['warning_count'], 1)
            self.assertEqual('Warning: ' + line['warnings'][0], warnings[0])
            image = json.loads((out / 'manifest.opencorr-raw.json').read_text())['input_image']
            self.assertEqual((image['depth'], image['reference_source_depth'], image['target_source_depth_mismatches']),
                             ('8', 'CV_16U', 0))
            self.assertEqual(image['reference_source_range'][1], max(self.base) * scale)
            # The warning costs no bytes: the source depth is not in the contract.
            contract = json.loads((out / '.opencorr-state/contract.json').read_text())
            self.assertNotIn('source', json.dumps(contract['options']))
            self.assertIn('Warning: the reference image is stored as CV_16U', self.run_cli(out).stderr)

            explicit = self.root / f'explicit-{scale}'
            depth = ['--input-depth', '16'] if scale == 257 else ['--input-depth', '12', '--input-align', 'right']
            run = self.run_cli(explicit, extra=depth)
            self.assertNotIn('Warning: ', run.stderr)
            self.assertEqual(self.status(run)['warnings'], [])
            image = json.loads((explicit / 'manifest.opencorr-raw.json').read_text())['input_image']
            self.assertEqual((image['reference_source_depth'], image['target_source_depth_mismatches']), ('CV_16U', 0))

        # 12-bit data stored shifted left by 4: every value a multiple of 16, not of 256.
        folder = self.inputs / 'left'
        folder.mkdir()
        twelve = lambda pixels: [value * 16 + value % 16 for value in pixels]
        write_pgm16(folder / 'ref.pgm', twelve(self.base), 16)
        write_pgm16(folder / 'tar0.pgm', twelve(self.frames[0]), 16)
        self.assertGreater(max(twelve(self.base)) * 16, 4095)
        single = self.call([folder / 'ref.pgm', folder / 'tar0.pgm', '7', '5', self.root / 'left.csv'])
        self.assertEqual(single.returncode, 0, single.stderr[-1500:])
        self.assertIn('--input-depth 12 --input-align left', single.stderr)
        self.assertEqual(self.status(single)['warning_count'], 1)
        # 8-bit data in a 16-bit file: still reported, and said to lose nothing.
        write_pgm16(folder / 'wide.pgm', self.base, 256)
        single = self.call([folder / 'wide.pgm', folder / 'wide.pgm', '7', '5', self.root / 'wide.csv'])
        self.assertEqual(single.returncode, 0, single.stderr[-1500:])
        self.assertIn('every value is a multiple of 256', single.stderr)

    def test_eight_bit_inputs_raise_no_depth_warning(self):
        out = self.root / 'out'
        run = self.run_cli(out)
        self.assertNotIn('Warning: ', run.stderr)
        image = json.loads((out / 'manifest.opencorr-raw.json').read_text())['input_image']
        self.assertEqual((image['reference_source_depth'], image['reference_source_range'],
                          image['target_source_depth_mismatches']),
                         ('CV_8U', [min(self.base), max(self.base)], 0))

    def test_target_stored_deeper_than_the_reference_warns(self):
        _, deep = self.deep_inputs(256)
        for label, extra in (('default', []), ('depth16', ['--input-depth', '16'])):
            self.write_listing([self.targets[0], deep[1], self.targets[2], self.targets[3]])
            out = self.root / f'one-{label}'
            run = self.run_cli(out, extra=extra)
            warnings = [line for line in run.stderr.splitlines() if line.startswith('Warning: ')]
            self.assertEqual(len(warnings), 1, run.stderr[-1500:])
            self.assertIn('target frame_000001.csv is stored as', warnings[0])
            self.assertIn(str(deep[1]), warnings[0])
            self.assertEqual(self.status(run)['warning_count'], 1)
            image = json.loads((out / 'manifest.opencorr-raw.json').read_text())['input_image']
            self.assertEqual(image['target_source_depth_mismatches'], 1)

            self.write_listing([self.targets[0], deep[1], deep[2], self.targets[3]])
            out = self.root / f'two-{label}'
            run = self.run_cli(out, extra=extra)
            warnings = [line for line in run.stderr.splitlines() if line.startswith('Warning: ')]
            self.assertEqual(len(warnings), 2, run.stderr[-1500:])
            self.assertIn('2 of 4 target frames differ from the reference in stored bit depth', warnings[1])
            image = json.loads((out / 'manifest.opencorr-raw.json').read_text())['input_image']
            self.assertEqual(image['target_source_depth_mismatches'], 2)

    def multi_frame(self, out, targets):
        self.write_listing(targets)
        return self.call(['--multi-frame', '--ref', self.reference, '--tar-list', self.listing, '--output-dir', out,
                          '7', '5', '--threads', '2', '--output-format', 'raw'])

    def test_header_probe_reads_every_container_this_build_decodes(self):
        writers = {'pnm': (write_pgm16, 'pgm'),
                   'png': (write_png16, 'png'),
                   'tiff-ii': (lambda path, pixels, scale: write_tiff16(path, pixels, scale, '<'), 'tif'),
                   'tiff-mm': (lambda path, pixels, scale: write_tiff16(path, pixels, scale, '>'), 'tif'),
                   # BitsPerSample stored as a LONG instead of the standard SHORT
                   'tiff-long-bits': (lambda path, pixels, scale: write_tiff16(path, pixels, scale, '<', 4), 'tif')}
        decoded = []
        for label, (writer, suffix) in writers.items():
            with self.subTest(container=label):
                deep = self.inputs / f'deep-{label}.{suffix}'
                writer(deep, self.frames[1], 256)
                run = self.multi_frame(self.root / label, [self.targets[0], deep])
                if run.returncode != 0 and 'Fail to load file' in run.stderr:
                    self.skipTest(f'this OpenCV build has no decoder for {label}')  # counted as a skip
                self.assertEqual(run.returncode, 0, run.stderr[-1500:])
                warnings = [line for line in run.stderr.splitlines() if line.startswith('Warning: ')]
                self.assertEqual(len(warnings), 1, (label, run.stderr[-1500:]))
                self.assertIn('target frame_000001.csv is stored as 16-bit samples but the reference as 8-bit',
                              warnings[0])
                # The same file as the reference: the decode with IMREAD_ANYDEPTH sees its depth.
                run = self.call([deep, deep, '7', '5', self.root / f'{label}.csv'])
                self.assertEqual(run.returncode, 0, run.stderr[-1500:])
                self.assertIn('Warning: the reference image is stored as CV_16U', run.stderr)
                decoded.append(label)
        # Netpbm needs no codec library: whatever else this build lacks, the probe was exercised.
        self.assertIn('pnm', decoded)
        (self.root / 'containers.json').write_text(json.dumps({'checked': decoded,
                                                                'skipped': sorted(set(writers) - set(decoded))}) + '\n')

    def test_depths_up_to_eight_bits_are_one_scale_and_an_unread_header_is_not_compared(self):
        # OpenCV expands a 4-bit grey PNG to full-range 8-bit samples, as it reads the 8-bit reference.
        shallow = self.inputs / 'shallow.png'
        write_png4(shallow, self.frames[1])
        run = self.multi_frame(self.root / 'shallow', [self.targets[0], shallow, self.targets[2]])
        if run.returncode != 0 and 'Fail to load file' in run.stderr:
            self.skipTest('this OpenCV build has no PNG decoder')
        self.assertEqual(run.returncode, 0, run.stderr[-1500:])
        self.assertNotIn('Warning: ', run.stderr)
        image = json.loads((self.root / 'shallow/manifest.opencorr-raw.json').read_text())['input_image']
        self.assertEqual(image['target_source_depth_mismatches'], 0)
        self.assertEqual(self.status(run)['warning_count'], 0)

        # Documented limit: a header the probe cannot read is not compared. Here a
        # comment pushes the PGM maxval past the bytes the probe looks at.
        hidden = self.inputs / 'long-comment.pgm'
        hidden.write_bytes(b'P5\n#' + b'c' * 700 + f'\n{WIDTH} {HEIGHT}\n65535\n'.encode('ascii')
                           + struct.pack(f'>{WIDTH * HEIGHT}H', *(value * 257 for value in self.frames[1])))
        run = self.multi_frame(self.root / 'hidden', [self.targets[0], hidden])
        self.assertEqual(run.returncode, 0, run.stderr[-1500:])
        self.assertNotIn('Warning: ', run.stderr)
        image = json.loads((self.root / 'hidden/manifest.opencorr-raw.json').read_text())['input_image']
        self.assertEqual(image['target_source_depth_mismatches'], 0)
        # As the reference the same file is decoded once more, and that read sees its depth.
        run = self.call([hidden, hidden, '7', '5', self.root / 'hidden.csv'])
        self.assertEqual(run.returncode, 0, run.stderr[-1500:])
        self.assertIn('Warning: the reference image is stored as CV_16U', run.stderr)

    # ---- #21: contract, rejection messages, --verify-only
    def test_scheduling_options_are_not_in_the_resume_contract(self):
        out = self.root / 'out'
        self.run_cli(out)
        expected = hashes(out)
        contract = json.loads((out / '.opencorr-state/contract.json').read_text())
        for key in SCHEDULING_OPTIONS:
            self.assertNotIn(key, contract['options'])
        engine = json.loads((out / 'manifest.opencorr-raw.json').read_text())['engine']
        self.assertEqual((engine['threads'], engine['decode_threads'], engine['preload'], engine['gpu_device']),
                         (2, 1, True, 0))
        before = stat_snapshot(out)
        run = self.run_cli(out, extra=['--threads', '1', '--no-preload', '--decode-threads', '3'])
        self.assertIn('Verified complete run', run.stderr)
        self.assertEqual(stat_snapshot(out), before)
        # An interrupted run finishes under other scheduling options, to the same bytes.
        (out / 'frame_000002.bin').unlink()
        (out / '.opencorr-state/complete.json').unlink()
        self.run_cli(out, extra=['--threads', '3', '--decode-threads', '2'])
        self.assertEqual(hashes(out), expected)
        self.assertEqual(json.loads((out / 'manifest.opencorr-raw.json').read_text())['engine']['threads'], 3)
        self.verify(out, 'complete')

    def test_a_wrong_gpu_device_no_longer_poisons_the_directory(self):
        if VERSION['backend'] != 'cuda': self.skipTest('CPU-only builds never open a CUDA device')
        out = self.root / 'out'
        argv = ['--multi-frame', '--ref', self.reference, '--tar-list', self.listing, '--output-dir', out, '7', '5',
                '--threads', '2', '--output-format', 'raw']
        failed = self.call(argv + ['--gpu-device', '63'])
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(self.status(failed)['frames_processed'], 0)
        retried = self.call(argv + ['--gpu-device', '0'])
        self.assertEqual(retried.returncode, 0, retried.stderr[-1500:])
        self.assertEqual(self.verify(out, 'complete')['frames_verified'], 4)

    def test_contract_written_with_scheduling_options_still_resumes(self):
        # A contract in the layout binaries wrote before 2026-10-05: the four
        # scheduling options are listed. They are not compared, so the same
        # binary resumes it, whatever their values, and rewrites nothing but the ledger.
        out = self.root / 'out'
        self.run_cli(out)
        expected = hashes(out)
        marks = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in out.glob('frame_*.bin')}
        path = out / '.opencorr-state/contract.json'
        contract = json.loads(path.read_text())
        contract['options'].update({'gpu_device': 1, 'threads': 39, 'preload': False, 'decode_threads': 8})
        path.write_text(json.dumps(contract, indent=2) + '\n')
        # Its completion record digests that older contract text, not today's.
        completion = out / '.opencorr-state/complete.json'
        record = json.loads(completion.read_text())
        record['contract_digest'] = 'f' * 32
        completion.write_text(json.dumps(record, indent=2) + '\n')
        legacy = self.verify(out, 'incomplete')
        self.assertEqual(legacy['contract']['legacy_scheduling_options'], list(SCHEDULING_OPTIONS))
        self.assertEqual(legacy['frames_verified'], 4)
        run = self.run_cli(out)
        self.assertNotIn('resume rejected', run.stderr)
        self.assertEqual(hashes(out), expected)
        self.assertEqual(marks, {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in out.glob('frame_*.bin')})
        for key in SCHEDULING_OPTIONS:
            self.assertNotIn(key, json.loads(path.read_text())['options'])
        self.assertIn('Verified complete run', self.run_cli(out).stderr)
        self.assertEqual(self.verify(out, 'complete')['contract']['legacy_scheduling_options'], [])

    def test_rejected_resume_names_the_field_and_both_values(self):
        out = self.root / 'out'
        self.run_cli(out)
        before = stat_snapshot(out)

        run = self.run_cli(out, extra=['--max-iter', '11', '--no-poi-fallback'], success=False)
        self.assertIn('resume rejected: run configuration/reference/binary changed', run.stderr)
        self.assertIn('options.max_iteration: 10 -> 11', run.stderr)
        self.assertIn('options.poi_fallback: true -> false', run.stderr)
        self.assertNotIn('options.threads', run.stderr)
        line = self.status(run)
        self.assertIn('options.max_iteration: 10 -> 11', line['error'])
        self.assertEqual((line['frames_total'], line['frames_processed']), (4, 0))
        run = self.run_cli(out, extra=['--conv-threshold', '0.01'], success=False)
        self.assertIn('options.convergence: 0.001 -> 0.01;', run.stderr)

        self.write_listing([self.targets[1], self.targets[0], self.targets[2], self.targets[3]])
        run = self.run_cli(out, success=False)
        self.assertIn('resume rejected: targets changed/reordered', run.stderr)
        self.assertIn('target 1 of the recorded 4 differs', run.stderr)
        self.assertIn(f'input.path: "{self.targets[0]}" -> "{self.targets[1]}"', run.stderr)

        self.write_listing(self.targets[:2])
        run = self.run_cli(out, success=False)
        self.assertIn('the recorded run lists 4 targets, this command only 2', run.stderr)
        self.write_listing()
        self.assertEqual(stat_snapshot(out), before)

        # Another build of the binary: same code, different bytes.
        other = self.root / 'other-binary'
        other.write_bytes(args.binary.read_bytes() + b'\0')
        other.chmod(0o755)
        run = subprocess.run([str(other), '--multi-frame', '--ref', str(self.reference), '--tar-list', str(self.listing),
                              '--output-dir', str(out), '7', '5', '--threads', '2', '--output-format', 'raw'],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 1, run.stderr[-1500:])
        self.assertIn('binary.digest: ', run.stderr)
        self.assertIn('another build of the executable wrote this directory', run.stderr)
        self.assertEqual(stat_snapshot(out), before)

        contract = out / '.opencorr-state/contract.json'
        record = json.loads(contract.read_text())
        record['reference']['content']['digest'] = '0' * 32
        contract.write_text(json.dumps(record))
        run = self.run_cli(out, success=False)
        self.assertIn('reference.content.digest: "' + '0' * 32 + '" -> ', run.stderr)
        self.assertIn('the reference image path or content differs', run.stderr)

    def test_verify_only_passes_a_complete_run_and_writes_nothing(self):
        inside = self.root / 'raw'
        inside.mkdir()
        listing = inside / 'tar_list.txt'  # where the pipeline keeps its target list
        self.write_listing(listing=listing)
        self.run_cli(inside, listing=listing)
        report = self.verify(inside, 'complete')
        self.assertEqual((report['frames_expected'], report['frames_verified']), (4, 4))
        self.assertEqual((report['files_expected'], report['files_verified']), (5, 5))
        self.assertEqual(report['bytes_verified'],
                         sum(p.stat().st_size for p in inside.glob('frame_*.bin'))
                         + (inside / 'manifest.opencorr-raw.json').stat().st_size)
        self.assertEqual(report['output_dir'], str(inside.resolve()))
        self.assertEqual(report['completion'], {'present': True, 'state': 'current', 'durability': 'rename'})
        self.assertEqual(report['manifests'], [{'file': 'manifest.opencorr-raw.json', 'state': 'ok'}])
        self.assertEqual(report['inputs']['status'], 'match')
        self.assertEqual(report['inputs']['checked'], 5)
        self.assertEqual(report['contract']['git_hash'], VERSION['git_hash'])
        self.assertEqual(report['contract']['options']['radius'], 7)
        self.assertEqual((report['version'], report['git_hash']), (VERSION['version'], VERSION['git_hash']))
        self.assertEqual(report['problems'], [])
        self.assertEqual((report['untracked'], report['untracked_outputs'], report['residue']), (['tar_list.txt'], [], []))
        self.assertEqual(report['warnings'], [])
        for key in ('missing', 'mismatched', 'no_intent', 'invalid_intent'):
            self.assertEqual((report[key], report[key + '_count']), ([], 0))
        self.assertFalse((inside / '.opencorr-state/quarantine').exists())

        csv = self.root / 'csv'
        self.run_cli(csv, 'csv', extra=['--diagnostics', '--output-roi', '30,25,160,100'])
        report = self.verify(csv, 'complete')
        self.assertEqual((report['frames_expected'], report['frames_verified'], report['files_expected'],
                          report['files_verified']), (4, 4, 9, 9))
        self.assertEqual([m['file'] for m in report['manifests']], ['diagnostics/manifest.opencorr-diagnostics.json'])

        if os.geteuid() != 0:
            entries = [csv, *csv.rglob('*')]
            modes = {p: p.stat().st_mode for p in entries}
            for p in entries: p.chmod(modes[p] & ~0o222)
            try:
                self.assertEqual(self.verify(csv, 'complete')['frames_verified'], 4)
            finally:
                for p in entries: p.chmod(modes[p])

    def test_verify_only_tells_unfinished_from_damaged(self):
        fresh = self.root / 'fresh'
        self.run_cli(fresh)

        def copy(label):
            return self.copy_of(fresh, label)

        out = copy('missing-frame')
        (out / 'frame_000002.bin').unlink()
        report = self.verify(out, 'incomplete')
        self.assertEqual((report['missing'], report['frames_verified']), (['frame_000002.bin'], 3))

        out = copy('no-completion')
        (out / '.opencorr-state/complete.json').unlink()
        report = self.verify(out, 'incomplete')
        self.assertEqual((report['completion']['state'], report['frames_verified']), ('missing', 4))

        out = copy('truncated')
        path = out / 'frame_000001.bin'
        path.write_bytes(path.read_bytes()[:-1])
        report = self.verify(out, 'damaged')
        self.assertEqual((report['mismatched'], report['frames_verified']), (['frame_000001.bin'], 3))

        out = copy('flipped')
        path = out / 'frame_000003.bin'
        data = bytearray(path.read_bytes()); data[100] ^= 1; path.write_bytes(data)
        self.assertEqual(self.verify(out, 'damaged')['mismatched'], ['frame_000003.bin'])

        out = copy('no-intent')
        intents = sorted(p for p in (out / '.opencorr-state').glob('*.json') if p.name not in ('contract.json', 'complete.json'))
        self.assertEqual(len(intents), 4)
        intents[0].unlink()
        self.assertEqual(self.verify(out, 'damaged')['no_intent_count'], 1)

        out = copy('foreign-frame')
        (out / 'frame_999999.bin').write_bytes(b'foreign')
        self.assertEqual(self.verify(out, 'damaged')['untracked_outputs'], ['frame_999999.bin'])

        out = copy('manifest-changed')
        path = out / 'manifest.opencorr-raw.json'
        manifest = json.loads(path.read_text()); manifest['frames'] = manifest['frames'][:3]; path.write_text(json.dumps(manifest))
        self.assertEqual(self.verify(out, 'damaged')['manifests'][0]['state'], 'changed')

        out = copy('contract-broken')
        (out / '.opencorr-state/contract.json').write_text('{"targets": [')
        self.assertIn('contract.json is not a usable run contract', self.verify(out, 'damaged')['problems'][0])

        out = copy('residue')
        (out / '.opencorr-tmp-12345-7').write_bytes(b'orphan payload')
        (out / '.opencorr-state/contract.pending.json').write_text('{abandoned append')
        report = self.verify(out, 'complete')
        self.assertEqual((report['residue'], report['pending_append']), (['.opencorr-tmp-12345-7'], True))
        self.assertEqual(len(report['warnings']), 2)
        self.assertTrue((out / '.opencorr-tmp-12345-7').exists())

        out = copy('input-changed')
        path = self.inputs / 'tar2.pgm'
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        report = self.verify(out, 'inputs_changed')
        self.assertEqual((report['inputs']['status'], report['inputs']['changed'], report['frames_verified']),
                         ('changed', [str(path)], 4))
        path.unlink()
        report = self.verify(out, 'complete')
        self.assertEqual((report['inputs']['status'], report['inputs']['missing']), ('missing', [str(path)]))
        self.assertEqual(len(report['warnings']), 1)

    def test_verify_only_refuses_what_is_not_a_finished_run_directory(self):
        empty = self.root / 'empty'
        empty.mkdir()
        self.assertIn('no .opencorr-state/contract.json', self.verify(empty, 'not_a_run')['problems'][0])
        self.verify(self.root / 'absent', 'not_found')
        self.verify(self.reference, 'not_a_run')
        single = self.root / 'single'
        single.mkdir()
        self.assertEqual(self.call([self.reference, self.targets[0], '7', '5', single / 'out.csv']).returncode, 0)
        self.verify(single, 'not_a_run')

        out = self.root / 'out'
        self.run_cli(out)
        descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertIn('holds this directory', self.verify(out, 'locked')['problems'][0])
        finally:
            os.close(descriptor)
        self.verify(out, 'complete')

        # A wrong argument count is still answered with one JSON verdict on stdout.
        before = stat_snapshot(out)
        for argv in (['--verify-only'], ['--verify-only', out, 'extra'], ['--verify-only', '']):
            run = self.call(argv)
            self.assertEqual(run.returncode, 1)
            self.assertIn('Error: --verify-only takes exactly one argument', run.stderr)
            lines = run.stdout.splitlines()
            self.assertEqual(len(lines), 1, run.stdout[:300])
            report = json.loads(lines[0])
            self.assertEqual((report['schema'], report['status'], report['complete'], report['exit_code'],
                              report['output_dir']), (VERIFY_SCHEMA, 'error', False, 1, None))
            self.assertIn('--verify-only takes exactly one argument', report['problems'][0])
            self.assertEqual((report['version'], report['git_hash']), (VERSION['version'], VERSION['git_hash']))
        self.assertEqual(stat_snapshot(out), before)

    def test_verify_only_sees_an_interrupted_run_and_an_abandoned_append(self):
        out = self.root / 'crashed'
        run = self.run_cli(out, fault='crash-before-publish', target='frame_000002.bin', success=False)
        self.assertEqual(run.returncode, 86)
        self.assertEqual(run.stdout, '')  # killed: no status line, which is itself the signal
        report = self.verify(out, 'incomplete')
        self.assertEqual((report['frames_verified'], report['missing']), (2, ['frame_000002.bin', 'frame_000003.bin']))
        self.assertEqual(report['completion']['state'], 'missing')
        self.run_cli(out)
        self.verify(out, 'complete')

        out = self.root / 'append'
        self.write_listing(self.targets[:2])
        self.run_cli(out)
        self.write_listing()
        run = self.run_cli(out, fault='crash-before-publish', target='frame_000002.bin', success=False)
        self.assertEqual(run.returncode, 86)
        report = self.verify(out, 'complete')  # still the finished two-frame run
        self.assertEqual((report['frames_expected'], report['pending_append']), (2, True))
        self.run_cli(out)
        self.assertEqual(self.verify(out, 'complete')['frames_expected'], 4)


    def test_verify_only_reads_an_append_cut_off_after_its_first_new_frame(self):
        # The window between "first appended frame published" and "longer contract in force".
        out = self.root / 'out'
        self.write_listing(self.targets[:2])
        self.run_cli(out)
        self.write_listing()
        run = self.run_cli(out, fault='crash-after-publish', target='frame_000002.bin', success=False)
        self.assertEqual(run.returncode, 86)
        state = out / '.opencorr-state'
        self.assertTrue((out / 'frame_000002.bin').is_file())
        self.assertTrue((state / 'contract.pending.json').is_file())
        self.assertEqual(len(json.loads((state / 'contract.json').read_text())['targets']), 2)

        report = self.verify(out, 'incomplete')
        self.assertEqual((report['frames_expected'], report['frames_verified']), (2, 2))
        self.assertEqual((report['pending_append'], report['pending_append_state']), (True, 'append'))
        self.assertEqual((report['appended_published'], report['appended_published_count']), (['frame_000002.bin'], 1))
        for key in ('untracked_outputs', 'mismatched', 'no_intent', 'invalid_intent', 'missing'):
            self.assertEqual(report[key], [], key)
        # The kill came after link() and before the temporary name was removed.
        self.assertEqual(report['residue_count'], 1)
        self.assertEqual(len(report['warnings']), 1)
        self.assertIn('temporary files of an interrupted run', report['warnings'][0])
        self.assertEqual(len(report['problems']), 1)
        for phrase in ('rerun the append command', 'no --repair needed', 'shorter list is no longer accepted'):
            self.assertIn(phrase, report['problems'][0])

        # Only a frame that the pending contract promises and that matches its own
        # ledger record is an unfinished append. Everything short of that stays damaged.
        flipped = self.copy_of(out, 'flipped')
        data = bytearray((flipped / 'frame_000002.bin').read_bytes()); data[100] ^= 1
        (flipped / 'frame_000002.bin').write_bytes(data)
        report = self.verify(flipped, 'damaged')
        self.assertEqual((report['mismatched'], report['appended_published']), (['frame_000002.bin'], []))
        self.assertTrue(any('--repair' in problem for problem in report['problems']))

        orphan = self.copy_of(out, 'orphan')
        self.intent_of(orphan, 'frame/frame_000002.bin').unlink()
        report = self.verify(orphan, 'damaged')
        self.assertEqual((report['no_intent'], report['appended_published']), (['frame_000002.bin'], []))

        linked = self.copy_of(out, 'linked')
        (linked / 'frame_000002.bin').rename(linked / 'elsewhere')
        (linked / 'frame_000002.bin').symlink_to('elsewhere')
        report = self.verify(linked, 'damaged')
        self.assertEqual((report['mismatched'], report['appended_published']), (['frame_000002.bin'], []))

        unreadable = self.copy_of(out, 'unreadable')
        (unreadable / '.opencorr-state/contract.pending.json').write_text('{"targets": [')
        report = self.verify(unreadable, 'damaged')
        self.assertEqual((report['untracked_outputs'], report['pending_append_state'], report['appended_published']),
                         (['frame_000002.bin'], 'unreadable', []))

        unrelated = self.copy_of(out, 'unrelated')
        path = unrelated / '.opencorr-state/contract.pending.json'
        pending = json.loads(path.read_text()); pending['options']['max_iteration'] += 1
        path.write_text(json.dumps(pending))
        report = self.verify(unrelated, 'damaged')
        self.assertEqual((report['untracked_outputs'], report['pending_append_state']), (['frame_000002.bin'], 'unrelated'))

        reordered = self.copy_of(out, 'reordered')
        path = reordered / '.opencorr-state/contract.pending.json'
        pending = json.loads(path.read_text()); pending['targets'][0], pending['targets'][1] = pending['targets'][1], pending['targets'][0]
        path.write_text(json.dumps(pending))
        self.assertEqual(self.verify(reordered, 'damaged')['pending_append_state'], 'unrelated')

        foreign = self.copy_of(out, 'foreign')
        (foreign / 'frame_000009.bin').write_bytes(b'foreign')
        report = self.verify(foreign, 'damaged')
        self.assertEqual((report['untracked_outputs'], report['appended_published']), (['frame_000009.bin'], ['frame_000002.bin']))

        # What the verdict says is what happens: the short list is refused and
        # changes nothing, the long list continues without --repair.
        before = stat_snapshot(out)
        self.write_listing(self.targets[:2])
        refused = self.run_cli(out, success=False)
        self.assertIn('untracked entry in output directory', refused.stderr)
        self.assertEqual(stat_snapshot(out), before)
        self.write_listing()
        self.run_cli(out)
        report = self.verify(out, 'complete')
        self.assertEqual((report['frames_expected'], report['pending_append'], report['pending_append_state'],
                          report['appended_published']), (4, False, 'none', []))
        fresh = self.root / 'fresh'
        self.run_cli(fresh)
        self.assertEqual(hashes(out), hashes(fresh))
        # And the damaged variants are what --repair with the long list regenerates.
        for damaged in (flipped, orphan):
            self.run_cli(damaged, extra=['--repair'])
            self.assertEqual(hashes(damaged), hashes(fresh))
            self.verify(damaged, 'complete')

    def test_verify_only_reports_the_other_pending_contract_states(self):
        out = self.root / 'out'
        self.write_listing(self.targets[:2])
        self.run_cli(out)
        self.write_listing()
        self.assertEqual(self.run_cli(out, fault='crash-before-publish', target='frame_000002.bin',
                                      success=False).returncode, 86)
        report = self.verify(out, 'complete')
        self.assertEqual((report['pending_append_state'], report['appended_published']), ('append', []))
        self.assertIn('stopped before any appended output was published', ' | '.join(report['warnings']))

        promoted = self.copy_of(out, 'promoted')
        state = promoted / '.opencorr-state'
        (state / 'contract.pending.json').write_bytes((state / 'contract.json').read_bytes())
        report = self.verify(promoted, 'complete')
        self.assertEqual(report['pending_append_state'], 'promoted')
        self.assertIn('repeats the recorded contract', ' | '.join(report['warnings']))

        garbage = self.copy_of(out, 'garbage')
        (garbage / '.opencorr-state/contract.pending.json').write_text('{abandoned append')
        report = self.verify(garbage, 'complete')
        self.assertEqual(report['pending_append_state'], 'unreadable')
        self.assertIn('cannot be read as a contract', ' | '.join(report['warnings']))

    def test_verify_only_counts_entries_named_like_outputs_whatever_their_type(self):
        fresh = self.root / 'fresh'
        self.run_cli(fresh, extra=['--diagnostics'])
        self.verify(fresh, 'complete')
        cases = {
            'symlink': lambda out: (out / 'frame_000099.bin').symlink_to('frame_000000.bin'),
            'dangling-symlink': lambda out: (out / 'frame_000099.bin').symlink_to('nowhere'),
            'directory': lambda out: (out / 'frame_000099.bin').mkdir(),
            'file': lambda out: (out / 'frame_000099.bin').write_bytes(b'foreign'),
        }
        for label, plant in cases.items():
            out = self.copy_of(fresh, label)
            plant(out)
            report = self.verify(out, 'damaged')
            self.assertEqual((report['untracked_outputs'], report['untracked'], report['frames_verified']),
                             (['frame_000099.bin'], [], 4), label)
        # The same rule in the diagnostics directory, whose outputs are .csv files.
        out = self.copy_of(fresh, 'diagnostics-symlink')
        (out / 'diagnostics/frame_000099.diagnostics.csv').symlink_to('frame_000000.diagnostics.csv')
        self.assertEqual(self.verify(out, 'damaged')['untracked_outputs'], ['diagnostics/frame_000099.diagnostics.csv'])

        # Names that are not output-like are listed and do not change the verdict,
        # but the ledger's own rule is stricter: the original command no longer
        # accepts the directory. "complete" is a statement about the outputs.
        out = self.copy_of(fresh, 'other-names')
        (out / 'notes.txt').write_text('kept by hand\n')
        (out / 'sub').mkdir()
        (out / 'frame_000000.csv').write_text('not this run\'s format\n')
        report = self.verify(out, 'complete')
        self.assertEqual((sorted(report['untracked']), report['untracked_outputs']),
                         (['frame_000000.csv', 'notes.txt', 'sub'], []))
        before = stat_snapshot(out)
        rerun = self.run_cli(out, extra=['--diagnostics'], success=False)
        self.assertIn('untracked entry in output directory', rerun.stderr)
        self.assertEqual(stat_snapshot(out), before)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CliInterfaceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT / 'test-summary.json').write_text(json.dumps(
        {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
         'skips': len(result.skipped), 'pass': result.wasSuccessful()}, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
