#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""End-to-end output integrity tests; all evidence is kept in a unique directory."""
import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from generate_synthetic_dic import make_base, write_pgm

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--fault-lib', type=Path)
parser.add_argument('--work-dir', type=Path)
args = parser.parse_args()
args.binary = args.binary.resolve()
if args.fault_lib: args.fault_lib = args.fault_lib.resolve()
ROOT = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix='opencorr-output-integrity-'))
if args.work_dir: ROOT.mkdir(parents=True, exist_ok=False)
print('Evidence:', ROOT, flush=True)


def hashes(folder, suffix):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('frame_*' + suffix))}


def snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and not p.name.startswith('.opencorr-tmp-')}


class OutputIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self._testMethodName
        self.root.mkdir()
        self.inputs = self.root / '图像 inputs'
        self.inputs.mkdir()
        self.serial = 0
        width, height = 192, 128
        base = make_base(width, height, 20260916)
        write_pgm(self.inputs / 'ref.pgm', width, height, base)
        for i, shift in enumerate((.25, .65, 1.1, .4)):
            pixels = bytearray([127]) * (width * height)
            for y in range(height):
                for x in range(width):
                    sx = x - shift; ix = math.floor(sx); fraction = sx - ix
                    if 0 <= ix < width - 1:
                        pixels[y * width + x] = round((1 - fraction) * base[y * width + ix] + fraction * base[y * width + ix + 1])
            write_pgm(self.inputs / f'tar{i}.pgm', width, height, pixels)
        self.listing = self.root / 'targets.txt'
        self.set_targets(4)

    def set_targets(self, count, order=None):
        if order is None: order = range(count)
        self.listing.write_text(''.join(f'{self.inputs}/tar{i}.pgm frame_{j:06d}.csv\n' for j, i in enumerate(order)))

    def run_cli(self, out, fmt='raw', extra=(), fault=None, target='frame_000001.bin', success=True, cwd=None,
                durability=None, count=None, count_root=None):
        command = [str(args.binary), '--multi-frame', '--ref', str(self.inputs / 'ref.pgm'),
                   '--tar-list', str(self.listing), '--output-dir', str(out), '7', '5',
                   '--threads', '2', '--gpu-device', '0', '--output-format', fmt] + list(extra)
        if durability: command += ['--durability', durability]
        env = dict(os.environ)
        if fault or count:
            if not args.fault_lib: self.skipTest('requires Linux syscall fault library')
            env['LD_PRELOAD'] = str(args.fault_lib)
        if fault: env.update(OPENCORR_TEST_IO_FAULT=fault, OPENCORR_TEST_IO_TARGET=target)
        if count:
            env['OPENCORR_TEST_IO_COUNT'] = str(count)
            if count_root: env['OPENCORR_TEST_IO_COUNT_ROOT'] = str(count_root)
        run = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True,
                             errors='replace', timeout=120)
        self.serial += 1
        (self.root / f'run-{self.serial}.log').write_text(run.stdout + run.stderr)
        (self.root / f'run-{self.serial}.json').write_text(json.dumps({'command':command, 'fault':fault, 'target':target, 'rc':run.returncode}, indent=2)+'\n')
        if success: self.assertEqual(run.returncode, 0, run.stderr[-2500:])
        else: self.assertNotEqual(run.returncode, 0)
        return run

    def test_append_raw_in_frames_directory_preserves_prefix_and_history(self):
        out = self.root / 'frames'
        self.set_targets(2)
        self.run_cli('frames', cwd=self.root, extra=['--temporal-extrapolation'])
        old = {p.name:(p.stat().st_ino,p.stat().st_mtime_ns,p.read_bytes()) for p in out.glob('frame_*.bin')}
        manifest = out / 'manifest.opencorr-raw.json'
        manifest.write_text(json.dumps(json.loads(manifest.read_text()), ensure_ascii=True, indent=4))
        self.set_targets(4)
        self.run_cli('frames', cwd=self.root, extra=['--temporal-extrapolation'])
        fresh = self.root / 'fresh'
        self.run_cli(fresh, extra=['--temporal-extrapolation'])
        self.assertEqual(hashes(out,'.bin'), hashes(fresh,'.bin'))
        for name,(inode,mtime,data) in old.items():
            p=out/name; self.assertEqual((p.stat().st_ino,p.stat().st_mtime_ns,p.read_bytes()),(inode,mtime,data))
        before=snapshot(out)
        run=self.run_cli('frames', cwd=self.root, extra=['--temporal-extrapolation'])
        self.assertIn('Verified complete run',run.stderr)
        self.assertEqual(snapshot(out),before)

    def test_csv_resume_with_crop_and_diagnostics(self):
        extra=['--diagnostics','--output-roi','30,25,160,100','--output-step','2']
        out=self.root/'csv'
        self.set_targets(2); self.run_cli(out,'csv',extra)
        old=hashes(out,'.csv')
        self.set_targets(4); self.run_cli(out,'csv',extra)
        fresh=self.root/'fresh'; self.run_cli(fresh,'csv',extra)
        self.assertEqual(hashes(out,'.csv'),hashes(fresh,'.csv'))
        self.assertEqual(hashes(out/'diagnostics','.diagnostics.csv'),hashes(fresh/'diagnostics','.diagnostics.csv'))
        self.assertTrue(all(hashes(out,'.csv')[name]==value for name,value in old.items()))

    def test_zero_byte_untracked_output_is_rejected_and_preserved(self):
        out=self.root/'out';out.mkdir(); (out/'frame_000000.bin').write_bytes(b'')
        before=snapshot(out);self.run_cli(out,success=False)
        self.assertEqual(snapshot(out),before)

    def test_truncated_tracked_output_is_rejected_and_preserved(self):
        out=self.root/'out';self.run_cli(out)
        path=out/'frame_000001.bin';path.write_bytes(path.read_bytes()[:-1])
        before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_same_length_corruption_is_rejected(self):
        out=self.root/'out';self.run_cli(out)
        path=out/'frame_000001.bin'; data=bytearray(path.read_bytes());data[100]^=1;path.write_bytes(data)
        before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_changed_configuration_is_rejected(self):
        out=self.root/'out';self.run_cli(out)
        before=snapshot(out);self.run_cli(out,extra=['--max-iter','11'],success=False);self.assertEqual(snapshot(out),before)

    def test_changed_input_same_size_and_timestamp_is_rejected(self):
        out=self.root/'out';self.run_cli(out)
        path=self.inputs/'tar2.pgm';stat=path.stat();data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_reordered_targets_are_rejected(self):
        out=self.root/'out';self.run_cli(out)
        before=snapshot(out);self.set_targets(4,[1,0,2,3]);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_changed_version_record_is_rejected(self):
        out=self.root/'out';self.run_cli(out)
        path=out/'.opencorr-state/contract.json';j=json.loads(path.read_text());j['git_hash']='foreign';path.write_text(json.dumps(j))
        before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_malformed_and_duplicate_json_are_rejected(self):
        for label,content in [('truncated','{"frames":['),('duplicate','{"format":"x","format":"y","frames":[]}')]:
            out=self.root/label;self.run_cli(out)
            (out/'manifest.opencorr-raw.json').write_text(content)
            before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_semantically_changed_complete_manifest_is_rejected(self):
        out=self.root/'out';self.run_cli(out)
        path=out/'manifest.opencorr-raw.json';j=json.loads(path.read_text());j['frames']=[];path.write_text(json.dumps(j))
        before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)

    def test_missing_committed_file_is_recreated_by_replay(self):
        out=self.root/'out';self.run_cli(out);expected=hashes(out,'.bin')
        (out/'frame_000002.bin').unlink()
        self.run_cli(out);self.assertEqual(hashes(out,'.bin'),expected)

    def test_empty_filtered_raw_is_valid_only_with_intent(self):
        out=self.root/'out';extra=['--output-roi','500,500,600,600']
        self.run_cli(out,extra=extra)
        self.assertTrue(all(p.stat().st_size==0 for p in out.glob('frame_*.bin')))
        self.run_cli(out,extra=extra)

    def test_untracked_extra_frame_and_symlink_are_rejected(self):
        out=self.root/'out';self.run_cli(out)
        (out/'frame_999999.bin').write_bytes(b'foreign')
        before=snapshot(out);self.run_cli(out,success=False);self.assertEqual(snapshot(out),before)
        (out/'frame_999999.bin').unlink()
        p=out/'frame_000001.bin';copy=self.root/'outside';copy.write_bytes(p.read_bytes());p.unlink();p.symlink_to(copy)
        before=copy.read_bytes();self.run_cli(out,success=False);self.assertEqual(copy.read_bytes(),before)

    def test_unsafe_and_duplicate_output_names_are_rejected(self):
        for label,names in [('escape',['../escape.bin','ok.bin']),('duplicate',['same.csv','same.bin']),('reserved',['.opencorr-state','ok.bin'])]:
            self.listing.write_text(''.join(f'{self.inputs}/tar{i}.pgm {name}\n' for i,name in enumerate(names)))
            self.run_cli(self.root/label,success=False)
        self.assertFalse((self.root/'escape.bin').exists())

    def test_output_directory_lock_prevents_concurrent_run(self):
        out=self.root/'out';out.mkdir()
        fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.run_cli(out,success=False)
            self.assertFalse((out/'.opencorr-state').exists())
        finally: os.close(fd)

    def test_crash_windows_resume_to_identical_bytes(self):
        fresh=self.root/'fresh';self.run_cli(fresh)
        for policy in ('rename','fsync'):
            for mode in ('crash-before-publish','crash-after-publish','publish-error'):
                out=self.root/f'{policy}-{mode}';run=self.run_cli(out,fault=mode,success=False,durability=policy)
                if mode.startswith('crash'):self.assertEqual(run.returncode,86)
                self.assertFalse((out/'.opencorr-state/complete.json').exists())
                prefix={p.name:(p.stat().st_ino,p.read_bytes()) for p in out.glob('frame_*.bin')}
                self.assertTrue(all(len(d)==len(prefix[n][1]) for n,(_,d) in prefix.items()))
                self.run_cli(out,durability=policy)
                self.assertEqual(hashes(out,'.bin'),hashes(fresh,'.bin'))
                for name,(inode,data) in prefix.items(): self.assertEqual(((out/name).stat().st_ino,(out/name).read_bytes()),(inode,data))

    def test_short_write_sync_and_close_failure_never_publish_partial_file(self):
        for policy in ('rename','fsync'):
            for mode in ('short-write','close-error'):
                out=self.root/f'{policy}-{mode}';self.run_cli(out,fault=mode,success=False,durability=policy)
                self.assertFalse((out/'.opencorr-state/complete.json').exists())
                self.assertEqual(list(out.glob('frame_*.bin')),[])
                self.run_cli(out,durability=policy)
                self.assertEqual(len(list(out.glob('frame_*.bin'))),4)
        out=self.root/'fsync-error';self.run_cli(out,fault='fsync-error',success=False,durability='fsync')
        self.assertEqual(list(out.glob('frame_*.bin')),[])
        self.run_cli(out,durability='fsync');self.assertEqual(len(list(out.glob('frame_*.bin'))),4)
        # rename issues no payload fsync, so the same injection cannot fire at all
        clean=self.root/'fsync-error-rename';self.run_cli(clean,fault='fsync-error',durability='rename')
        self.assertEqual(hashes(clean,'.bin'),hashes(out,'.bin'))

    def sync_calls(self, label, targets, durability):
        report=self.root/f'fsync-{label}.txt';out=(self.root/label).resolve()
        self.set_targets(targets);self.run_cli(out,durability=durability,count=report,count_root=out)
        return sum(int(v) for v in report.read_text().split()),out

    def test_rename_policy_costs_no_fsync_per_frame(self):
        rename_2,_=self.sync_calls('rename2',2,'rename')
        rename_4,rename_dir=self.sync_calls('rename4',4,'rename')
        fsync_2,_=self.sync_calls('fsync2',2,'fsync')
        fsync_4,fsync_dir=self.sync_calls('fsync4',4,'fsync')
        (self.root/'fsync-counts.json').write_text(json.dumps(
            {'rename_2':rename_2,'rename_4':rename_4,'fsync_2':fsync_2,'fsync_4':fsync_4},indent=2)+'\n')
        self.assertEqual(rename_4-rename_2,0)          # zero fsync per published frame
        self.assertLess(rename_4,8)                    # only the constant completion barrier
        self.assertGreaterEqual(fsync_4-fsync_2,8)     # at least four fsync per published frame
        self.assertEqual(hashes(rename_dir,'.bin'),hashes(fsync_dir,'.bin'))

    def test_durability_choice_is_not_part_of_the_resume_contract(self):
        out=self.root/'out';self.set_targets(2);self.run_cli(out,durability='fsync')
        old=hashes(out,'.bin')
        self.set_targets(4);self.run_cli(out,durability='rename')
        fresh=self.root/'fresh';self.run_cli(fresh,durability='rename')
        self.assertEqual(hashes(out,'.bin'),hashes(fresh,'.bin'))
        self.assertTrue(all(hashes(out,'.bin')[name]==value for name,value in old.items()))
        run=self.run_cli(out,extra=['--durability','nope'],success=False)
        self.assertIn('--durability must be rename or fsync',run.stderr)

    def test_manifests_record_the_effective_durability(self):
        for policy in ('rename','fsync'):
            out=self.root/policy;self.run_cli(out,extra=['--diagnostics'],durability=policy)
            self.assertEqual(json.loads((out/'manifest.opencorr-raw.json').read_text())['durability'],policy)
            diagnostics=out/'diagnostics/manifest.opencorr-diagnostics.json'
            self.assertEqual(json.loads(diagnostics.read_text())['durability'],policy)
            self.assertEqual(json.loads((out/'.opencorr-state/complete.json').read_text())['durability'],policy)

    def test_repair_regenerates_only_the_damaged_frame(self):
        fresh=self.root/'fresh';self.run_cli(fresh)
        out=self.root/'out';self.run_cli(out);expected=hashes(out,'.bin')
        self.assertEqual(expected,hashes(fresh,'.bin'))
        keep={p.name:(p.stat().st_ino,p.stat().st_mtime_ns) for p in out.glob('frame_*.bin') if p.name!='frame_000002.bin'}
        broken=out/'frame_000002.bin';size=broken.stat().st_size;broken.write_bytes(broken.read_bytes()[:-4])
        run=self.run_cli(out,success=False);self.assertIn('--repair',run.stderr)
        run=self.run_cli(out,extra=['--repair']);self.assertIn('Repair: quarantined',run.stderr)
        self.assertEqual(hashes(out,'.bin'),expected)
        for name,(inode,mtime) in keep.items():
            p=out/name;self.assertEqual((p.stat().st_ino,p.stat().st_mtime_ns),(inode,mtime))
        moved=sorted((out/'.opencorr-state/quarantine').rglob('frame_000002.bin'))
        self.assertEqual(len(moved),1);self.assertEqual(moved[0].stat().st_size,size-4)
        before=snapshot(out);run=self.run_cli(out)
        self.assertIn('Verified complete run',run.stderr);self.assertEqual(snapshot(out),before)

    def test_repair_adopts_missing_intent_and_crash_temporaries(self):
        out=self.root/'out';self.run_cli(out);expected=hashes(out,'.bin')
        state=out/'.opencorr-state'
        intents=sorted(p for p in state.glob('*.json') if p.name not in ('contract.json','complete.json'))
        self.assertEqual(len(intents),4);intents[0].unlink()
        (out/'.opencorr-tmp-999999-7').write_bytes(b'crash residue')
        self.run_cli(out,success=False)
        self.run_cli(out,extra=['--repair'])
        self.assertEqual(hashes(out,'.bin'),expected)
        self.assertFalse((out/'.opencorr-tmp-999999-7').exists())
        self.assertEqual(len(list((state/'quarantine').rglob('.opencorr-tmp-999999-7'))),1)

    def test_repair_still_rejects_foreign_files_symlinks_and_changed_options(self):
        out=self.root/'out';self.run_cli(out)
        (out/'frame_999999.bin').write_bytes(b'foreign')
        before=snapshot(out);self.run_cli(out,extra=['--repair'],success=False);self.assertEqual(snapshot(out),before)
        (out/'frame_999999.bin').unlink()
        self.run_cli(out,extra=['--repair','--max-iter','11'],success=False)
        self.set_targets(4,[1,0,2,3]);self.run_cli(out,extra=['--repair'],success=False);self.set_targets(4)
        p=out/'frame_000001.bin';copy=self.root/'outside';copy.write_bytes(p.read_bytes());p.unlink();p.symlink_to(copy)
        kept=copy.read_bytes();self.run_cli(out,extra=['--repair'],success=False)
        self.assertEqual(copy.read_bytes(),kept)

    def test_failed_append_can_still_resume_the_original_target_list(self):
        out=self.root/'out';self.set_targets(2);self.run_cli(out);two=hashes(out,'.bin')
        self.set_targets(4)
        run=self.run_cli(out,fault='crash-before-publish',target='frame_000002.bin',success=False)
        self.assertEqual(run.returncode,86)
        self.assertTrue((out/'.opencorr-state/contract.pending.json').exists())
        self.assertEqual(hashes(out,'.bin'),two)
        self.set_targets(2);run=self.run_cli(out)
        self.assertIn('Verified complete run',run.stderr);self.assertEqual(hashes(out,'.bin'),two)
        self.set_targets(4);self.run_cli(out)
        self.assertFalse((out/'.opencorr-state/contract.pending.json').exists())
        fresh=self.root/'fresh';self.run_cli(fresh)
        self.assertEqual(hashes(out,'.bin'),hashes(fresh,'.bin'))

    def test_non_utf8_output_directory_fails_before_any_write(self):
        broken=self.root/os.fsdecode(b'out-\xff\xfe')
        run=self.run_cli(broken,success=False)
        self.assertIn('non-UTF-8',run.stderr)
        self.assertFalse(broken.exists())

    def test_failed_manifest_publication_preserves_previous_manifest(self):
        out=self.root/'out';self.set_targets(1);self.run_cli(out)
        manifest=out/'manifest.opencorr-raw.json';old=manifest.read_bytes()
        self.set_targets(4);self.run_cli(out,fault='metadata-error',target=manifest.name,success=False)
        self.assertEqual(manifest.read_bytes(),old)
        self.run_cli(out)
        self.assertEqual(len(json.loads(manifest.read_text())['frames']),4)

    def test_complete_resume_archives_crash_residue_and_pending_contract(self):
        out=self.root/'out';self.run_cli(out);before=hashes(out,'.bin')
        state=out/'.opencorr-state'
        (out/'.opencorr-tmp-12345-7').write_bytes(b'orphan payload')
        (state/'.opencorr-tmp-12345-8').write_bytes(b'orphan intent')
        (state/'contract.pending.json').write_text('{abandoned append')
        run=self.run_cli(out)
        self.assertIn('Verified complete run',run.stderr)
        self.assertEqual(hashes(out,'.bin'),before)
        for name in ('.opencorr-tmp-12345-7','.opencorr-tmp-12345-8','contract.pending.json'):
            self.assertFalse((out/name).exists());self.assertFalse((state/name).exists())
            self.assertEqual(len(list((state/'quarantine').rglob(name))),1)

    def test_repair_invalid_completion_preserves_verified_payloads(self):
        for label in ('syntax','schema'):
            out=self.root/label;self.run_cli(out);before=hashes(out,'.bin')
            stats={p.name:(p.stat().st_ino,p.stat().st_mtime_ns) for p in out.glob('frame_*.bin')}
            complete=out/'.opencorr-state/complete.json';complete.write_text('{truncated' if label=='syntax' else '{}')
            rejected=self.run_cli(out,success=False);self.assertIn('--repair',rejected.stderr)
            self.run_cli(out,extra=['--repair'])
            self.assertEqual(hashes(out,'.bin'),before)
            self.assertEqual(stats,{p.name:(p.stat().st_ino,p.stat().st_mtime_ns) for p in out.glob('frame_*.bin')})
            self.assertEqual(len(list((out/'.opencorr-state/quarantine').rglob('complete.json'))),1)
            self.run_cli(out)

    def test_resume_skips_reserved_names_that_are_not_regular_files(self):
        out=self.root/'out';self.set_targets(2);self.run_cli(out);two=hashes(out,'.bin')
        state=out/'.opencorr-state'
        (out/'.opencorr-tmp-9999999-1').mkdir()
        (state/'.opencorr-tmp-9999999-2').symlink_to(self.inputs/'ref.pgm')
        (out/'.opencorr-tmp-9999999-3').write_bytes(b'orphan payload')
        run=self.run_cli(out)
        self.assertIn('Verified complete run',run.stderr)
        self.assertIn('not a regular file',run.stderr)
        self.assertIn('Archived 1 abandoned private file(s)',run.stderr)
        self.assertEqual(hashes(out,'.bin'),two)
        self.assertFalse((out/'.opencorr-tmp-9999999-3').exists())
        self.assertEqual(len(list((state/'quarantine').rglob('.opencorr-tmp-9999999-3'))),1)
        self.set_targets(4);run=self.run_cli(out)  # exercises the sweep after finish()
        self.assertIn('not a regular file',run.stderr)
        self.assertTrue((out/'.opencorr-tmp-9999999-1').is_dir())
        self.assertTrue((state/'.opencorr-tmp-9999999-2').is_symlink())
        fresh=self.root/'fresh';self.run_cli(fresh)
        self.assertEqual(hashes(out,'.bin'),hashes(fresh,'.bin'))

    def test_unreadable_metadata_is_an_io_fault_not_damage(self):
        if os.geteuid()==0: self.skipTest('chmod 000 does not deny reads to root')
        out=self.root/'out';self.set_targets(2);self.run_cli(out);before=hashes(out,'.bin')
        state=out/'.opencorr-state';complete=state/'complete.json'
        stats={p.name:(p.stat().st_ino,p.stat().st_mtime_ns) for p in out.glob('frame_*.bin')}
        complete.chmod(0o000)
        try:
            for extra in ([],['--repair']):
                run=self.run_cli(out,extra=extra,success=False)
                self.assertEqual(run.returncode,1)
                self.assertIn(str(complete),run.stderr)
                self.assertIn('errno=%d'%errno.EACCES,run.stderr)
                self.assertIn(os.strerror(errno.EACCES),run.stderr)
                self.assertNotIn('use --repair',run.stderr)
                self.assertFalse((state/'quarantine').exists())
        finally:
            complete.chmod(0o600)
        run=self.run_cli(out)
        self.assertIn('Verified complete run',run.stderr)
        self.assertEqual(hashes(out,'.bin'),before)
        self.assertEqual(stats,{p.name:(p.stat().st_ino,p.stat().st_mtime_ns) for p in out.glob('frame_*.bin')})
        complete.write_text('{truncated')
        self.assertIn('--repair',self.run_cli(out,success=False).stderr)
        self.run_cli(out,extra=['--repair'])
        self.assertEqual(hashes(out,'.bin'),before)
        self.assertEqual(len(list((state/'quarantine').rglob('complete.json'))),1)

    def test_reserved_names_that_are_not_regular_files_survive_both_modes(self):
        out=self.root/'out';self.set_targets(2);self.run_cli(out);two=hashes(out,'.bin')
        state=out/'.opencorr-state'
        (out/'.opencorr-tmp-8888888-1').mkdir()
        (state/'.opencorr-tmp-8888888-2').symlink_to(self.inputs/'ref.pgm')
        (out/'.opencorr-tmp-nondigits').write_bytes(b'malformed reserved name')
        (out/'.opencorr-tmp-8888888-3').write_bytes(b'orphan payload')
        kept=(out/'.opencorr-tmp-8888888-1',state/'.opencorr-tmp-8888888-2',out/'.opencorr-tmp-nondigits')
        marks={p:(p.lstat().st_ino,p.lstat().st_mtime_ns) for p in kept}
        for extra in (['--repair'],[]):
            run=self.run_cli(out,extra=extra)
            self.assertIn('Verified complete run',run.stderr)
            self.assertIn('not a regular file',run.stderr)
            self.assertIn('unrecognized reserved name',run.stderr)
            self.assertEqual(hashes(out,'.bin'),two)
            self.assertTrue((out/'.opencorr-tmp-8888888-1').is_dir())
            self.assertTrue((state/'.opencorr-tmp-8888888-2').is_symlink())
            self.assertEqual(marks,{p:(p.lstat().st_ino,p.lstat().st_mtime_ns) for p in kept})
        self.assertFalse((out/'.opencorr-tmp-8888888-3').exists())
        self.assertEqual(len(list((state/'quarantine').rglob('.opencorr-tmp-8888888-3'))),1)

    def test_async_diagnostics_failure_is_propagated_and_resumable(self):
        for policy in ('rename','fsync'):
            extra=['--diagnostics','--durability',policy]
            out=self.root/policy
            self.run_cli(out,extra=extra,fault='publish-error',target='frame_000001.diagnostics.csv',success=False)
            self.assertFalse((out/'.opencorr-state/complete.json').exists())
            self.run_cli(out,extra=extra)
            fresh=self.root/(policy+'-fresh');self.run_cli(fresh,extra=extra)
            self.assertEqual(hashes(out,'.bin'),hashes(fresh,'.bin'))
            self.assertEqual(hashes(out/'diagnostics','.csv'),hashes(fresh/'diagnostics','.csv'))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(OutputIntegrityTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'test-summary.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'pass':result.wasSuccessful()},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
