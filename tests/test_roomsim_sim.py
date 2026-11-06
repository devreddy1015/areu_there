"""Room Sim runs: schedule, impairments, datagrams, sequence numbers,
draw-sequence independence and export (spec/roomsim.md §10, §12-13, §15)."""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roomsim_helpers import BEDROOM, HALL, MULTI, make_scene  # noqa: E402

from wifisense.cli import main as cli_main  # noqa: E402
from wifisense.roomsim import (  # noqa: E402
    BUFFER_ACTIVE, BUFFER_K, Config, RoomSim, export_run, load_scene, manifest,
)
from wifisense.roomsim.impair import _smooth  # noqa: E402
from wifisense.sources.esp32_udp import build_esp32_frame, parse_esp32_frame  # noqa: E402
from wifisense.sources.recorder import PcapWriter, iter_frames, read_pcap  # noqa: E402


def run(path=BEDROOM, **kw):
    kw.setdefault("duration_s", 0.25)
    sim = RoomSim(load_scene(path), Config(**kw))
    return sim, list(sim.transmissions())


def packets(trs):
    return [p for tr in trs for p in tr.packets]


class TestBuffer(unittest.TestCase):
    def test_buffer_k_and_active(self):
        fft = list(range(32)) + list(range(-32, 0))
        self.assertEqual(list(BUFFER_K), fft + fft)
        self.assertEqual(sum(BUFFER_ACTIVE[:64]), 52)
        self.assertEqual(sum(BUFFER_ACTIVE[64:]), 56)
        for e, k in enumerate(BUFFER_K):
            want = (1 <= abs(k) <= 26) if e < 64 else (1 <= abs(k) <= 28)
            self.assertEqual(BUFFER_ACTIVE[e], want)
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0))
        self.assertEqual(sim.buffer_k, BUFFER_K)
        self.assertEqual(sim.buffer_active, BUFFER_ACTIVE)

    def test_nulls_are_exactly_zero(self):
        _sim, trs = run(duration_s=0.1)
        for p in packets(trs):
            if p.lost:
                continue
            for e in range(128):
                if not BUFFER_ACTIVE[e]:
                    self.assertEqual(p.buffer_pre[e], (0.0, 0.0))
                    self.assertEqual(p.buffer[e], (0, 0))

    def test_smoothing_formula(self):
        buf = [(float(i), float(-2 * i)) for i in range(128)]
        idx = [3, 4, 5, 6]
        out = _smooth(buf, idx, (0.25, 0.5, 0.25), (2.0 / 3.0, 1.0 / 3.0))
        self.assertEqual(out[3][0], 2.0 / 3.0 * 3.0 + 1.0 / 3.0 * 4.0)
        self.assertEqual(out[4][0], (0.25 * 3.0 + 0.5 * 4.0) + 0.25 * 5.0)
        self.assertEqual(out[6][1], 2.0 / 3.0 * -12.0 + 1.0 / 3.0 * -10.0)
        self.assertEqual(out[7], buf[7])


class TestSchedule(unittest.TestCase):
    def test_order_times_and_tx_period(self):
        sim, trs = run(MULTI, duration_s=0.5, tx_period={1: 2})
        nT = 3
        self.assertEqual(sim.transmitters, [0, 1, 2])
        self.assertEqual(sim.links, [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        last = (-1, -1)
        for tr in trs:
            self.assertEqual(tr.t, float(tr.m * nT + tr.slot) * 0.0125)
            self.assertLess(tr.t, 0.5)
            self.assertGreater((tr.m, tr.slot), last)
            last = (tr.m, tr.slot)
            if tr.tx == 1:
                self.assertEqual(tr.m % 2, 0)
            self.assertEqual([p.rx for p in tr.packets], [n for n in (0, 1, 2) if n != tr.tx])
            for p in tr.packets:
                self.assertEqual(p.link_index, sim.links.index((p.tx, p.rx)))
                self.assertEqual(p.label, "SIM")
            self.assertEqual(tr.truth.label, "SIM")
        self.assertEqual(sim.per_link_rate_hz(0), 1.0 / ((3.0 * 0.0125) * 1.0))
        self.assertEqual(sim.per_link_rate_hz(1), 1.0 / ((3.0 * 0.0125) * 2.0))

    def test_roles(self):
        sim, trs = run(BEDROOM, duration_s=0.2, roles={0: "tx", 1: "rx", 2: "rx", 3: "rx"})
        self.assertEqual(sim.transmitters, [0])
        self.assertEqual(sim.links, [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(len(trs), 16)
        self.assertEqual(trs[1].t, 0.0125)


class TestSequence(unittest.TestCase):
    def test_seq_per_receiver_counts_received_packets(self):
        _sim, trs = run(HALL, duration_s=0.5, loss_prob=0.3)
        pk = packets(trs)
        lost = [p for p in pk if p.lost]
        self.assertGreater(len(lost), 20)
        expect = {}
        for p in pk:
            if p.lost:
                self.assertIsNone(p.seq)
                self.assertIsNone(p.datagram)
                self.assertIsNone(p.H)
                continue
            self.assertEqual(p.seq, expect.get(p.rx, 0))
            expect[p.rx] = p.seq + 1

    def test_loss_prob_link(self):
        _sim, trs = run(BEDROOM, duration_s=0.5, loss_prob=0.0, loss_prob_link={"0,1": 1.0})
        for p in packets(trs):
            self.assertEqual(p.lost, (p.tx, p.rx) == (0, 1))


class TestImpairments(unittest.TestCase):
    def test_impair_false_is_proportional_to_H(self):
        _sim, trs = run(MULTI, duration_s=0.2, impair=False, loss_prob=0.5)
        for p in packets(trs):
            self.assertFalse(p.lost)
            mags = []
            for e in range(128):
                if not BUFFER_ACTIVE[e]:
                    continue
                k = BUFFER_K[e]
                h = p.H[k + 32]
                b = p.buffer_pre[e]
                mags.append(math.sqrt(b[0] * b[0] + b[1] * b[1]) / math.sqrt(h[0] * h[0] + h[1] * h[1]))
                # same phase too: buffer = c * H
                self.assertAlmostEqual(b[0] * h[1] - b[1] * h[0], 0.0, delta=1e-9 * abs(b[0] * h[0] + b[1] * h[1]))
            for m in mags:
                self.assertAlmostEqual(m / mags[0], 1.0, delta=1e-12)
            self.assertEqual(p.rssi, max(-100, min(0, round(p.rssi_true))))

    def test_rssi_and_counts_are_plausible(self):
        _sim, trs = run(BEDROOM, duration_s=0.25)
        for p in packets(trs):
            if p.lost:
                continue
            self.assertTrue(-100 <= p.rssi <= 0)
            self.assertTrue(-97 <= p.noise_floor <= -92)
            self.assertTrue(-80.0 < p.rssi_true < 0.0)
            for re, im in p.buffer:
                self.assertTrue(-128 <= re <= 127 and -128 <= im <= 127)
                self.assertIsInstance(re, int)

    def test_quantize_false(self):
        _sim, trs = run(BEDROOM, duration_s=0.1, quantize=False)
        for p in packets(trs):
            if not p.lost:
                self.assertIsNone(p.datagram)
                self.assertEqual(p.buffer, p.buffer_pre)

    def test_every_datagram_parses(self):
        for path, kw in ((BEDROOM, {}), (MULTI, {"fw_subcarriers": 64, "channel": 11}), (HALL, {"fw_subcarriers": 8})):
            sim, trs = run(path, duration_s=0.15, **kw)
            n = 0
            for p in packets(trs):
                if p.lost:
                    continue
                f = parse_esp32_frame(p.datagram)
                n += 1
                self.assertEqual(f.node_id, p.rx)
                self.assertEqual(f.n_antennas, 1)
                self.assertEqual(f.n_subcarriers, sim.cfg.fw_subcarriers)
                self.assertEqual(f.freq_mhz, 2412 + 5 * (sim.cfg.channel - 1))
                self.assertEqual(f.sequence, p.seq)
                self.assertEqual(f.rssi_dbm, p.rssi)
                self.assertEqual(f.noise_floor_dbm, p.noise_floor)
                self.assertEqual((f.ppdu_type, f.flags), (0, 0))
                self.assertEqual(f.iq, [(im, re) for re, im in p.buffer[: sim.cfg.fw_subcarriers]])
                self.assertEqual(len(p.datagram), 20 + 2 * sim.cfg.fw_subcarriers)
            self.assertGreater(n, 10)


class TestDrawSequences(unittest.TestCase):
    """§2 / §12.5: switches change how a drawn value is used, never what is drawn."""

    def fingerprint(self, **kw):
        sim, trs = run(MULTI, duration_s=0.3, **kw)
        pk = packets(trs)
        devs = [(d.alpha, d.beta, d.gamma, d.delta, d.g_dev_db, d.nf_dbm, d.tau_dev) for d in sim.devices.values()]
        truth = [(tr.t, [(p.x, p.y, p.heading, p.walking) for p in tr.truth.people]) for tr in trs]
        clutter = [(c.pos, c.sigma) for c in sim.clutter]
        return {
            "lost": [p.lost for p in pk], "agc": [p.agc_db for p in pk], "seq": [p.seq for p in pk],
            "devs": devs, "truth": truth, "clutter": clutter,
            "tail": sim._tail_draws,
        }

    def test_switches_never_change_draws(self):
        base = self.fingerprint()
        switches = [
            dict(ripple=False), dict(cpe=False), dict(snr_noise=False), dict(ltf_merge=False),
            dict(smoothing=True), dict(quantize=False), dict(tail=False), dict(shadowing=False),
            dict(transmission=False), dict(object_specular=False), dict(object_scatter=False),
            dict(object_blockage=False), dict(floor_ceiling=False), dict(reflection_order=0),
        ]
        for sw in switches:
            fp = self.fingerprint(**sw)
            for key in base:
                self.assertEqual(fp[key], base[key], f"{sw}: {key}")
        off = self.fingerprint(impair=False)
        self.assertFalse(any(off["lost"]))
        self.assertEqual(off["agc"], base["agc"])
        for key in ("devs", "truth", "clutter", "tail"):
            self.assertEqual(off[key], base[key])

    def test_runs_repeat(self):
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.1))
        a = [p.datagram for p in sim.packets()]
        b = [p.datagram for p in sim.packets()]
        self.assertEqual(a, b)


class TestExport(unittest.TestCase):
    def test_export_reads_back(self):
        sim = RoomSim(load_scene(MULTI), Config(duration_s=0.5, tx_period={1: 2}))
        with tempfile.TemporaryDirectory() as d:
            prefix = os.path.join(d, "run")
            info = export_run(sim, prefix)
            recs = list(read_pcap(info["pcap"]))
            self.assertEqual(len(recs), info["datagrams"])
            want = [p for p in sim.packets() if not p.lost]
            self.assertEqual(len(recs), len(want))
            for r, p in zip(recs, want):
                self.assertEqual(r.payload, p.datagram)
                self.assertEqual(r.src, (f"10.1.{p.rx}.1", 5566))
                self.assertEqual(r.dst, ("10.0.0.1", 5566))
                self.assertEqual(r.t_ns, 1_790_000_000_000_000_000 + round(p.t * 1e9))
            self.assertEqual(len(list(iter_frames(info["pcap"]))), len(recs))
            with open(info["manifest"]) as fh:
                man = json.load(fh)
            self.assertEqual(man["label"], "SIM")
            self.assertEqual(man["spec"], "roomsim v1")
            self.assertEqual(man["scene_name"], "roomsim-multi-fixture")
            self.assertEqual(man["scene_sha256"], sim.scene.sha256)
            self.assertEqual(man["config"]["tx_period"], {"1": 2})
            self.assertEqual(len(man["config"]), len(Config().to_dict()))
            self.assertIn("assumptions", man)
            self.assertAlmostEqual(man["per_link_rate_hz"]["1,0"], 1.0 / (3 * 0.0125 * 2))
            # <prefix>.manifest.json, so two exports can share a directory;
            # links with their packet counts, and the run's totals.
            self.assertEqual(info["manifest"], prefix + ".manifest.json")
            self.assertEqual([(lk["tx"], lk["rx"]) for lk in man["links"]], list(sim.links))
            self.assertEqual([lk["link_index"] for lk in man["links"]], list(range(len(sim.links))))
            all_pk = list(sim.packets())
            for lk in man["links"]:
                mine = [p for p in all_pk if p.link_index == lk["link_index"]]
                self.assertEqual(lk["packets"], len(mine))
                self.assertEqual(lk["lost"], sum(p.lost for p in mine))
                self.assertEqual(lk["datagrams"], sum(not p.lost for p in mine))
            self.assertEqual(man["counts"], {
                "transmissions": info["transmissions"], "packets": len(all_pk),
                "lost": sum(p.lost for p in all_pk), "datagrams": len(recs),
            })
            with open(info["truth"]) as fh:
                lines = [json.loads(x) for x in fh]
            self.assertEqual(len(lines), info["transmissions"])
            self.assertEqual(lines[0]["people"][2]["name"], "people[2]")
            # no Person.id or Phone.id in any output (§3.1)
            with open(info["truth"]) as fh:
                text = fh.read() + json.dumps(man)
            for pid in ('"first"', '"second"', '"third"', '"cabinet"'):
                self.assertNotIn(pid, text)
            self.assertEqual(list(man["receiver_rate_hz"]), ["0", "1", "2"])
            self.assertIs(man["gateway_faithful"], False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli_main(["stats", info["pcap"]])
            self.assertEqual(rc, 0)
            self.assertIn("node", out.getvalue().lower())
            self.assertTrue(out.getvalue().startswith("SIM: "))
            with self.assertRaises(FileExistsError):
                export_run(sim, prefix)

    def test_source_address_is_a_convention_per_receiver(self):
        from wifisense.roomsim.export import GATEWAY, source_address
        self.assertEqual(source_address(0), ("10.1.0.1", 5566))
        self.assertEqual(source_address(254), ("10.1.254.1", 5566))
        self.assertEqual(GATEWAY, ("10.0.0.1", 5566))
        for rx in range(255):
            self.assertNotEqual(source_address(rx)[0], GATEWAY[0])
            self.assertFalse(source_address(rx)[0].endswith(".0"))

    def test_manifest_receiver_rates_and_gateway_faithful(self):
        # three transmitters, node 1 every other superframe
        sim = RoomSim(load_scene(MULTI), Config(duration_s=0.0, tx_period={1: 2}))
        man = manifest(sim)
        r0 = 1.0 / ((3.0 * 0.0125) * 1.0)
        r1 = 1.0 / ((3.0 * 0.0125) * 2.0)
        # receiver -> plain sum over its links, in link order, of their rates
        self.assertEqual(man["receiver_rate_hz"], {"0": r1 + r0, "1": r0 + r0, "2": r0 + r1})
        self.assertEqual(list(man["receiver_rate_hz"]), ["0", "1", "2"])
        self.assertIs(man["gateway_faithful"], False)
        # today's firmware: one transmitter at 20 Hz (slot_s = 0.05, §10)
        roles = {0: "tx", 1: "rx", 2: "rx", 3: "rx"}
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0, roles=roles, slot_s=0.05))
        man = manifest(sim)
        self.assertEqual(man["receiver_rate_hz"], {"1": 20.0, "2": 20.0, "3": 20.0})
        self.assertIs(man["gateway_faithful"], True)
        self.assertEqual(man["per_link_rate_hz"], {"0,1": 20.0, "0,2": 20.0, "0,3": 20.0})
        # a lone transmitter that is also a receiver hears nobody: 0.0
        roles = {0: "txrx", 1: "rx", 2: "rx", 3: "rx"}
        man = manifest(RoomSim(load_scene(BEDROOM), Config(duration_s=0.0, roles=roles)))
        self.assertEqual(man["receiver_rate_hz"], {"0": 0.0, "1": 80.0, "2": 80.0, "3": 80.0})
        self.assertIs(man["gateway_faithful"], True)
        # every receiver of the all-txrx mesh hears the other three
        man = manifest(RoomSim(load_scene(BEDROOM), Config(duration_s=0.0)))
        self.assertEqual(man["receiver_rate_hz"], {str(n): 20.0 + 20.0 + 20.0 for n in range(4)})
        self.assertNotIn("assumptions", man)

    def test_assumptions_spell_numbers_as_python_floats(self):
        sim = RoomSim(make_scene(), Config.from_dict({"duration_s": 0, "implied_wall_thickness_m": 1}))
        (a,) = manifest(sim)["assumptions"]
        self.assertEqual(a, "SIM: the scene lists no walls; the 4 footprint edges are walls of drywall, "
                            "1.0 m thick, full height")
        (a,) = manifest(RoomSim(load_scene(MULTI), Config(duration_s=0.0)))["assumptions"]
        self.assertTrue(a.startswith("SIM: "))
        self.assertIn(", 0.1 m thick", a)


class TestExportFiles(unittest.TestCase):
    """§13: refuse to start when any of the three files exists, unless
    overwrite=True, which first deletes the old manifest; the manifest is
    written last."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.prefix = os.path.join(self._tmp.name, "run")
        self.paths = [self.prefix + ext for ext in (".sim.pcap", ".truth.jsonl", ".manifest.json")]
        self.sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.1))

    def tearDown(self):
        self._tmp.cleanup()

    def touch(self, path, text="old"):
        with open(path, "w") as fh:
            fh.write(text)

    def test_refuses_naming_the_first_existing_file_and_opens_nothing(self):
        for present in ([1], [2], [1, 2], [0, 2], [0, 1, 2]):
            for path in self.paths:
                if os.path.exists(path):
                    os.remove(path)
            for i in present:
                self.touch(self.paths[i])
            with self.assertRaises(FileExistsError) as cm:
                export_run(self.sim, self.prefix)
            self.assertEqual(cm.exception.filename, self.paths[present[0]], present)
            for i, path in enumerate(self.paths):
                self.assertEqual(os.path.exists(path), i in present, (present, path))
                if i in present:
                    with open(path) as fh:
                        self.assertEqual(fh.read(), "old")

    def test_overwrite_replaces_all_three(self):
        export_run(self.sim, self.prefix)
        for path in self.paths:
            self.touch(path)
        info = export_run(self.sim, self.prefix, overwrite=True)
        with open(info["manifest"]) as fh:
            self.assertEqual(json.load(fh)["label"], "SIM")
        self.assertEqual(len(list(read_pcap(info["pcap"]))), info["datagrams"])
        with open(info["truth"]) as fh:
            self.assertEqual(sum(1 for _ in fh), info["transmissions"])

    def test_a_failed_run_never_leaves_a_manifest(self):
        export_run(self.sim, self.prefix)
        real = self.sim.transmissions

        def broken():
            it = real()
            yield next(it)
            raise RuntimeError("cut off part-way")

        self.sim.transmissions = broken
        with self.assertRaises(RuntimeError):
            export_run(self.sim, self.prefix, overwrite=True)
        self.assertFalse(os.path.exists(self.paths[2]))
        # and a fresh prefix: the manifest is written last, after the run
        with self.assertRaises(RuntimeError):
            export_run(self.sim, self.prefix + "2")
        self.assertTrue(os.path.exists(self.prefix + "2.sim.pcap"))
        self.assertFalse(os.path.exists(self.prefix + "2.manifest.json"))


class TestSimBanner(unittest.TestCase):
    """`wifisense stats` / `replay` print a SIM banner first for a Room Sim
    export: a `.sim.pcap`, or one whose PREFIX.manifest.json says SIM and
    whose counts.datagrams equals the file's record count (spec/roomsim.md
    §15, §16)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = cls._tmp.name
        # The default bedroom has four txrx nodes: not gateway-faithful (§12.4).
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=2.0, loss_prob=0.2))
        cls.info = export_run(sim, os.path.join(cls.dir, "run"))
        with open(cls.info["manifest"]) as fh:
            man = json.load(fh)
        cls.lost = man["counts"]["lost"]
        cls.datagrams = man["counts"]["datagrams"]
        cls.n_tx = len({ln["tx"] for ln in man["links"]})
        assert man["gateway_faithful"] is False and cls.n_tx == 4
        with open(cls.info["pcap"], "rb") as fh:
            cls.pcap_bytes = fh.read()
        assert len(list(read_pcap(cls.info["pcap"]))) == cls.datagrams

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(list(argv))
        return rc, out.getvalue()

    @staticmethod
    def banner(path, lost=None, n_tx=None):
        line = (f"SIM: {path} is a Room Sim export, not a measurement; its seq-based loss cannot see "
                "simulated loss (spec/roomsim.md §12.4)")
        if lost is not None:
            line += f" — the manifest's counts.lost is {lost}"
        if n_tx is not None:
            line += (f"; {n_tx} transmitters interleave in each receiver's stream with no transmitter id, "
                     "so a replay treats them as one link (spec/roomsim.md §12.4)")
        return line

    @staticmethod
    def computed(path):
        return f"per-node delivery computed from a SIM export (not a measurement): {path}"

    @staticmethod
    def measured(path):
        return f"per-node delivery measured at the gateway: {path}"

    def copy(self, name, manifest_doc=None, data=None):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            fh.write(self.pcap_bytes if data is None else data)
        prefix = path[: -len(".sim.pcap")] if path.endswith(".sim.pcap") else path[: -len(".pcap")]
        if manifest_doc is not None:
            with open(prefix + ".manifest.json", "w") as fh:
                json.dump(manifest_doc, fh)
        return path

    def test_stats_on_an_export(self):
        self.assertGreater(self.lost, 0)
        rc, out = self.cli("stats", self.info["pcap"])
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines()[0], self.banner(self.info["pcap"], self.lost, self.n_tx))
        self.assertEqual(out.splitlines()[1], self.computed(self.info["pcap"]))
        self.assertNotIn("measured at the gateway", out)

    def test_replay_on_an_export(self):
        rc, out = self.cli("replay", self.info["pcap"], "--calibration-s", "0.5")
        self.assertEqual(out.splitlines()[0], self.banner(self.info["pcap"], self.lost, self.n_tx))

    def test_sim_pcap_without_a_manifest(self):
        path = self.copy("lone.sim.pcap")
        rc, out = self.cli("stats", path)
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines()[0], self.banner(path))
        self.assertEqual(out.splitlines()[1], self.computed(path))

    def test_renamed_export_with_a_sim_manifest(self):
        path = self.copy("renamed.pcap", {"label": "SIM", "counts": {"lost": 7, "datagrams": self.datagrams}})
        rc, out = self.cli("stats", path)
        self.assertEqual(out.splitlines()[0], self.banner(path, 7))
        self.assertEqual(out.splitlines()[1], self.computed(path))
        rc, out = self.cli("replay", path, "--calibration-s", "0.5")
        self.assertEqual(out.splitlines()[0], self.banner(path, 7))

    def test_interleaved_transmitters_note(self):
        links = [{"tx": a, "rx": b} for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))]
        doc = {"label": "SIM", "counts": {"lost": 3, "datagrams": self.datagrams}, "links": links}
        path = self.copy("three.pcap", dict(doc, gateway_faithful=False))
        for cmd in (("stats", path), ("replay", path, "--calibration-s", "0.5")):
            rc, out = self.cli(*cmd)
            self.assertEqual(out.splitlines()[0], self.banner(path, 3, 3), cmd[0])
        # one transmitter (gateway-faithful), or no such field: no note
        for name, extra in (("one.pcap", {"gateway_faithful": True, "links": links[:2]}),
                            ("old.pcap", {})):
            path = self.copy(name, dict(doc, **extra))
            rc, out = self.cli("stats", path)
            self.assertEqual(out.splitlines()[0], self.banner(path, 3), name)

    def test_real_capture_beside_a_sim_manifest(self):
        # A real recording that shares an export's prefix: the manifest's
        # datagram count is not this file's record count, so no banner.
        path = os.path.join(self.dir, "beside.pcap")
        with PcapWriter(path) as w:
            for i in range(5):
                w.write(1_700_000_000_000_000_000 + i * 50_000_000, ("10.0.0.21", 40001), ("10.0.0.1", 5566),
                        build_esp32_frame([(1, 2), (3, 4)], node_id=1, sequence=i))
        self.assertNotEqual(self.datagrams, 5)
        with open(os.path.join(self.dir, "beside.manifest.json"), "w") as fh:
            json.dump({"label": "SIM", "counts": {"lost": 9, "datagrams": self.datagrams},
                       "gateway_faithful": False, "links": [{"tx": 0}, {"tx": 1}]}, fh)
        rc, out = self.cli("stats", path)
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines()[0], self.measured(path))
        self.assertNotIn("SIM", out)
        rc, out = self.cli("replay", path, "--calibration-s", "0.1")
        self.assertNotIn("SIM", out)
        # the export's own records beside a SIM manifest that lacks or misstates the count: no banner
        for name, counts in (("nocount.pcap", {"lost": 9}), ("offby1.pcap", {"datagrams": self.datagrams + 1}),
                             ("boolcount.pcap", {"datagrams": True})):
            path = self.copy(name, {"label": "SIM", "counts": counts})
            rc, out = self.cli("stats", path)
            self.assertEqual(out.splitlines()[0], self.measured(path), name)
            self.assertNotIn("SIM", out)

    def test_no_banner_otherwise(self):
        for name, doc in (("plain.pcap", None), ("real.pcap", {"label": "REAL"}), ("junk.pcap", ["SIM"]),
                          ("real2.pcap", {"label": "REAL", "counts": {"datagrams": self.datagrams}})):
            path = self.copy(name, doc)
            rc, out = self.cli("stats", path)
            self.assertEqual(rc, 0)
            self.assertEqual(out.splitlines()[0], self.measured(path))
            self.assertNotIn("SIM", out)


if __name__ == "__main__":
    unittest.main()
