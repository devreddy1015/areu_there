// Global setup: build the gateway, write the SYNTHETIC fixture night,
// configure three users with fresh random passwords, and replay the night
// with the dashboard served. Tests read the address and the credentials
// from SAANS_URL and SAANS_USERS; nothing is written outside a temp dir.

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const RUST = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');

// Who signs in. `warden` is limited to `rest`, so `hall` is out of scope.
const USERS = [
  { name: 'admin', role: 'admin', spaces: [] },
  { name: 'warden', role: 'safety', spaces: ['rest'] },
  { name: 'facilities', role: 'occupancy', spaces: [] },
];

function freePort() {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

async function waitFor(what, ms, check) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (await check()) return;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`timed out waiting for ${what}`);
}

export default async function setup() {
  execFileSync(
    'cargo',
    ['build', '-q', '-p', 'saans-gateway', '--bin', 'saans-gateway', '--example', 'fixture_night'],
    { cwd: RUST, stdio: 'inherit' },
  );
  const target = JSON.parse(
    execFileSync('cargo', ['metadata', '--format-version', '1', '--no-deps'], { cwd: RUST }),
  ).target_directory;
  const gateway = join(target, 'debug', 'saans-gateway');

  const dir = mkdtempSync(join(tmpdir(), 'saans-browser-'));
  const pcap = join(dir, 'night.pcap');
  execFileSync(join(target, 'debug', 'examples', 'fixture_night'), [pcap], { stdio: 'inherit' });

  const users = {};
  let usersToml = '';
  for (const u of USERS) {
    const password = randomBytes(18).toString('base64url');
    const r = spawnSync(gateway, ['--hash-password'], { input: password + '\n' });
    if (r.status !== 0) throw new Error(`--hash-password failed: ${r.stderr}`);
    const hash = r.stdout.toString().trim();
    users[u.name] = { password, role: u.role };
    usersToml += `[[user]]\nname = "${u.name}"\nrole = "${u.role}"\npassword_hash = "${hash}"\n`
      + `spaces = ${JSON.stringify(u.spaces)}\n\n`;
  }

  const port = await freePort();
  const config = join(dir, 'saans.toml');
  writeFileSync(config, `http_addr = "127.0.0.1:${port}"
data_dir = '${dir}'

[auth]
required = true

[[space]]
id = "rest"
name = "Rest room"
vitals = true
links = [{ rx = 3, tx = 4 }, { rx = 5, tx = 4 }]

[[space]]
id = "hall"
name = "Hall"
vitals = false
links = [{ rx = 1, tx = 255 }]

${usersToml}`);

  const proc = spawn(gateway, ['-c', config, '--replay', pcap, '--speed', '0', '--serve'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let log = '';
  proc.stdout.on('data', (d) => { log += d; });
  proc.stderr.on('data', (d) => { log += d; });
  let exited = null;
  proc.on('exit', (code) => { exited = code; });
  try {
    await waitFor('the replay to finish', 60_000, () => {
      if (exited !== null) throw new Error(`gateway exited (${exited}):\n${log}`);
      return log.includes('replay finished');
    });
  } catch (e) {
    proc.kill('SIGKILL');
    throw e;
  }

  process.env.SAANS_URL = `http://127.0.0.1:${port}`;
  process.env.SAANS_USERS = JSON.stringify(users);

  return async () => {
    if (exited === null) {
      proc.kill('SIGINT');
      await new Promise((r) => proc.once('exit', r));
    }
    rmSync(dir, { recursive: true, force: true });
  };
}
