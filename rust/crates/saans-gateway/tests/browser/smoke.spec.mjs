// The manual checklist of docs/COMPLETION_REPORT.md §6, automated against
// the gateway replaying the SYNTHETIC fixture night (setup.mjs).
//
// Spaces: `rest` (vitals consented, a breather at 14 BPM at the end) and
// `hall` (not consented; its node leaks 17.25 BPM, which nobody may see).
// Users: admin, warden (safety, `rest` only), facilities (occupancy).

import { expect, test } from '@playwright/test';
import { request as httpRequest } from 'node:http';

const base = () => process.env.SAANS_URL;
const users = () => JSON.parse(process.env.SAANS_USERS);
const LEAKED = '17.25';

async function signIn(page, name) {
  const r = await page.request.post(base() + '/api/login', {
    data: { user: name, password: users()[name].password },
  });
  expect(r.status(), `sign in as ${name}`).toBe(200);
}

// The one failing request a correct page makes here: with no alert
// routes configured, /api/alerts is 404, which is how the dashboard learns
// to hide the alerts panel.
const EXPECTED_FAILURES = [/^404 \/api\/alerts\?/];

/** Uncaught exceptions, console errors and failed requests, collected for
 *  the whole page. Chrome logs every 4xx as a console error, so those are
 *  judged by their response, against EXPECTED_FAILURES, instead. */
function watchErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) errors.push(m.text());
  });
  page.on('response', (r) => {
    if (r.status() < 400) return;
    const u = new URL(r.url());
    const what = `${r.status()} ${u.pathname}${u.search}`;
    if (!EXPECTED_FAILURES.some((re) => re.test(what))) errors.push(what);
  });
  return errors;
}

const card = (page, name) => page.locator('#ov-grid .card', { hasText: name });

// What each view must show once its data has arrived.
const VIEWS = {
  overview: async (page) => {
    await expect(card(page, 'Rest room')).toContainText('14.0 BPM');
    await expect(card(page, 'Hall')).toContainText('Occupied');
  },
  signals: async (page) => {
    const charts = page.locator('#sig-charts');
    await expect(charts).toContainText('latest: rate 14.00 BPM');
    await expect(charts).toContainText('latest: CSI rate 20.00 Hz');
    await expect(charts.locator('svg').first()).toBeVisible();
  },
  topology: async (page) => {
    await expect(page.locator('#topo')).toContainText(/rest room/i);
    await expect(page.locator('#topo svg').first()).toBeVisible();
  },
  events: async (page) => {
    await expect(page.locator('#ev-count')).toHaveText('4 of 4 shown');
    await expect(page.locator('#ev-list')).toContainText('Someone arrived');
  },
  system: async (page) => {
    await expect(page.locator('#sys-body')).toContainText('datagrams / s');
  },
};

async function openView(page, view) {
  await page.goto(`${base()}/app#${view}`);
  await expect(page.locator(`#view-${view}`)).toBeVisible();
  await VIEWS[view](page);
}

test('all five views render with data, and the leaked rate is nowhere', async ({ page }) => {
  const errors = watchErrors(page);
  await signIn(page, 'admin');
  for (const view of Object.keys(VIEWS)) {
    await openView(page, view);
    expect(await page.locator('body').innerText(), view).not.toContain(LEAKED);
  }
  // The hall is occupied and its node sends breathing, so an estimate
  // would be on screen if consent were not enforced. Even for admin.
  await openView(page, 'overview');
  const hall = card(page, 'Hall');
  await expect(hall).toContainText('node sends vitals without consent');
  await expect(hall).not.toContainText(/breathing|BPM/i);
  await openView(page, 'signals');
  await page.selectOption('#sig-space', 'hall');
  await expect(page.locator('#sig-charts')).toContainText('latest: CSI rate');
  await expect(page.locator('#sig-charts')).not.toContainText(/latest: rate|BPM/);
  expect(errors).toEqual([]);
});

test('dark and light themes both render', async ({ browser }) => {
  const backgrounds = {};
  for (const colorScheme of ['dark', 'light']) {
    const context = await browser.newContext({ colorScheme });
    const page = await context.newPage();
    const errors = watchErrors(page);
    await signIn(page, 'admin');
    await openView(page, 'overview');
    backgrounds[colorScheme] = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    expect(errors, colorScheme).toEqual([]);
    await context.close();
  }
  expect(backgrounds.dark).not.toEqual(backgrounds.light);
});

test('at 375 px no view scrolls sideways', async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 375, height: 812 } });
  const page = await context.newPage();
  await signIn(page, 'admin');
  for (const view of Object.keys(VIEWS)) {
    await openView(page, view);
    const { scroll, client } = await page.evaluate(() => ({
      scroll: document.documentElement.scrollWidth,
      client: document.documentElement.clientWidth,
    }));
    expect(scroll, `${view}: scrollWidth ${scroll} > clientWidth ${client}`).toBeLessThanOrEqual(client);
  }
  await context.close();
});

/** Status of a WebSocket upgrade request: 101 if it was accepted. */
function upgradeStatus(url, headers = {}) {
  return new Promise((resolve, reject) => {
    const req = httpRequest(url, {
      headers: {
        Connection: 'Upgrade',
        Upgrade: 'websocket',
        'Sec-WebSocket-Version': '13',
        'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==',
        ...headers,
      },
    });
    req.on('upgrade', (res, socket) => { socket.destroy(); resolve(res.statusCode); });
    req.on('response', (res) => { res.resume(); resolve(res.statusCode); });
    req.on('error', reject);
    req.end();
  });
}

test('anonymous: the sign-in screen, and every data route and the WebSocket refuse', async ({ page }) => {
  await page.goto(`${base()}/app`);
  await expect(page.locator('#view-login')).toBeVisible();
  await expect(page.locator('#view-overview')).toBeHidden();
  for (const path of ['/api/spaces', '/api/spaces/rest', '/api/events', '/api/history?space=rest',
    '/api/links', '/api/health', '/metrics']) {
    expect((await page.request.get(base() + path)).status(), path).toBe(401);
  }
  expect(await upgradeStatus(`${base()}/api/live`)).toBe(401);

  // And the form works.
  await page.fill('#login-user', 'warden');
  await page.fill('#login-pass', users().warden.password);
  await page.click('#login-go');
  await expect(page.locator('#view-login')).toBeHidden();
  await expect(card(page, 'Rest room')).toContainText('14.0 BPM');
});

test('occupancy role: breathing is "not shown to your role", never "not detected"', async ({ page }) => {
  await signIn(page, 'facilities');
  await page.goto(`${base()}/app#overview`);
  const rest = card(page, 'Rest room');
  await expect(rest).toContainText('Occupied');
  // Withheld and absent are different facts; a facilities user must not
  // be told nobody is breathing in an occupied room.
  await expect(rest).toContainText('not shown to your role');
  await expect(rest).not.toContainText('not confidently detected');
  await expect(rest).not.toContainText('BPM');
  await expect(page.locator('#tab-system')).toHaveCount(0);
  expect((await page.request.get(base() + '/api/health')).status()).toBe(403);
  const state = await (await page.request.get(base() + '/api/spaces/rest')).json();
  expect(state.state.occupied).toBe(true);
  expect('breathing' in state.state).toBe(false);
});

test('safety role: consented breathing in scope; the other space is 404', async ({ page }) => {
  await signIn(page, 'warden');
  await page.goto(`${base()}/app#overview`);
  await expect(card(page, 'Rest room')).toContainText('14.0 BPM (confidence 0.85)');
  await expect(card(page, 'Hall')).toHaveCount(0);
  expect((await page.request.get(base() + '/api/health')).status()).toBe(403);
  expect((await page.request.get(base() + '/api/spaces/hall')).status()).toBe(404);
  expect((await page.request.get(base() + '/api/spaces/rest')).status()).toBe(200);
});

test('a login flood is refused with 429 while /api/me stays fast', async ({ page, playwright }) => {
  const flood = await playwright.request.newContext();
  const attempts = Array.from({ length: 20 }, () =>
    flood.post(base() + '/api/login', { data: { user: 'admin', password: 'wrong' } }));
  // Asked while the attempts are in flight.
  const t0 = performance.now();
  const me = await page.request.get(base() + '/api/me');
  const meMs = performance.now() - t0;
  const statuses = (await Promise.all(attempts)).map((r) => r.status());
  await flood.dispose();

  expect(me.status()).toBe(200);
  expect(statuses.every((s) => s === 401 || s === 429), String(statuses)).toBe(true);
  expect(statuses.filter((s) => s === 429).length, String(statuses)).toBeGreaterThan(0);
  // Chosen before measuring: an /api/me queued behind Argon2 hashing would
  // take a large fraction of a second per attempt ahead of it.
  expect(meMs).toBeLessThan(250);
});
