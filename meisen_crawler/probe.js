'use strict';
/**
 * Read-only probe: open (or navigate to) the target page inside the CDP-enabled
 * Quark, inspect DOM + storage key names + network traffic, then print a
 * sanitized structure report. No credentials or response bodies are persisted.
 *
 * Usage: node probe.js [url]
 */
const { CDP, cdpHttp, sanitizeUrl } = require('./cdp_client');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TARGET = process.argv[2] || 'https://mx.2026.naaifu.cn/';

async function main() {
  console.log('PROBE_TARGET=', TARGET);
  const version = await cdpHttp('/json/version');
  console.log('BROWSER=', version.Browser || '?');

  // Create a blank tab first so Network.enable is active BEFORE navigation.
  let tab;
  try {
    tab = await cdpHttp('/json/new?about:blank', 'PUT');
  } catch (e) {
    throw new Error('cannot create CDP tab: ' + e.message);
  }
  if (!tab || !tab.webSocketDebuggerUrl) {
    throw new Error('CDP /json/new returned no websocket url');
  }
  console.log('TAB_ID=', tab.id);

  const c = new CDP(tab.webSocketDebuggerUrl);
  await c.connect();
  await c.send('Page.enable');
  await c.send('Runtime.enable');
  await c.send('Network.enable');

  await c.send('Page.navigate', { url: TARGET });

  for (let i = 0; i < 24; i++) {
    await sleep(500);
    const state = await c.evaluate('document.readyState').catch(() => null);
    if (state === 'complete' && i >= 3) break;
  }
  // SPA / IM clients keep streaming after load; give XHR and WS frames a moment.
  await sleep(3000);

  const page = await c.evaluate(`(() => {
    const body = document.body ? document.body.innerText : '';
    const links = Array.from(document.querySelectorAll('a')).slice(0, 30)
      .map((a) => ((a.innerText || '').trim().slice(0, 40) + ' -> ' + (a.getAttribute('href') || '')).slice(0, 120));
    const buttons = Array.from(document.querySelectorAll('button,[role=button],input[type=button]')).slice(0, 30)
      .map((b) => ((b.innerText || b.getAttribute('aria-label') || b.value || '').trim()).slice(0, 40));
    let lsKeys = [], ssKeys = [];
    try { lsKeys = Object.keys(localStorage); } catch (e) {}
    try { ssKeys = Object.keys(sessionStorage); } catch (e) {}
    const classes = {};
    document.querySelectorAll('[class]').forEach((el) => {
      String(el.className || '').split(/\\s+/).forEach((k) => {
        if (k && k.length < 60) classes[k] = (classes[k] || 0) + 1;
      });
    });
    const topClasses = Object.entries(classes).sort((a, b) => b[1] - a[1]).slice(0, 40);
    const inputs = Array.from(document.querySelectorAll('input')).slice(0, 15)
      .map((i) => ({ type: i.type, name: i.name, placeholder: i.placeholder }));
    return {
      href: location.href,
      title: document.title,
      readyState: document.readyState,
      textLen: body.length,
      text: body.slice(0, 3000),
      links, buttons, inputs, lsKeys, ssKeys, topClasses,
      iframeCount: document.querySelectorAll('iframe').length,
      canvasCount: document.querySelectorAll('canvas').length,
      cookieKeys: document.cookie.split(';').map((x) => x.split('=')[0].trim()).filter(Boolean),
    };
  })()`);

  console.log('---PAGE---');
  console.log(JSON.stringify(page, null, 2));

  const reqs = c.events
    .filter((e) => e.method === 'Network.requestWillBeSent')
    .map((e) => ({
      id: e.params.requestId,
      type: e.params.type,
      method: e.params.request.method,
      url: sanitizeUrl(e.params.request.url),
    }))
    .filter((x) => ['XHR', 'Fetch', 'Document', 'WebSocket', 'Script'].includes(x.type));
  console.log('---REQUESTS (sanitized, first 70)---');
  console.log(JSON.stringify(reqs.slice(0, 70), null, 2));

  const xhrIds = reqs.filter((r) => r.type === 'XHR' || r.type === 'Fetch').slice(0, 12);
  console.log('---XHR RESPONSE PREVIEWS---');
  for (const r of xhrIds) {
    try {
      const b = await c.send('Network.getResponseBody', { requestId: r.id });
      const text = b.base64Encoded
        ? Buffer.from(b.body, 'base64').toString('utf8')
        : b.body;
      console.log('BODY|', r.url, '|', String(text).slice(0, 700).replace(/\s+/g, ' '));
    } catch (e) {
      console.log('BODY_FAIL|', r.url, '|', e.message.slice(0, 80));
    }
  }

  const wsCreated = c.events.filter((e) => e.method === 'Network.webSocketCreated')
    .map((e) => sanitizeUrl(e.params.url));
  const wsFrames = c.events.filter((e) => e.method === 'Network.webSocketFrameReceived')
    .slice(0, 12)
    .map((e) => String((e.params.response.payloadData || '')).slice(0, 300));
  console.log('---WEBSOCKET---');
  console.log(JSON.stringify({ created: wsCreated.slice(0, 8), frames: wsFrames }, null, 2));

  console.log('EVENT_COUNT=', c.events.length);
  c.close();
  console.log('PROBE_OK');
}

main().catch((e) => {
  console.error('PROBE_FAIL', e.message);
  process.exit(1);
});
