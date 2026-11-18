'use strict';
/** Minimal CDP connectivity diagnostic. */
const { cdpHttp, CDP } = require('./cdp_client');

(async () => {
  let version;
  try {
    version = await cdpHttp('/json/version');
    console.log('VERSION_OK=', version.Browser);
  } catch (e) {
    console.log('VERSION_FAIL=', e.message);
    process.exit(2);
  }
  const list = await cdpHttp('/json/list');
  console.log('TARGETS=', JSON.stringify(list.map((t) => ({ type: t.type, url: (t.url || '').slice(0, 90), id: t.id })).slice(0, 12)));
  const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  if (!page) { console.log('NO_PAGE_TARGET'); process.exit(3); }
  console.log('PICK=', page.id, (page.url || '').slice(0, 90));

  // Raw websocket round-trip first.
  await new Promise((resolve) => {
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    const timer = setTimeout(() => { console.log('RAW_TIMEOUT'); ws.close(); resolve(); }, 5000);
    ws.addEventListener('open', () => {
      console.log('RAW_OPEN');
      ws.send(JSON.stringify({ id: 1, method: 'Runtime.evaluate', params: { expression: '1+1', returnByValue: true } }));
    });
    ws.addEventListener('message', (ev) => {
      const d = ev.data;
      const raw = typeof d === 'string' ? d : new TextDecoder().decode(d);
      console.log('RAW_MSG_TYPE=', typeof d, 'RAW_MSG=', String(raw).slice(0, 300));
      clearTimeout(timer); ws.close(); resolve();
    });
    ws.addEventListener('error', () => { console.log('RAW_ERROR'); clearTimeout(timer); resolve(); });
    ws.addEventListener('close', (e) => console.log('RAW_CLOSE=', e.code, String(e.reason).slice(0, 80)));
  });

  // Then test the wrapper class.
  try {
    const c = new CDP(page.webSocketDebuggerUrl);
    await c.connect();
    await c.send('Runtime.enable');
    const v = await c.evaluate('2+3');
    console.log('WRAPPER_OK=', v);
    const href = await c.evaluate('location.href');
    console.log('HREF=', href);
    c.close();
  } catch (e) {
    console.log('WRAPPER_FAIL=', e.message);
  }
})().catch((e) => { console.error('DIAG_FAIL', e.message); process.exit(1); });
