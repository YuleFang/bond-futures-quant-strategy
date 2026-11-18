'use strict';
/**
 * Minimal zero-dependency CDP client for the local Quark debug port.
 * Node 22 provides a global WebSocket implementation.
 * Nothing here is sent anywhere except 127.0.0.1:9222.
 */
const http = require('http');

const PORT = Number(process.env.CDP_PORT || 9222);

function cdpHttp(path, method = 'GET') {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port: PORT, path, method, timeout: 5000 },
      (res) => {
        let buf = '';
        res.on('data', (d) => (buf += d));
        res.on('end', () => {
          try { resolve(JSON.parse(buf || '{}')); }
          catch { resolve({ _raw: buf.slice(0, 500) }); }
        });
      }
    );
    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('cdp http timeout')));
    req.end();
  });
}

class CDP {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
  }

  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      this.ws = ws;
      const timer = setTimeout(() => reject(new Error('ws connect timeout')), 8000);
      ws.addEventListener('open', () => { clearTimeout(timer); resolve(); });
      ws.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error('ws connect failed (check --remote-allow-origins)'));
      });
      ws.addEventListener('close', () => {
        clearTimeout(timer);
        for (const p of this.pending.values()) p.reject(new Error('ws closed'));
        this.pending.clear();
      });
      ws.addEventListener('message', (ev) => {
        let msg;
        try {
          const d = ev.data;
          const raw = typeof d === 'string'
            ? d
            : (d instanceof ArrayBuffer || ArrayBuffer.isView(d)
                ? new TextDecoder().decode(d)
                : String(d));
          msg = JSON.parse(raw);
        } catch { return; }
        if (msg.id && this.pending.has(msg.id)) {
          const p = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          if (msg.error) p.reject(new Error(JSON.stringify(msg.error)));
          else p.resolve(msg.result);
        } else if (msg.method) {
          this.events.push({ method: msg.method, params: msg.params, ts: Date.now() });
        }
      });
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('cdp timeout: ' + method));
      }, 15000);
      this.pending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const r = await this.send('Runtime.evaluate', {
      expression, returnByValue: true, awaitPromise: true,
    });
    if (r.exceptionDetails) {
      throw new Error('eval: ' + (r.exceptionDetails.text || '') + ' '
        + ((r.exceptionDetails.exception || {}).description || ''));
    }
    return r.result && r.result.value;
  }

  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

async function getTargets() { return cdpHttp('/json/list'); }

async function openTab(url) {
  // Chromium 111+ requires PUT for /json/new
  try {
    const t = await cdpHttp('/json/new?' + url, 'PUT');
    if (t && t.webSocketDebuggerUrl) return t;
  } catch { /* fall through */ }
  const list = await getTargets();
  const page = list.find((t) => t.type === 'page');
  if (!page) throw new Error('no page target available');
  const c = new CDP(page.webSocketDebuggerUrl);
  await c.connect();
  await c.send('Page.enable');
  await c.send('Page.navigate', { url });
  c.close();
  return page;
}

function sanitizeUrl(u) {
  try {
    const x = new URL(u);
    let q = '';
    if (x.search) {
      q = '?' + [...new URLSearchParams(x.search).keys()].map((k) => k + '=*').join('&');
    }
    return x.origin + x.pathname + q;
  } catch { return String(u).slice(0, 180); }
}

module.exports = { cdpHttp, CDP, getTargets, openTab, sanitizeUrl, PORT };
