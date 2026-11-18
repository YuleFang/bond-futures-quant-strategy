"""本地清华 PyPI 镜像转发服务（环境适配工具，非策略代码）。

背景：某些网络环境下 pip 直连 pypi.tuna.tsinghua.edu.cn 被网关返回 403，
而普通 HTTP 客户端（curl / urllib）访问同一 URL 返回 200 且索引页完整。
本服务在 127.0.0.1 起一个只读转发，pip 只连 localhost，由服务端向清华镜像
取 simple 索引页与 packages wheel，数据源仍然是清华镜像。

用法（两 terminal，或后台启动本脚本后执行 pip）：

    python tools/pypi_tuna_proxy.py
    python -m pip install -r requirements.txt \
        -i http://127.0.0.1:8765/simple/ \
        --trusted-host 127.0.0.1

安全边界：只转发 /simple/ 与 /packages/ 两个只读路径，不执行任何远程代码，
不写入工程数据目录；wheel 内容由 pip 按页面 sha256 自行校验。
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import posixpath
import shutil
import urllib.error
import urllib.request

UPSTREAM = "https://pypi.tuna.tsinghua.edu.cn"
HOST = "127.0.0.1"
PORT = 8765
TIMEOUT = 180
CHUNK = 256 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        # 仅在出错时由 send_error 输出，正常流量不刷屏
        pass

    def do_GET(self):
        self._forward(head_only=False)

    def do_HEAD(self):
        self._forward(head_only=True)

    def _forward(self, head_only: bool) -> None:
        raw_path = self.path.split("?", 1)[0].split("#", 1)[0]
        # 索引页中的 ../../packages/... 相对链接需要归一化为 /packages/...
        path = posixpath.normpath(raw_path)
        if not (path.startswith("/simple/") or path.startswith("/packages/")):
            self.send_error(404, "only /simple/ and /packages/ are proxied")
            return

        req = urllib.request.Request(
            UPSTREAM + path,
            headers={"User-Agent": "curl/8.0", "Accept": "*/*"},
        )
        try:
            upstream = urllib.request.urlopen(req, timeout=TIMEOUT)
        except urllib.error.HTTPError as exc:
            self.send_error(exc.code, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 网络错误统一回 502
            self.send_error(502, f"upstream error: {exc}")
            return

        try:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                upstream.headers.get("Content-Type", "application/octet-stream"),
            )
            length = upstream.headers.get("Content-Length")
            if length:
                self.send_header("Content-Length", length)
            self.send_header("Connection", "close")
            self.end_headers()
            if not head_only:
                shutil.copyfileobj(upstream, self.wfile, length=CHUNK)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            upstream.close()
            self.close_connection = True


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"TUNA proxy listening on http://{HOST}:{PORT}/simple/ -> {UPSTREAM}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
