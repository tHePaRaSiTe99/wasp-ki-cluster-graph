"""
Hintergrunddienst für den Autostart: hält die Daten aktuell und zeigt den Graphen unter http://localhost:8000.

  python src/updater.py                       # Dienst starten (läuft endlos)
  python src/updater.py --once                # nur einen Lauf, dann beenden
  python src/updater.py --install-autostart   # Windows: startet den Dienst bei jeder Anmeldung (unsichtbar)
  python src/updater.py --uninstall-autostart

Verhalten:
  * Beim Start und danach alle 30 Min. wird geprüft: Liegt der letzte erfolgreiche Lauf >= 24 h zurück,
    läuft die Pipeline. War der Rechner aus, wird also beim nächsten Start nachgeholt.
  * Bei einem Fehler wird frühestens nach 1 h erneut versucht; die alte graph.json bleibt bestehen.
  * Es gibt nur eine Instanz (zweiter Start beendet sich sofort).
  * Protokoll: data/updater.log
"""
import argparse
import functools
import logging
import os
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path

import state as state_store

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
LOG_PATH = ROOT / "data" / "updater.log"
LOCK_PORT = 47831  # nur zur Erkennung "läuft schon"
log = logging.getLogger("updater")


class _LogWriter:
    """Leitet print()-Ausgaben ins Logfile um (unter pythonw gibt es sonst kein Terminal)."""
    def write(self, s):
        for line in str(s).splitlines():
            if line.strip():
                log.info(line.rstrip())
    def flush(self):
        pass
    def isatty(self):
        return False


def _setup_logging(to_file: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = RotatingFileHandler(LOG_PATH, maxBytes=500_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    if not to_file:
        console = logging.StreamHandler(sys.__stdout__)
        console.setFormatter(fmt)
        log.addHandler(console)
    else:
        sys.stdout = sys.stderr = _LogWriter()


def _serve(port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):  # keine Zeile pro Seitenaufruf im Log
            pass

    handler = functools.partial(Handler, directory=str(WEB_DIR))
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        log.info(f"Webserver auf Port {port} nicht gestartet: {e}")
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info(f"Graph erreichbar unter http://localhost:{port}")


def _due(interval_h: float, retry_h: float) -> bool:
    st = state_store.load()
    now = time.time()
    if now - st.get("last_success", 0) < interval_h * 3600:
        return False
    return now - st.get("last_attempt", 0) >= retry_h * 3600 or st.get("last_attempt", 0) <= st.get("last_success", 0)


def _run_once(force: bool = False) -> None:
    from pipeline import run  # erst hier importieren: der Import lädt ggf. das Modell-Framework
    try:
        log.info("Aktualisierung startet ...")
        run(force=force)
    except Exception as e:  # Netz weg, Seite geändert, ... -> alte Daten bleiben, später erneut versuchen
        log.exception(f"Aktualisierung fehlgeschlagen: {e}")


# ---------------------------------------------------------------------------
# Autostart (Windows)
# ---------------------------------------------------------------------------
def _startup_file() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise SystemExit("Autostart-Einrichtung gibt es nur für Windows. Unter Linux/macOS: Cron-Eintrag  "
                         f"@reboot cd {ROOT} && {sys.executable} src/updater.py")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "thk-ai-graph.vbs"


def install_autostart() -> None:
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    runner = pyw if pyw.exists() else exe
    script = ROOT / "src" / "updater.py"
    # VBS statt BAT: startet ohne aufblitzendes Konsolenfenster
    vbs = (f'Set sh = CreateObject("WScript.Shell")\r\n'
           f'sh.CurrentDirectory = "{ROOT}"\r\n'
           f'sh.Run """{runner}"" ""{script}""", 0, False\r\n')
    target = _startup_file()
    target.write_text(vbs, encoding="utf-8")
    print(f"Autostart eingerichtet: {target}\nDer Dienst startet ab der nächsten Anmeldung. Log: {LOG_PATH}")


def uninstall_autostart() -> None:
    target = _startup_file()
    if target.exists():
        target.unlink()
        print("Autostart entfernt. (Ein gerade laufender Dienst muss im Task-Manager beendet werden: pythonw.exe)")
    else:
        print("Kein Autostart-Eintrag vorhanden.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="einen Lauf ausführen und beenden")
    ap.add_argument("--force", action="store_true", help="mit --once: alles neu laden und extrahieren")
    ap.add_argument("--port", type=int, default=8000, help="Port für den Webserver, 0 = aus")
    ap.add_argument("--interval-hours", type=float, default=24)
    ap.add_argument("--retry-hours", type=float, default=1)
    ap.add_argument("--check-minutes", type=float, default=30)
    ap.add_argument("--install-autostart", action="store_true")
    ap.add_argument("--uninstall-autostart", action="store_true")
    a = ap.parse_args()

    if a.install_autostart:
        return install_autostart()
    if a.uninstall_autostart:
        return uninstall_autostart()

    daemon = not a.once and not sys.stdout.isatty()
    _setup_logging(to_file=daemon or sys.stdout is None)
    if a.once:
        return _run_once(a.force)

    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        log.info("Dienst läuft bereits - beende diese zweite Instanz.")
        return
    if a.port:
        _serve(a.port)
    log.info(f"Dienst gestartet. Aktualisierung alle {a.interval_hours:g} h.")
    while True:
        if _due(a.interval_hours, a.retry_hours):
            _run_once()
        time.sleep(a.check_minutes * 60)


if __name__ == "__main__":
    main()
