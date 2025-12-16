"""
터미널용 예쁜 로그 유틸리티 (외부 의존성 없이 ANSI 컬러 사용)

환경 변수
- PRETTY_LOG: 1(기본) / 0 -> 예쁜 출력 비활성화
- LOG_LEVEL: DEBUG / INFO(기본) / WARN / ERROR
"""
import os
from typing import Optional, Dict


def _supports_ansi() -> bool:
    if os.getenv("PRETTY_LOG", "1") == "0":
        return False
    return True


ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2m",
    "bold": "\x1b[1m",
    "blue": "\x1b[34m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "magenta": "\x1b[35m",
    "gray": "\x1b[90m",
}


ICONS = {
    "info": "ℹ️",
    "success": "✅",
    "warn": "⚠️",
    "error": "❌",
    "step": "▶️",
    "image": "🖼️",
    "search": "🔎",
    "llm": "🤖",
    "rocket": "🚀",
    "bug": "🪲",
}


_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
_LOG_LEVEL = _LEVEL_ORDER.get(os.getenv("LOG_LEVEL", "INFO").upper(), 20)
_ANSI_ON = _supports_ansi()


def _c(text: str, color: str) -> str:
    if not _ANSI_ON:
        return text
    return f"{ANSI.get(color, '')}{text}{ANSI['reset']}"


def divider(width: int = 64, char: str = "─") -> None:
    line = char * width
    if _ANSI_ON:
        print(_c(line, "gray"))
    else:
        print(line)


def section(title: str, icon: str = "rocket") -> None:
    divider()
    head = f"{ICONS.get(icon, '')} {title}"
    if _ANSI_ON:
        print(_c(head, "bold"))
    else:
        print(head)
    divider()


def _should_log(level: str) -> bool:
    return _LEVEL_ORDER[level] >= _LOG_LEVEL


def _log(level: str, msg: str, *, icon: str, color: str, kv: Optional[Dict[str, str]] = None) -> None:
    if not _should_log(level):
        return
    prefix = f"{ICONS.get(icon, '')}"
    body = msg.strip()
    if kv:
        # key=value 형식으로 한 줄 요약
        pairs = [f"{k}={v}" for k, v in kv.items()]
        body = f"{body}  " + _c(" ", "dim").join([_c(p, "dim") for p in pairs])
    if _ANSI_ON:
        print(f"{_c(prefix, color)} {_c(body, color)}")
    else:
        print(f"{prefix} {body}")


def debug(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("DEBUG", msg, icon="bug", color="gray", kv=kv)


def info(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("INFO", msg, icon="info", color="cyan", kv=kv)


def success(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("INFO", msg, icon="success", color="green", kv=kv)


def warn(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("WARN", msg, icon="warn", color="yellow", kv=kv)


def error(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("ERROR", msg, icon="error", color="red", kv=kv)


def step(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("INFO", msg, icon="step", color="blue", kv=kv)


def search(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("INFO", msg, icon="search", color="blue", kv=kv)


def llm(msg: str, kv: Optional[Dict[str, str]] = None) -> None:
    _log("INFO", msg, icon="llm", color="cyan", kv=kv)

