"""One install folder, one owner.

A tester's 0.0.1.6 install died for good on ``.venv\\Scripts: Access is denied``
and never recovered, across half an hour and three game restarts. The log shows
why: the installer was started a SECOND time while the first bootstrap was still
downloading torch. Nothing stopped the second bootstrap, so it ran ``uv venv
--clear`` on a venv the first was still writing into (``Lib: The directory is not
empty`` — os error 145, files appearing while it deleted), and from then on the
leftovers of the first run held ``.venv\\Scripts`` and every later attempt failed.

Two pieces close that:

* :class:`InstanceLock` — an OS-level lock in ``.state``. The OS drops it when
  the holder dies, however it dies, so a stale lock cannot exist. A new bootstrap
  **takes over**: the one already running is either orphaned (its host is gone,
  nobody reads its output) or about to be replaced on purpose (a reinstall), and
  in both cases the newest request is the one somebody is waiting for.
* :func:`stale_processes` — once we own the folder, every process running an
  executable from inside it is by definition a leftover (an orphaned server, a
  half-finished ``uv`` of a killed bootstrap) and may be stopped. Only then can
  Windows delete the venv. The selection is a pure function so it is tested
  without touching real processes.

Standard library only: the bootstrap imports this before any venv exists.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

IS_WINDOWS = sys.platform.startswith("win")

LOCK_NAME = "bootstrap.lock"
PID_NAME = "bootstrap.pid"


class InstanceLock:
    """Exclusive, non-blocking lock on ``<directory>/bootstrap.lock``.

    The holder's pid lives in a SEPARATE file: a Windows byte-range lock makes the
    locked bytes unreadable to everybody else, so a pid written into the lock file
    itself could never be read by the process that needs it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.lock_path = self.directory / LOCK_NAME
        self.pid_path = self.directory / PID_NAME
        self._fd: int | None = None

    def try_acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.directory.mkdir(parents=True, exist_ok=True)
        # os.open is non-inheritable by default (PEP 446): the server child must
        # NOT carry the lock on after the bootstrap is gone.
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock_fd(fd)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        return True

    def holder_pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            _unlock_fd(self._fd)
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None


if IS_WINDOWS:
    import msvcrt

    def _lock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def take_over(
    lock: InstanceLock,
    terminate: Callable[[int], bool],
    timeout: float = 20.0,
    poll: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int | None:
    """Acquire ``lock``, stopping whoever holds it. Returns the pid that was
    stopped (``None`` if the lock was free). Raises ``TimeoutError`` if the
    lock is still held after ``timeout`` seconds."""
    if lock.try_acquire():
        return None
    holder = lock.holder_pid()
    if holder is not None and holder != os.getpid():
        terminate(holder)
    deadline = clock() + timeout
    while clock() < deadline:
        sleep(poll)
        if lock.try_acquire():
            return holder
    raise TimeoutError(holder)


# ------------------------------------------------------------ stale processes
@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    exe: str | None


def _is_under(path: str, root: str) -> bool:
    p = os.path.normcase(os.path.normpath(path))
    r = os.path.normcase(os.path.normpath(root))
    return p.startswith(r.rstrip("\\/") + os.sep)


def stale_processes(
    processes: Iterable[ProcessInfo], roots: Iterable[Path], own_pid: int
) -> list[int]:
    """Pids to stop: every process whose executable lies under one of ``roots``
    (the runtime venv and our ``uv`` — deliberately NOT the whole wrapper folder,
    a developer's ``.venv-test`` lives there too), plus all their descendants —
    minus ourselves and our ancestors.

    Descendants matter because a Windows venv ``python.exe`` is only a launcher:
    the real interpreter is its CHILD, runs from the base Python elsewhere, and
    is the one holding the venv's ``.pyd`` files. Ancestors matter because the
    host starts us through ``.uv/uv.exe run`` — our own parent is under a root
    and stopping it would kill us.
    """
    procs = {p.pid: p for p in processes}
    protected: set[int] = set()
    pid = own_pid
    while pid in procs and pid not in protected:
        protected.add(pid)
        pid = procs[pid].ppid
    protected.add(own_pid)

    children: dict[int, list[int]] = {}
    for p in procs.values():
        if p.ppid != p.pid:
            children.setdefault(p.ppid, []).append(p.pid)

    selected: set[int] = set()
    root_strs = [str(r) for r in roots]
    stack = [
        p.pid for p in procs.values()
        if p.exe and any(_is_under(p.exe, r) for r in root_strs)
    ]
    while stack:
        pid = stack.pop()
        if pid in selected or pid in protected:
            continue
        selected.add(pid)
        stack.extend(children.get(pid, ()))
    return sorted(selected)


def list_processes() -> list[ProcessInfo]:
    """All processes with parent pid and image path (Windows only — on POSIX an
    open file does not stop a directory from being deleted, so there is nothing
    to sweep and this returns an empty list)."""
    if not IS_WINDOWS:
        return []
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declare HANDLE returns on x64, or ctypes truncates them to 32-bit ints
    # (the same trap as the kill-on-close job in bootstrap.py).
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]

    TH32CS_SNAPPROCESS = 0x2
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    INVALID = wintypes.HANDLE(-1).value

    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID:
        return []
    result: list[ProcessInfo] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            exe = None
            handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                try:
                    buf = ctypes.create_unicode_buffer(32768)
                    size = wintypes.DWORD(len(buf))
                    if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                        exe = buf.value
                finally:
                    k32.CloseHandle(handle)
            result.append(ProcessInfo(pid, int(entry.th32ParentProcessID), exe))
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return result


def terminate(pid: int) -> bool:
    """Stop ``pid`` hard. On Windows this also takes the bootstrap's whole tree
    down with it: its kill-on-close job closes with its last handle."""
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        PROCESS_TERMINATE = 0x1
        handle = k32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if not handle:
            return False
        try:
            return bool(k32.TerminateProcess(handle, 1))
        finally:
            k32.CloseHandle(handle)
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False
