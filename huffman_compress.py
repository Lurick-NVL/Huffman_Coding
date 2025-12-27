
# Tách từ huffman_core_stable.py: phần nén (HF2 / HFZ).

import heapq
import mmap
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# Stable, deterministic Huffman codec optimized for large UTF-8 text.
# - Streaming encode/decode (low RAM)
# - Varint frequencies (smaller header)
# - Deterministic tree tie-breaking (decode always matches encode)

MAGIC = b"HF2"
VERSION = 1

# LZ77 + Huffman-bytes container
MAGIC_LZ = b"HFZ"  # Huffman + LZ77 token stream
VERSION_LZ = 1

# LZ77 token format:
#   literal: 0x00 <byte>
#   match:   0x01 <varint distance> <varint length>
LZ_T_LITERAL = 0
LZ_T_MATCH = 1

LZ_WINDOW = 32768
LZ_MIN_MATCH = 4
LZ_MAX_MATCH = 258
LZ_HASH_LEN = 3
LZ_MAX_CANDIDATES = 64

# I/O tuning
READ_CHUNK_SIZE = 262_144
WRITE_BYTES_FLUSH = 1_048_576

# Type alias
ProgressCallback = Optional[Callable[[int, int], None]]


# Mã hóa số nguyên không âm theo varint (7-bit/byte).
def _encode_varint(n: int) -> bytes:
    if n < 0:
        raise ValueError("Varint only supports non-negative integers")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(0x80 | b)
        else:
            out.append(b)
            break
    return bytes(out)


# Encode unsigned 16-bit big-endian.
def _u16be(n: int) -> bytes:
    return n.to_bytes(2, "big")


@dataclass
class _Node:
    freq: int
    min_sym: str
    sym: Optional[str] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


@dataclass
class _BNode:
    freq: int
    min_sym: int
    sym: Optional[int] = None
    left: Optional["_BNode"] = None
    right: Optional["_BNode"] = None


# Xây cây Huffman (deterministic) cho bảng tần suất ký tự.
def _build_tree(freq_table: Dict[str, int]) -> Optional[_Node]:
    if not freq_table:
        return None

    serial = 0
    heap = []
    for sym in sorted(freq_table.keys()):
        serial += 1
        node = _Node(freq_table[sym], sym, sym=sym)
        heap.append((node.freq, node.min_sym, serial, node))
    heapq.heapify(heap)

    if len(heap) == 1:
        _, _, _, only = heap[0]
        parent = _Node(only.freq, only.min_sym, sym=None, left=only, right=None)
        return parent

    while len(heap) > 1:
        f1, m1, _, n1 = heapq.heappop(heap)
        f2, m2, _, n2 = heapq.heappop(heap)
        serial += 1
        merged = _Node(f1 + f2, m1 if m1 < m2 else m2, sym=None, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, merged.min_sym, serial, merged))

    return heap[0][3]


# Sinh mã Huffman dạng '0'/'1' từ cây.
def _build_codes(root: Optional[_Node]) -> Dict[str, str]:
    codes: Dict[str, str] = {}

    def dfs(node: _Node, prefix: str) -> None:
        if node.sym is not None:
            codes[node.sym] = prefix or "0"
            return
        if node.left is not None:
            dfs(node.left, prefix + "0")
        if node.right is not None:
            dfs(node.right, prefix + "1")

    if root is not None:
        dfs(root, "")
    return codes


# Xây cây Huffman (deterministic) cho bảng tần suất byte (0..255).
def _build_btree(freq_table: Dict[int, int]) -> Optional[_BNode]:
    if not freq_table:
        return None

    serial = 0
    heap = []
    for sym in sorted(freq_table.keys()):
        serial += 1
        node = _BNode(freq_table[sym], sym, sym=sym)
        heap.append((node.freq, node.min_sym, serial, node))
    heapq.heapify(heap)

    if len(heap) == 1:
        _, _, _, only = heap[0]
        return _BNode(only.freq, only.min_sym, sym=None, left=only, right=None)

    while len(heap) > 1:
        f1, m1, _, n1 = heapq.heappop(heap)
        f2, m2, _, n2 = heapq.heappop(heap)
        serial += 1
        merged = _BNode(f1 + f2, m1 if m1 < m2 else m2, sym=None, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, merged.min_sym, serial, merged))

    return heap[0][3]


# Sinh mã Huffman dạng '0'/'1' cho byte symbols.
def _build_bcodes(root: Optional[_BNode]) -> Dict[int, str]:
    codes: Dict[int, str] = {}

    def dfs(node: _BNode, prefix: str) -> None:
        if node.sym is not None:
            codes[node.sym] = prefix or "0"
            return
        if node.left is not None:
            dfs(node.left, prefix + "0")
        if node.right is not None:
            dfs(node.right, prefix + "1")

    if root is not None:
        dfs(root, "")
    return codes


# Nén Huffman thuần (HF2) cho file văn bản UTF-8.
def compress_to_file(
    input_path: str,
    output_path: str,
    progress_callback: ProgressCallback = None,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    counter = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(READ_CHUNK_SIZE), ""):
            if not chunk:
                break
            counter.update(chunk)
    freq_table = dict(counter)
    total_chars = sum(freq_table.values())

    root = _build_tree(freq_table)
    codes = _build_codes(root)
    code_ints: Dict[str, Tuple[int, int]] = {
        ch: (int(code, 2), len(code)) for ch, code in codes.items()
    }

    bit_buf = 0
    bit_len = 0
    total_bits = 0

    def flush_full_bytes(out_bytes: bytearray) -> None:
        nonlocal bit_buf, bit_len
        while bit_len >= 8:
            bit_len -= 8
            out_bytes.append((bit_buf >> bit_len) & 0xFF)
            bit_buf &= (1 << bit_len) - 1

    out_data = bytearray()
    processed = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(READ_CHUNK_SIZE), ""):
            if not chunk:
                break
            for ch in chunk:
                code_int, clen = code_ints[ch]
                bit_buf = (bit_buf << clen) | code_int
                bit_len += clen
                total_bits += clen
                flush_full_bytes(out_data)
            processed += len(chunk)
            if progress_callback and total_chars:
                progress_callback(processed, total_chars)

    padding = (8 - total_bits % 8) % 8
    if padding:
        bit_buf <<= padding
        bit_len += padding
    flush_full_bytes(out_data)

    with open(output_path, "wb") as out:
        out.write(MAGIC)
        out.write(VERSION.to_bytes(1, "big"))
        out.write(len(freq_table).to_bytes(4, "big"))
        for ch in sorted(freq_table.keys()):
            chb = ch.encode("utf-8")
            if len(chb) > 255:
                raise ValueError("UTF-8 symbol too long")
            out.write(len(chb).to_bytes(1, "big"))
            out.write(chb)
            out.write(_encode_varint(freq_table[ch]))
        out.write(padding.to_bytes(1, "big"))
        out.write(out_data)

    if progress_callback and total_chars:
        progress_callback(total_chars, total_chars)

    return freq_table, codes


# Mã hóa LZ77 cho bytes trong file -> ghi ra stream token.
# Token format:
#   literal: 0x00 <byte>
#   match:   0x01 <varint distance> <varint length>
# Returns number of token bytes written.
def _lz77_encode_to_stream(
    src_path: str,
    token_out,
    freq_counter,
    stats: Optional[Dict[str, int]] = None,
    progress_callback: ProgressCallback = None,
    *,
    max_candidates: int = LZ_MAX_CANDIDATES,
    max_match: int = LZ_MAX_MATCH,
    window_size: int = LZ_WINDOW,
    insert_step: int = 1,
) -> int:
    file_size = os.path.getsize(src_path)
    if file_size == 0:
        if progress_callback:
            progress_callback(0, 0)
        return 0

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if insert_step <= 0:
        raise ValueError("insert_step must be positive")

    with open(src_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            data = mm
            n = len(data)
            i = 0
            token_bytes_written = 0

            pos_lists: Dict[int, list] = {}
            pos_heads: Dict[int, int] = {}

            def h3(pos: int) -> int:
                if pos + LZ_HASH_LEN > n:
                    return -1
                return (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2]

            _pos_lists_get = pos_lists.get
            _pos_heads_get = pos_heads.get
            _pos_heads_set = pos_heads.__setitem__
            _pos_lists_setdefault = pos_lists.setdefault
            _encode_varint_local = _encode_varint
            _freq_arr = freq_counter if isinstance(freq_counter, list) else None
            _freq_update = None if _freq_arr is not None else freq_counter.update
            _token_out_write = token_out.write

            last_progress = -1
            while i < n:
                if progress_callback and file_size:
                    cur_mb = i >> 20
                    if cur_mb != last_progress:
                        last_progress = cur_mb
                        progress_callback(i, n)

                best_len = 0
                best_dist = 0

                if i + LZ_MIN_MATCH <= n:
                    key = h3(i)
                    if key != -1:
                        lst = _pos_lists_get(key)
                        if lst:
                            head = _pos_heads_get(key, 0)
                            window_start = i - window_size
                            while head < len(lst) and lst[head] < window_start:
                                head += 1
                            if head > 4096 and head > (len(lst) >> 1):
                                del lst[:head]
                                head = 0
                            _pos_heads_set(key, head)

                            checked = 0
                            idx = len(lst) - 1
                            while idx >= head and checked < max_candidates:
                                pos = lst[idx]
                                idx -= 1
                                checked += 1
                                dist = i - pos
                                if dist <= 0 or dist > window_size:
                                    continue
                                max_len = min(max_match, n - i)
                                if max_len <= best_len:
                                    continue

                                l = LZ_HASH_LEN
                                while l < max_len and data[pos + l] == data[i + l]:
                                    l += 1
                                if l >= LZ_MIN_MATCH and l > best_len:
                                    best_len = l
                                    best_dist = dist
                                    if best_len == max_match:
                                        break

                if best_len >= LZ_MIN_MATCH:
                    token = bytearray()
                    token.append(LZ_T_MATCH)
                    token.extend(_encode_varint_local(best_dist))
                    token.extend(_encode_varint_local(best_len))
                    _token_out_write(token)
                    if _freq_arr is not None:
                        for _bb in token:
                            _freq_arr[_bb] += 1
                    else:
                        _freq_update(token)
                    token_bytes_written += len(token)

                    if stats is not None:
                        stats["matches"] = stats.get("matches", 0) + 1
                        stats["match_bytes"] = stats.get("match_bytes", 0) + best_len

                    end = i + best_len
                    for p in range(i, end, insert_step):
                        if p + LZ_HASH_LEN <= n:
                            k = h3(p)
                            if k != -1:
                                _pos_lists_setdefault(k, []).append(p)
                    i = end
                else:
                    b = data[i]
                    token = bytes((LZ_T_LITERAL, b))
                    _token_out_write(token)
                    if _freq_arr is not None:
                        _freq_arr[LZ_T_LITERAL] += 1
                        _freq_arr[b] += 1
                    else:
                        _freq_update(token)
                    token_bytes_written += 2

                    if stats is not None:
                        stats["literals"] = stats.get("literals", 0) + 1

                    if i + LZ_HASH_LEN <= n:
                        k = h3(i)
                        if k != -1:
                            _pos_lists_setdefault(k, []).append(i)
                    i += 1

            if progress_callback and file_size:
                progress_callback(n, n)

            return token_bytes_written
        finally:
            mm.close()


# Ghi bitstream Huffman ra file (buffer nội bộ bằng int).
class _BitWriter:

    # Khởi tạo writer cho bitstream Huffman, ghi ra file-like object.
    def __init__(self, f) -> None:
        self.f = f
        self.bit_buf = 0
        self.bit_len = 0
        self.total_bits = 0
        self._out_buf = bytearray()

    # Flush buffer byte ra file.
    def _flush_out(self) -> None:
        if self._out_buf:
            self.f.write(self._out_buf)
            self._out_buf.clear()

    # Ghi clen bit của code_int vào bitstream.
    def write_bits(self, code_int: int, clen: int) -> None:
        if clen <= 0:
            return
        self.bit_buf = (self.bit_buf << clen) | code_int
        self.bit_len += clen
        self.total_bits += clen
        while self.bit_len >= 8:
            self.bit_len -= 8
            b = (self.bit_buf >> self.bit_len) & 0xFF
            self._out_buf.append(b)
            if len(self._out_buf) >= WRITE_BYTES_FLUSH:
                self._flush_out()
            self.bit_buf &= (1 << self.bit_len) - 1 if self.bit_len else 0

    # Kết thúc stream: thêm padding và flush phần còn lại, trả về số bit padding (0..7).
    def finish(self) -> int:
        padding = (8 - (self.total_bits % 8)) % 8
        if padding:
            self.bit_buf <<= padding
            self.bit_len += padding
            while self.bit_len >= 8:
                self.bit_len -= 8
                b = (self.bit_buf >> self.bit_len) & 0xFF
                self._out_buf.append(b)
                if len(self._out_buf) >= WRITE_BYTES_FLUSH:
                    self._flush_out()
                self.bit_buf &= (1 << self.bit_len) - 1 if self.bit_len else 0
        self._flush_out()
        return padding


# Nén file token bytes bằng Huffman-over-bytes, ghi thẳng ra out_f.
# Trả về padding bits (0..7).
def _compress_token_file_huffman_bytes(
    token_path: str,
    out_f,
    freq_table: Dict[int, int],
    progress_callback: ProgressCallback = None,
) -> int:
    root = _build_btree(freq_table)
    codes = _build_bcodes(root)
    code_table = [(0, 0)] * 256
    for sym, code in codes.items():
        code_table[sym] = (int(code, 2), len(code))

    total = sum(freq_table.values())
    processed = 0
    writer = _BitWriter(out_f)

    write_bits = writer.write_bits

    with open(token_path, "rb") as tf:
        for chunk in iter(lambda: tf.read(READ_CHUNK_SIZE), b""):
            if not chunk:
                break
            for bb in chunk:
                ci, clen = code_table[bb]
                write_bits(ci, clen)
            processed += len(chunk)
            if progress_callback and total:
                progress_callback(processed, total)

    padding = writer.finish()
    if progress_callback and total:
        progress_callback(total, total)
    return padding


# API expected by GUI
compress_pure_huffman = compress_to_file


# Nén theo pipeline LZ77 + Huffman-over-bytes.
# Nếu LZ77 không mang lại lợi ích (token phình), tự fallback về HF2.
def compress_huffman_lz77(
    input_path: str,
    output_path: str,
    progress_callback: ProgressCallback = None,
    *,
    fast: bool = False,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    src_size = os.path.getsize(input_path)

    freq_arr = [0] * 256
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="hfz_tokens_", suffix=".bin")
    os.close(tmp_fd)

    try:
        with open(tmp_path, "wb") as tok_out:
            lz_stats: Dict[str, int] = {}

            def cb1(done: int, total: int) -> None:
                if not progress_callback:
                    return
                if total <= 0:
                    return
                progress_callback(done, src_size * 2)

            _lz77_encode_to_stream(
                input_path,
                tok_out,
                freq_arr,
                stats=lz_stats,
                progress_callback=cb1,
                max_candidates=(12 if fast else LZ_MAX_CANDIDATES),
                max_match=(96 if fast else LZ_MAX_MATCH),
                window_size=(16_384 if fast else LZ_WINDOW),
                insert_step=(4 if fast else 1),
            )

        matches = lz_stats.get("matches", 0)
        match_bytes = lz_stats.get("match_bytes", 0)
        token_size = os.path.getsize(tmp_path)
        if src_size > 0 and (matches == 0 or match_bytes < max(4096, src_size // 50)):
            if token_size >= int(src_size * 1.20):
                return compress_to_file(input_path, output_path, progress_callback=progress_callback)

        freq_b = {i: c for i, c in enumerate(freq_arr) if c}

        with open(output_path, "wb+") as out:
            out.write(MAGIC_LZ)
            out.write(VERSION_LZ.to_bytes(1, "big"))
            out.write((1).to_bytes(1, "big"))
            out.write(_encode_varint(src_size))

            out.write(_u16be(len(freq_b)))
            for sym in sorted(freq_b.keys()):
                out.write(bytes((sym,)))
                out.write(_encode_varint(freq_b[sym]))

            pad_pos = out.tell()
            out.write(b"\x00")

            def cb2(done: int, total: int) -> None:
                if not progress_callback:
                    return
                if total <= 0:
                    return
                mapped = src_size + int(done * src_size / total)
                if mapped > src_size * 2:
                    mapped = src_size * 2
                progress_callback(mapped, src_size * 2)

            padding = _compress_token_file_huffman_bytes(
                tmp_path,
                out,
                freq_b,
                progress_callback=cb2,
            )

            out.seek(pad_pos)
            out.write(padding.to_bytes(1, "big"))

        if progress_callback:
            progress_callback(src_size * 2, src_size * 2)
        return {}, {}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
