import heapq
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

MAGIC = b"HF2"
MAGIC_LZ = b"HFZ"

LZ_T_LITERAL = 0
LZ_T_MATCH = 1
LZ_WINDOW = 32768
LZ_MIN_MATCH = 4
LZ_MAX_MATCH = 258
READ_CHUNK_SIZE = 262_144
WRITE_CHARS_FLUSH = 200_000
DECODE_TABLE_BITS = 12

ProgressCallback = Optional[Callable[[int, int], None]]

# Giải mã varint trực tiếp từ file/stream.
def _decode_varint_from_stream(f) -> int:
    shift = 0
    value = 0
    while True:
        b = f.read(1)
        if len(b) != 1:
            raise ValueError("Invalid varint (truncated)")
        byte = b[0]
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value
        shift += 7
        if shift > 63:
            raise ValueError("Invalid varint (too long)")

# Đọc đúng n bytes, thiếu thì raise.
def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("Unexpected EOF")
    return data


# Chuyển 2 bytes big-endian -> số nguyên.
def _from_u16be(b: bytes) -> int:
    return int.from_bytes(b, "big")


class _BitReader:
    # Khởi tạo bộ đọc bit theo luồng, giới hạn tổng số bit cần đọc.
    def __init__(self, f, total_bits: int) -> None:
        self.f = f
        self.total_bits = total_bits
        self.bits_read = 0
        self.bit_buf = 0
        self.bit_len = 0

    # Nạp thêm byte từ file vào buffer để đảm bảo có ít nhất min_bits trong buffer.
    def _fill(self, min_bits: int) -> None:
        while self.bit_len < min_bits and self.bits_read < self.total_bits:
            b = self.f.read(1)
            if len(b) != 1:
                raise ValueError("Unexpected EOF")
            byte = b[0]
            remaining = self.total_bits - self.bits_read
            take = 8 if remaining >= 8 else remaining
            if take < 8:
                byte >>= 8 - take
            self.bit_buf = (self.bit_buf << take) | byte
            self.bit_len += take
            self.bits_read += take

    # Số bit còn lại có thể đọc được (tính theo total_bits).
    def remaining_bits(self) -> int:
        consumed = self.bits_read - self.bit_len
        return self.total_bits - consumed

    # Xem trước n bit tiếp theo nhưng không loại khỏi buffer.
    def peek(self, n: int) -> int:
        if n <= 0:
            return 0
        self._fill(n)
        if self.bit_len < n:
            raise ValueError("Unexpected EOF")
        return (self.bit_buf >> (self.bit_len - n)) & ((1 << n) - 1)

    # Bỏ đi n bit khỏi buffer (sau khi đã peek/giải mã).
    def drop(self, n: int) -> None:
        if n <= 0:
            return
        if self.bit_len < n:
            self._fill(n)
        if self.bit_len < n:
            raise ValueError("Unexpected EOF")
        self.bit_len -= n
        if self.bit_len:
            self.bit_buf &= (1 << self.bit_len) - 1
        else:
            self.bit_buf = 0

# Tạo bảng tra nhanh Huffman để decode theo prefix (để tăng tốc giải nén).
def _make_decode_table(root, table_bits: int):
    size = 1 << table_bits
    table = [None] * size
    for prefix in range(size):
        node = root
        clen = 0
        for i in range(table_bits):
            if node is None:
                break
            if node.sym is not None:
                break
            bit = (prefix >> (table_bits - 1 - i)) & 1
            node = node.left if bit == 0 else node.right
            clen += 1
            if node is None:
                break
            if node.sym is not None:
                break

        if node is None:
            table[prefix] = (None, 0, None)
        elif node.sym is not None:
            table[prefix] = (node.sym, clen, None)
        else:
            table[prefix] = (None, table_bits, node)
    return table


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
        return _Node(only.freq, only.min_sym, sym=None, left=only, right=None)

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

def _read_hfz_header(f) -> Tuple[int, Dict[int, int], int]:
    ver_b = f.read(1)
    if len(ver_b) != 1:
        raise ValueError("Invalid HFZ file (missing version)")
    _ = ver_b[0]
    flags_b = f.read(1)
    if len(flags_b) != 1:
        raise ValueError("Invalid HFZ file (missing flags)")
    _flags = flags_b[0]

    orig_len = _decode_varint_from_stream(f)
    n_syms = _from_u16be(_read_exact(f, 2))

    freq_b: Dict[int, int] = {}
    for _i in range(n_syms):
        sym = _read_exact(f, 1)[0]
        freq_b[sym] = _decode_varint_from_stream(f)

    pad_b = f.read(1)
    if len(pad_b) != 1:
        raise ValueError("Invalid HFZ file (missing padding)")
    padding = pad_b[0]
    return orig_len, freq_b, padding


class _LZ77StreamDecoder:
    # Khởi tạo bộ giải mã LZ77 theo luồng
    def __init__(self, out_file) -> None:
        self.out_file = out_file
        self.win = bytearray(LZ_WINDOW)
        self.win_pos = 0
        self.produced = 0
        self.write_buf = bytearray()

        self.state = "type"
        self.need = None
        self.v_shift = 0
        self.v_value = 0
        self.dist = 0

    # Ghi ra 1 byte output, đồng thời cập nhật window và buffer ghi file.
    def _emit(self, b: int) -> None:
        self.write_buf.append(b)
        self.win[self.win_pos] = b
        self.win_pos = (self.win_pos + 1) % LZ_WINDOW
        self.produced += 1
        if len(self.write_buf) >= 1_048_576:
            self.out_file.write(self.write_buf)
            self.write_buf.clear()

    # Flush phần buffer ghi file còn lại.
    def _flush(self) -> None:
        if self.write_buf:
            self.out_file.write(self.write_buf)
            self.write_buf.clear()

    # Feed từng byte varint và trả về giá trị khi đủ byte (ngược lại trả None).
    def _varint_feed(self, byte: int) -> Optional[int]:
        self.v_value |= (byte & 0x7F) << self.v_shift
        if byte & 0x80:
            self.v_shift += 7
            if self.v_shift > 63:
                raise ValueError("Invalid varint in LZ77 token stream")
            return None
        val = self.v_value
        self.v_shift = 0
        self.v_value = 0
        return val

    # Nhận 1 byte token LZ77 và giải mã theo state machine, xuất ra output bytes.
    def feed(self, b: int) -> None:
        if self.state == "type":
            if b == LZ_T_LITERAL:
                self.state = "lit"
                return
            if b == LZ_T_MATCH:
                self.state = "varint"
                self.need = "dist"
                return
            raise ValueError("Unknown LZ77 token type")

        if self.state == "lit":
            self._emit(b)
            self.state = "type"
            return

        val = self._varint_feed(b)
        if val is None:
            return
        if self.need == "dist":
            self.dist = val
            if self.dist <= 0 or self.dist > LZ_WINDOW:
                raise ValueError("Invalid LZ77 distance")
            if self.dist > self.produced:
                raise ValueError("LZ77 distance beyond output")
            self.need = "len"
            return

        length = val
        if length < LZ_MIN_MATCH or length > LZ_MAX_MATCH:
            raise ValueError("Invalid LZ77 length")

        read_pos = (self.win_pos - self.dist) % LZ_WINDOW
        for _ in range(length):
            bb = self.win[read_pos]
            self._emit(bb)
            read_pos = (read_pos + 1) % LZ_WINDOW
        self.state = "type"
        self.need = None

    # Kết thúc decode và flush; trả về số byte đã tạo ra.
    def finish(self) -> int:
        if self.state != "type":
            raise ValueError("Truncated LZ77 token stream")
        self._flush()
        return self.produced

def _decompress_hfz(f, file_size: int, output_path: str, progress_callback: ProgressCallback,) -> None:
    
    orig_len, freq_b, padding = _read_hfz_header(f)
    total_tokens = sum(freq_b.values())

    if total_tokens == 0:
        with open(output_path, "wb") as out:
            out.write(b"")
        if progress_callback:
            progress_callback(0, 0)
        return

    root = _build_btree(freq_b)
    if root is None:
        raise ValueError("Invalid HFZ tree")

    decode_table = _make_decode_table(root, DECODE_TABLE_BITS)

    data_start = f.tell()
    data_bytes = file_size - data_start
    total_bits = data_bytes * 8 - padding if data_bytes > 0 else 0

    decoded_tokens = 0
    progress_interval = max(1024, total_tokens // 200)
    br = _BitReader(f, total_bits)

    with open(output_path, "wb") as out:
        lz = _LZ77StreamDecoder(out)
        while decoded_tokens < total_tokens:
            rem = br.remaining_bits()
            if rem <= 0:
                break

            if rem >= DECODE_TABLE_BITS:
                prefix = br.peek(DECODE_TABLE_BITS)
                sym, clen, node = decode_table[prefix]
                if sym is not None:
                    br.drop(clen)
                    lz.feed(sym)
                else:
                    if node is None:
                        raise ValueError("Corrupted HFZ data")
                    br.drop(DECODE_TABLE_BITS)
                    cur = node
                    while cur.sym is None:
                        bit = br.peek(1)
                        br.drop(1)
                        cur = cur.left if bit == 0 else cur.right
                        if cur is None:
                            raise ValueError("Corrupted HFZ data")
                    lz.feed(cur.sym)
            else:
                cur = root
                while cur.sym is None:
                    bit = br.peek(1)
                    br.drop(1)
                    cur = cur.left if bit == 0 else cur.right
                    if cur is None:
                        raise ValueError("Corrupted HFZ data")
                lz.feed(cur.sym)

            decoded_tokens += 1
            if progress_callback and (
                decoded_tokens % progress_interval == 0 or decoded_tokens == total_tokens
            ):
                progress_callback(decoded_tokens, total_tokens)
        produced = lz.finish()

    if decoded_tokens != total_tokens:
        raise ValueError("Truncated HFZ data")

    if orig_len != produced:
        raise ValueError("HFZ output length mismatch")
    if progress_callback:
        progress_callback(total_tokens, total_tokens)

def _read_hf2_freq_table(f) -> Dict[str, int]:
    ver_b = f.read(1)
    if len(ver_b) != 1:
        raise ValueError("Invalid file (missing version)")
    _ = ver_b[0]
    n_chars_b = f.read(4)
    if len(n_chars_b) != 4:
        raise ValueError("Invalid file (missing n_chars)")
    n_chars = int.from_bytes(n_chars_b, "big")

    freq_table: Dict[str, int] = {}
    for _i in range(n_chars):
        lb = f.read(1)
        if len(lb) != 1:
            raise ValueError("Invalid file (missing symbol length)")
        L = lb[0]
        chb = f.read(L)
        if len(chb) != L:
            raise ValueError("Invalid file (missing symbol bytes)")
        ch = chb.decode("utf-8", errors="strict")
        freq_table[ch] = _decode_varint_from_stream(f)
    return freq_table


# Đọc header legacy (không magic), trả về freq_table.
def _read_legacy_freq_table(f) -> Dict[str, int]:
    n_chars_b = f.read(4)
    if len(n_chars_b) != 4:
        raise ValueError("Invalid file")
    n_chars = int.from_bytes(n_chars_b, "big")
    freq_table: Dict[str, int] = {}

    for _i in range(n_chars):
        lb = f.read(1)
        if len(lb) != 1:
            raise ValueError("Invalid file")
        L = lb[0]
        chb = f.read(L)
        if len(chb) != L:
            raise ValueError("Invalid file")
        ch = chb.decode("utf-8", errors="strict")
        fb = f.read(4)
        if len(fb) != 4:
            raise ValueError("Invalid file")
        freq_table[ch] = int.from_bytes(fb, "big")
    return freq_table

# Giải mã payload Huffman text  và ghi ra output_path.
def _decode_huffman_text_payload(
    f,
    file_size: int,
    output_path: str,
    freq_table: Dict[str, int],
    progress_callback: ProgressCallback,
) -> Dict[str, str]:
    
    pad_b = f.read(1)
    if len(pad_b) != 1:
        raise ValueError("Invalid file (missing padding)")
    padding = pad_b[0]

    data_start = f.tell()
    data_bytes = file_size - data_start
    total_bits = data_bytes * 8 - padding if data_bytes > 0 else 0

    total_chars = sum(freq_table.values())
    if total_chars == 0:
        with open(output_path, "w", encoding="utf-8", newline="") as out:
            out.write("")
        return {}

    root = _build_tree(freq_table)
    if root is None:
        raise ValueError("Invalid tree")

    decode_table = _make_decode_table(root, DECODE_TABLE_BITS)
    decoded = 0
    buf = []
    progress_interval = max(1000, total_chars // 200)
    br = _BitReader(f, total_bits)

    with open(output_path, "w", encoding="utf-8", newline="") as out:
        while decoded < total_chars:
            rem = br.remaining_bits()
            if rem <= 0:
                break

            if rem >= DECODE_TABLE_BITS:
                prefix = br.peek(DECODE_TABLE_BITS)
                sym, clen, node = decode_table[prefix]
                if sym is not None:
                    br.drop(clen)
                    buf.append(sym)
                else:
                    if node is None:
                        raise ValueError("Corrupted data")
                    br.drop(DECODE_TABLE_BITS)
                    cur = node
                    while cur.sym is None:
                        bit = br.peek(1)
                        br.drop(1)
                        cur = cur.left if bit == 0 else cur.right
                        if cur is None:
                            raise ValueError("Corrupted data")
                    buf.append(cur.sym)
            else:
                cur = root
                while cur.sym is None:
                    bit = br.peek(1)
                    br.drop(1)
                    cur = cur.left if bit == 0 else cur.right
                    if cur is None:
                        raise ValueError("Corrupted data")
                buf.append(cur.sym)

            decoded += 1
            if len(buf) >= WRITE_CHARS_FLUSH:
                out.write("".join(buf))
                buf.clear()
            if progress_callback and (
                decoded % progress_interval == 0 or decoded == total_chars
            ):
                progress_callback(decoded, total_chars)

        if buf:
            out.write("".join(buf))

    if progress_callback:
        progress_callback(total_chars, total_chars)

    return _build_codes(root)


# Giải nén định dạng (Huffman text) theo header .
def _decompress_hf2(
    f,
    file_size: int,
    output_path: str,
    progress_callback: ProgressCallback,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    freq_table = _read_hf2_freq_table(f)
    codes = _decode_huffman_text_payload(
        f, file_size, output_path, freq_table, progress_callback
    )
    return freq_table, codes

def _decompress_legacy(f, file_size: int, output_path: str, progress_callback: ProgressCallback,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    f.seek(0)
    freq_table = _read_legacy_freq_table(f)
    codes = _decode_huffman_text_payload(
        f, file_size, output_path, freq_table, progress_callback
    )
    return freq_table, codes


# API giải nén chính: tự nhận dạng định dạng và ghi ra output.
def decompress_from_file(input_path: str, output_path: str, progress_callback: ProgressCallback = None,
) -> Tuple[Dict[str, int], Dict[str, str]]:

    file_size = os.path.getsize(input_path)

    with open(input_path, "rb") as f:
        head = f.read(len(MAGIC))

        if head == MAGIC_LZ:
            _decompress_hfz(f, file_size, output_path, progress_callback)
            return {}, {}

        if head == MAGIC:
            return _decompress_hf2(f, file_size, output_path, progress_callback)

        return _decompress_legacy(f, file_size, output_path, progress_callback)
