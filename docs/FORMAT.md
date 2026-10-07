# On-disk index format (version 1)

Written by `Index::save`, read by `Index::load(path, mmap)`. All integers are
little-endian (the endianness marker rejects files from a big-endian writer).
The file is a fixed 256-byte header followed by six sections, each starting at
a 64-byte-aligned offset; gaps between sections are zero bytes.

## Header (256 bytes)

| offset | type | field | notes |
|---:|---|---|---|
| 0 | char[8] | magic | `"HNSWENG\0"` |
| 8 | u32 | version | `1` |
| 12 | u32 | endian marker | `0x01020304` |
| 16 | u64 | dim | 1 ≤ dim ≤ 2²⁰ |
| 24 | u32 | metric | 0 = L2, 1 = inner product, 2 = cosine |
| 32 | u64 | M | 2 ≤ M ≤ 2048; layer-0 degree is 2M |
| 40 | u64 | ef_construction | |
| 48 | u64 | ef_search | default query ef |
| 56 | u64 | seed | |
| 64 | u64 | max_elements | 0 = grow geometrically |
| 72 | u64 | count | number of nodes (< 2³²−1) |
| 80 | u64 | padded_dim | dim rounded up to a multiple of 16 floats (64 B) |
| 88 | i32 | max_level | −1 iff count = 0 |
| 92 | u32 | entry_point | internal id; `0xFFFFFFFF` iff count = 0 |
| 96 | u64 | upper_used | number of u32 slots in the upper-layer section |
| 104 | u64 | num_deleted | number of tombstones |
| 112 | u64 | rng_state | level-generator state (so appends continue the sequence) |
| 120 | u64 | off_vectors | section offsets (bytes from file start) |
| 128 | u64 | off_level0 | |
| 136 | u64 | off_labels | |
| 144 | u64 | off_levels | |
| 152 | u64 | off_deleted | |
| 160 | u64 | off_upper | |
| 168 | u64 | file_size | must equal the actual file size |
| 176 | u64 | payload_checksum | see below |
| 184 | u64 | header_checksum | FNV-1a 64 over the 256 header bytes with this field zeroed |
| 192 | — | reserved | zero |

## Sections (in file order)

| section | size (bytes) | content |
|---|---|---|
| vectors | count × padded_dim × 4 | float32 vectors, one 64-byte-aligned row per node, zero-padded; cosine vectors are stored unit-normalized |
| level0 | count × (1 + 2M) × 4 | per node: `[count: u32][neighbors: 2M × u32]` (internal ids) |
| labels | count × 8 | u64 external label of each internal id |
| levels | count × 1 | u8 top level of each node |
| deleted | count × 1 | u8 tombstone flag (0/1) |
| upper | upper_used × 4 | for each node with level L ≥ 1, in id order: L blocks of `[count: u32][neighbors: M × u32]` for layers 1..L |

The per-node offset into the upper section is not stored: it is the prefix
sum of `level × (1 + M)` over preceding ids and is recomputed at load time.

## Payload checksum

A 64-bit hash updated section by section (vectors, level0, labels, levels,
deleted, upper). For each section: every 8-byte little-endian word `w` is
mixed as `h ^= w·0x9E3779B97F4A7C15; h = rotl(h, 31); h *= 0xC2B2AE3D27D4EB4F`,
trailing bytes are mixed as `0x100 | byte`, and finally the section length is
mixed in. Each step is a bijection of `h` for a fixed input, so any single
changed word is guaranteed to change the result. Initial value
`0x243F6A8885A308D3`.

## Validation on load

`load` never trusts the file. In order, it checks: header size, magic,
endianness, version, header checksum, `file_size` against the real size, all
scalar ranges, that every section offset equals the layout `save` would
produce, zero padding, the payload checksum, and then the graph itself: each
node level ≤ max_level, the entry point is on the top level, the sum of upper
blocks equals `upper_used`, every neighbor count ≤ the layer's maximum degree,
every neighbor id < count and present on that layer, labels unique, and
tombstone bytes ∈ {0, 1} matching `num_deleted`. Any failure throws
`std::runtime_error` ("corrupt or incompatible index file ..."). The fuzz
tests truncate the file at many offsets and flip single bits at every header
byte and hundreds of random payload bytes; all must throw.

## mmap mode

`load(path, true)` maps the file read-only (`PROT_READ`, `MAP_PRIVATE`) and
points the vector, level-0, label, level and upper arrays directly into the
mapping (sections are 64-byte aligned within a page-aligned mapping). Only
the per-node upper offsets, the tombstone bytes and the label → id hash map
are materialized in memory. The index is read-only: `add` throws, but
`mark_deleted` and `set_ef` work because tombstones live in memory. Load cost
is one sequential pass over the file (checksum + validation), which also
warms the page cache.
