// Binary save/load (format documented in docs/FORMAT.md) and the POSIX
// read-only mapping used by load(mmap = true).
//
// Every field read from disk is validated before use: a truncated or
// corrupted file must produce std::runtime_error, never a crash.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <string>

#include "hnsw/index.hpp"
#include "index_impl.hpp"
#include "mmap_file.hpp"

namespace hnsw {

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------
namespace detail {

MappedFile::MappedFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  if (st.st_size <= 0) {
    ::close(fd);
    throw std::runtime_error(path + ": empty file");
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  data_ = static_cast<const unsigned char*>(p);
  size_ = size;
}

MappedFile::~MappedFile() {
  reset();
}

MappedFile::MappedFile(MappedFile&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MappedFile::reset() noexcept {
  if (data_ != nullptr) ::munmap(const_cast<unsigned char*>(data_), size_);
  data_ = nullptr;
  size_ = 0;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Format
// ---------------------------------------------------------------------------
namespace {

constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'E', 'N', 'G', '\0'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304;
constexpr std::size_t kHeaderSize = 256;
constexpr std::size_t kSectionAlign = 64;

struct Header {
  uint64_t dim = 0;
  uint32_t metric = 0;
  uint64_t M = 0;
  uint64_t ef_construction = 0;
  uint64_t ef_search = 0;
  uint64_t seed = 0;
  uint64_t max_elements = 0;
  uint64_t count = 0;
  uint64_t padded_dim = 0;
  int32_t max_level = -1;
  uint32_t entry_point = detail::kNoNode;
  uint64_t upper_used = 0;
  uint64_t num_deleted = 0;
  uint64_t rng_state = 0;
  uint64_t off_vectors = 0;
  uint64_t off_level0 = 0;
  uint64_t off_labels = 0;
  uint64_t off_levels = 0;
  uint64_t off_deleted = 0;
  uint64_t off_upper = 0;
  uint64_t file_size = 0;
  uint64_t payload_checksum = 0;
};

// Byte offsets of each field within the 256-byte header block.
namespace off {
constexpr std::size_t magic = 0, version = 8, endian = 12, dim = 16, metric = 24, M = 32,
                      ef_construction = 40, ef_search = 48, seed = 56, max_elements = 64, count = 72,
                      padded_dim = 80, max_level = 88, entry_point = 92, upper_used = 96, num_deleted = 104,
                      rng_state = 112, off_vectors = 120, off_level0 = 128, off_labels = 136,
                      off_levels = 144, off_deleted = 152, off_upper = 160, file_size = 168,
                      payload_checksum = 176, checksum = 184;
}  // namespace off

uint64_t fnv1a(const unsigned char* p, std::size_t n) noexcept {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

template <typename T>
void put(unsigned char* buf, std::size_t at, T v) {
  std::memcpy(buf + at, &v, sizeof(T));
}
template <typename T>
T get(const unsigned char* buf, std::size_t at) {
  T v;
  std::memcpy(&v, buf + at, sizeof(T));
  return v;
}

/// Header checksum: FNV-1a over the whole 256-byte header with the checksum
/// field itself zeroed.
uint64_t header_checksum(const unsigned char* buf) noexcept {
  unsigned char tmp[kHeaderSize];
  std::memcpy(tmp, buf, kHeaderSize);
  std::memset(tmp + off::checksum, 0, sizeof(uint64_t));
  return fnv1a(tmp, kHeaderSize);
}

/// Payload checksum, updated section by section. Each step is a bijection of
/// the state for a fixed input word, so any single changed word is detected.
struct PayloadHash {
  uint64_t h = 0x243F6A8885A308D3ULL;
  void mix(uint64_t w) noexcept {
    h ^= w * 0x9E3779B97F4A7C15ULL;
    h = (h << 31) | (h >> 33);
    h *= 0xC2B2AE3D27D4EB4FULL;
  }
  void update(const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      uint64_t w;
      std::memcpy(&w, p + i, 8);
      mix(w);
    }
    for (; i < n; ++i) mix(0x100ULL | p[i]);
    mix(n);  // bind section length
  }
};

std::size_t align_up(std::size_t v) {
  return (v + kSectionAlign - 1) / kSectionAlign * kSectionAlign;
}

struct Layout {
  std::size_t vectors, level0, labels, levels, deleted, upper;  // byte sizes
};

Layout section_sizes(uint64_t count, uint64_t padded_dim, uint64_t M, uint64_t upper_used) {
  return Layout{count * padded_dim * sizeof(float),
                count * (1 + 2 * M) * sizeof(uint32_t),
                count * sizeof(uint64_t),
                count,
                count,
                upper_used * sizeof(uint32_t)};
}

void fill_offsets(Header& h) {
  const Layout L = section_sizes(h.count, h.padded_dim, h.M, h.upper_used);
  std::size_t pos = kHeaderSize;
  h.off_vectors = pos;
  pos = align_up(pos + L.vectors);
  h.off_level0 = pos;
  pos = align_up(pos + L.level0);
  h.off_labels = pos;
  pos = align_up(pos + L.labels);
  h.off_levels = pos;
  pos = align_up(pos + L.levels);
  h.off_deleted = pos;
  pos = align_up(pos + L.deleted);
  h.off_upper = pos;
  pos += L.upper;
  h.file_size = pos;
}

void encode_header(const Header& h, unsigned char* buf) {
  std::memset(buf, 0, kHeaderSize);
  std::memcpy(buf + off::magic, kMagic, sizeof(kMagic));
  put<uint32_t>(buf, off::version, kVersion);
  put<uint32_t>(buf, off::endian, kEndianMarker);
  put(buf, off::dim, h.dim);
  put(buf, off::metric, h.metric);
  put(buf, off::M, h.M);
  put(buf, off::ef_construction, h.ef_construction);
  put(buf, off::ef_search, h.ef_search);
  put(buf, off::seed, h.seed);
  put(buf, off::max_elements, h.max_elements);
  put(buf, off::count, h.count);
  put(buf, off::padded_dim, h.padded_dim);
  put(buf, off::max_level, h.max_level);
  put(buf, off::entry_point, h.entry_point);
  put(buf, off::upper_used, h.upper_used);
  put(buf, off::num_deleted, h.num_deleted);
  put(buf, off::rng_state, h.rng_state);
  put(buf, off::off_vectors, h.off_vectors);
  put(buf, off::off_level0, h.off_level0);
  put(buf, off::off_labels, h.off_labels);
  put(buf, off::off_levels, h.off_levels);
  put(buf, off::off_deleted, h.off_deleted);
  put(buf, off::off_upper, h.off_upper);
  put(buf, off::file_size, h.file_size);
  put(buf, off::payload_checksum, h.payload_checksum);
  put<uint64_t>(buf, off::checksum, header_checksum(buf));
}

[[noreturn]] void corrupt(const std::string& path, const std::string& why) {
  throw std::runtime_error("corrupt or incompatible index file '" + path + "': " + why);
}

Header decode_header(const unsigned char* buf, std::size_t actual_size, const std::string& path) {
  if (actual_size < kHeaderSize) corrupt(path, "file too small for header");
  if (std::memcmp(buf + off::magic, kMagic, sizeof(kMagic)) != 0) corrupt(path, "bad magic");
  if (get<uint32_t>(buf, off::endian) != kEndianMarker) corrupt(path, "endianness mismatch");
  if (get<uint32_t>(buf, off::version) != kVersion) {
    corrupt(path, "unsupported version " + std::to_string(get<uint32_t>(buf, off::version)));
  }
  if (get<uint64_t>(buf, off::checksum) != header_checksum(buf)) corrupt(path, "header checksum mismatch");

  Header h;
  h.dim = get<uint64_t>(buf, off::dim);
  h.metric = get<uint32_t>(buf, off::metric);
  h.M = get<uint64_t>(buf, off::M);
  h.ef_construction = get<uint64_t>(buf, off::ef_construction);
  h.ef_search = get<uint64_t>(buf, off::ef_search);
  h.seed = get<uint64_t>(buf, off::seed);
  h.max_elements = get<uint64_t>(buf, off::max_elements);
  h.count = get<uint64_t>(buf, off::count);
  h.padded_dim = get<uint64_t>(buf, off::padded_dim);
  h.max_level = get<int32_t>(buf, off::max_level);
  h.entry_point = get<uint32_t>(buf, off::entry_point);
  h.upper_used = get<uint64_t>(buf, off::upper_used);
  h.num_deleted = get<uint64_t>(buf, off::num_deleted);
  h.rng_state = get<uint64_t>(buf, off::rng_state);
  h.off_vectors = get<uint64_t>(buf, off::off_vectors);
  h.off_level0 = get<uint64_t>(buf, off::off_level0);
  h.off_labels = get<uint64_t>(buf, off::off_labels);
  h.off_levels = get<uint64_t>(buf, off::off_levels);
  h.off_deleted = get<uint64_t>(buf, off::off_deleted);
  h.off_upper = get<uint64_t>(buf, off::off_upper);
  h.file_size = get<uint64_t>(buf, off::file_size);
  h.payload_checksum = get<uint64_t>(buf, off::payload_checksum);

  // Scalar sanity (bounds keep every later size computation overflow-free).
  if (h.file_size != actual_size) corrupt(path, "size mismatch (truncated or extended file)");
  if (h.dim == 0 || h.dim > (1u << 20)) corrupt(path, "invalid dim");
  if (h.metric > 2) corrupt(path, "invalid metric");
  if (h.M < 2 || h.M > 2048) corrupt(path, "invalid M");
  if (h.ef_construction == 0 || h.ef_search == 0) corrupt(path, "invalid ef");
  if (h.padded_dim != detail::round_up(h.dim, detail::kPadFloats)) corrupt(path, "invalid padded dim");
  if (h.count > actual_size || h.count >= detail::kNoNode) corrupt(path, "invalid count");
  if (h.upper_used > actual_size) corrupt(path, "invalid upper size");
  if (h.num_deleted > h.count) corrupt(path, "invalid deleted count");
  if (h.max_elements != 0 && (h.max_elements < h.count || h.max_elements >= detail::kNoNode)) {
    corrupt(path, "invalid max_elements");
  }
  if (h.count == 0) {
    if (h.max_level != -1 || h.entry_point != detail::kNoNode) corrupt(path, "invalid empty-index header");
  } else if (h.max_level < 0 || h.max_level > detail::kMaxLevel || h.entry_point >= h.count) {
    corrupt(path, "invalid entry point / max level");
  }

  // Section layout must be exactly what save() writes.
  Header expect = h;
  fill_offsets(expect);
  if (expect.off_vectors != h.off_vectors || expect.off_level0 != h.off_level0 ||
      expect.off_labels != h.off_labels || expect.off_levels != h.off_levels ||
      expect.off_deleted != h.off_deleted || expect.off_upper != h.off_upper ||
      expect.file_size != h.file_size) {
    corrupt(path, "section offsets inconsistent with header");
  }
  return h;
}

Params params_from(const Header& h) {
  Params p;
  p.dim = h.dim;
  p.metric = static_cast<Metric>(h.metric);
  p.M = h.M;
  p.ef_construction = h.ef_construction;
  p.max_elements = h.max_elements;
  p.seed = h.seed;
  return p;
}

/// Validate graph content and derive upper_offset / label map / deleted.
void validate_and_index(Index::Impl& im, const Header& h, const uint8_t* deleted_src,
                        const std::string& path) {
  const std::size_t n = h.count;
  // Levels and upper offsets.
  im.upper_offset.assign(n, 0);
  uint64_t upper_pos = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int lv = im.levels[i];
    if (lv > h.max_level) corrupt(path, "node level exceeds max level");
    im.upper_offset[i] = upper_pos;
    upper_pos += static_cast<uint64_t>(lv) * im.upper_stride;
  }
  if (upper_pos != h.upper_used) corrupt(path, "upper-layer size does not match levels");
  if (n > 0 && im.levels[h.entry_point] != h.max_level) corrupt(path, "entry point is not on the top level");

  // Adjacency: degree bounds, neighbor ids in range and present on that layer.
  for (std::size_t i = 0; i < n; ++i) {
    const auto id = static_cast<uint32_t>(i);
    for (int lc = 0; lc <= im.levels[i]; ++lc) {
      const uint32_t* l = im.list(id, lc);
      if (l[0] > im.max_degree(lc)) corrupt(path, "neighbor count exceeds maximum degree");
      for (uint32_t j = 0; j < l[0]; ++j) {
        const uint32_t nb = l[1 + j];
        if (nb >= n || im.levels[nb] < lc) corrupt(path, "neighbor id out of range");
      }
    }
  }

  // Labels must be unique; tombstones are 0/1.
  im.label_to_id.clear();
  im.label_to_id.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!im.label_to_id.emplace(im.labels[i], static_cast<uint32_t>(i)).second) {
      corrupt(path, "duplicate label");
    }
  }
  im.deleted.assign(deleted_src, deleted_src + n);
  std::size_t dels = 0;
  for (uint8_t d : im.deleted) {
    if (d > 1) corrupt(path, "invalid tombstone byte");
    dels += d;
  }
  if (dels != h.num_deleted) corrupt(path, "deleted count mismatch");
  im.num_deleted = dels;
}

/// Sequential section reader for the non-mmap path: verifies that the gap
/// before each section is zero padding and feeds the section to the hash.
class SectionReader {
 public:
  SectionReader(std::ifstream& in, const std::string& path) : in_(in), path_(path) {}
  void read(void* dst, std::size_t bytes, uint64_t offset) {
    if (offset < pos_) corrupt(path_, "section layout error");
    char pad[kSectionAlign];
    const std::size_t gap = offset - pos_;
    if (gap > sizeof(pad) || (gap > 0 && !in_.read(pad, static_cast<std::streamsize>(gap)))) {
      corrupt(path_, "short read");
    }
    for (std::size_t i = 0; i < gap; ++i) {
      if (pad[i] != 0) corrupt(path_, "non-zero padding");
    }
    if (bytes > 0 && !in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes))) {
      corrupt(path_, "short read");
    }
    hash_.update(dst, bytes);
    pos_ = offset + bytes;
  }
  uint64_t checksum() const { return hash_.h; }

 private:
  std::ifstream& in_;
  const std::string& path_;
  uint64_t pos_ = kHeaderSize;
  PayloadHash hash_;
};

/// mmap path: same checks directly on the mapping.
void verify_mapped_payload(const unsigned char* base, const Header& h, const std::string& path) {
  const Layout L = section_sizes(h.count, h.padded_dim, h.M, h.upper_used);
  const std::pair<uint64_t, std::size_t> sections[] = {{h.off_vectors, L.vectors}, {h.off_level0, L.level0},
                                                       {h.off_labels, L.labels},   {h.off_levels, L.levels},
                                                       {h.off_deleted, L.deleted}, {h.off_upper, L.upper}};
  PayloadHash hash;
  uint64_t pos = kHeaderSize;
  for (const auto& [offset, bytes] : sections) {
    for (uint64_t i = pos; i < offset; ++i) {
      if (base[i] != 0) corrupt(path, "non-zero padding");
    }
    hash.update(base + offset, bytes);
    pos = offset + bytes;
  }
  if (hash.h != h.payload_checksum) corrupt(path, "payload checksum mismatch");
}

}  // namespace

// ---------------------------------------------------------------------------
// save / load
// ---------------------------------------------------------------------------

void Index::save(const std::string& path) const {
  std::shared_lock<std::shared_mutex> api(impl_->api_mutex);
  const Impl& im = *impl_;
  Header h;
  h.dim = im.dim;
  h.metric = static_cast<uint32_t>(im.params.metric);
  h.M = im.M;
  h.ef_construction = im.params.ef_construction;
  h.ef_search = im.ef_search;
  h.seed = im.params.seed;
  h.max_elements = im.params.max_elements;
  h.count = im.count;
  h.padded_dim = im.padded_dim;
  h.max_level = im.max_level;
  h.entry_point = im.entry_point;
  h.upper_used = im.upper_used;
  h.num_deleted = im.num_deleted;
  h.rng_state = im.rng_state;
  fill_offsets(h);
  {
    const Layout L = section_sizes(h.count, h.padded_dim, h.M, h.upper_used);
    PayloadHash hash;
    hash.update(im.vecs, L.vectors);
    hash.update(im.l0, L.level0);
    hash.update(im.labels, L.labels);
    hash.update(im.levels, L.levels);
    hash.update(im.deleted.data(), L.deleted);
    hash.update(im.upper, L.upper);
    h.payload_checksum = hash.h;
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open " + path + " for writing");
  unsigned char header[kHeaderSize];
  encode_header(h, header);
  out.write(reinterpret_cast<const char*>(header), kHeaderSize);

  const Layout L = section_sizes(h.count, h.padded_dim, h.M, h.upper_used);
  const char zeros[kSectionAlign] = {};
  auto section = [&](uint64_t offset, const void* data, std::size_t bytes) {
    const auto pos = static_cast<uint64_t>(out.tellp());
    if (offset < pos) throw std::logic_error("save: section layout error");
    out.write(zeros, static_cast<std::streamsize>(offset - pos));
    if (bytes > 0) out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  };
  section(h.off_vectors, im.vecs, L.vectors);
  section(h.off_level0, im.l0, L.level0);
  section(h.off_labels, im.labels, L.labels);
  section(h.off_levels, im.levels, L.levels);
  section(h.off_deleted, im.deleted.data(), L.deleted);
  section(h.off_upper, im.upper, L.upper);
  out.flush();
  if (!out) throw std::runtime_error("write failed: " + path);
}

Index Index::load(const std::string& path, bool mmap) {
  if (mmap) {
    detail::MappedFile file(path);
    const Header h = decode_header(file.data(), file.size(), path);
    Params p = params_from(h);
    p.max_elements = 0;  // nothing is allocated: arrays live in the mapping
    auto im = std::make_unique<Impl>(p);
    im->params.max_elements = h.max_elements;
    im->count = h.count;
    im->capacity = h.count;
    im->ef_search = h.ef_search;
    im->upper_used = h.upper_used;
    im->entry_point = h.entry_point;
    im->max_level = h.max_level;
    im->rng_state = h.rng_state;
    const unsigned char* base = file.data();
    // Sections are 64-byte aligned within a page-aligned mapping.
    im->vecs = reinterpret_cast<const float*>(base + h.off_vectors);
    im->l0 = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(base + h.off_level0));
    im->labels = reinterpret_cast<const uint64_t*>(base + h.off_labels);
    im->levels = base + h.off_levels;
    im->upper = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(base + h.off_upper));
    verify_mapped_payload(base, h, path);
    validate_and_index(*im, h, base + h.off_deleted, path);
    im->mapping = std::move(file);
    im->read_only = true;
    return Index(std::move(im));
  }

  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const auto actual = static_cast<std::size_t>(in.tellg());
  unsigned char header[kHeaderSize];
  if (actual < kHeaderSize) corrupt(path, "file too small for header");
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(header), kHeaderSize)) corrupt(path, "short read");
  const Header h = decode_header(header, actual, path);

  Params p = params_from(h);
  p.max_elements = 0;  // allocate exactly count; growth policy restored below
  auto im = std::make_unique<Impl>(p);
  im->params.max_elements = h.max_elements;
  if (h.count > 0) im->reserve(std::max<std::size_t>(h.count, h.max_elements));
  const Layout L = section_sizes(h.count, h.padded_dim, h.M, h.upper_used);
  SectionReader reader(in, path);
  reader.read(im->vec_store.data(), L.vectors, h.off_vectors);
  reader.read(im->l0_store.data(), L.level0, h.off_level0);
  reader.read(im->label_store.data(), L.labels, h.off_labels);
  reader.read(im->level_store.data(), L.levels, h.off_levels);
  std::vector<uint8_t> deleted(h.count);
  reader.read(deleted.data(), L.deleted, h.off_deleted);
  im->upper_store.assign(h.upper_used, 0u);
  reader.read(im->upper_store.data(), L.upper, h.off_upper);
  if (reader.checksum() != h.payload_checksum) corrupt(path, "payload checksum mismatch");
  im->refresh_pointers();

  im->count = h.count;
  im->ef_search = h.ef_search;
  im->upper_used = h.upper_used;
  im->entry_point = h.entry_point;
  im->max_level = h.max_level;
  im->rng_state = h.rng_state;
  validate_and_index(*im, h, deleted.data(), path);
  im->deleted.resize(im->capacity, 0);
  return Index(std::move(im));
}

}  // namespace hnsw
