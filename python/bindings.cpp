// pybind11 bindings: NumPy-in / NumPy-out API around hnsw::Index. The GIL is
// released for add/search; Python filter callbacks re-acquire it per call.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hnsw/filter.hpp"
#include "hnsw/index.hpp"
#include "hnsw/metric.hpp"

namespace py = pybind11;
using hnsw::Index;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

/// Validate a (n, dim) or (dim,) float array. Non-float32 / non-contiguous
/// inputs were already converted by forcecast; anything non-numeric raises.
std::pair<std::size_t, bool> check_matrix(const FloatArray& a, std::size_t dim, const char* what) {
  if (a.ndim() == 1) {
    if (static_cast<std::size_t>(a.shape(0)) != dim) {
      throw py::value_error(std::string(what) + ": expected " + std::to_string(dim) + " values, got " +
                            std::to_string(a.shape(0)));
    }
    return {1, true};
  }
  if (a.ndim() != 2) throw py::value_error(std::string(what) + ": expected a 1-D or 2-D array");
  if (static_cast<std::size_t>(a.shape(1)) != dim) {
    throw py::value_error(std::string(what) + ": expected shape (n, " + std::to_string(dim) + "), got (" +
                          std::to_string(a.shape(0)) + ", " + std::to_string(a.shape(1)) + ")");
  }
  return {static_cast<std::size_t>(a.shape(0)), false};
}

FloatArray to_float_array(const py::handle& obj, const char* what) {
  FloatArray arr = FloatArray::ensure(obj);  // returns null (error cleared) on failure
  if (arr) return arr;
  throw py::type_error(std::string(what) + ": expected a numeric array convertible to float32");
}

std::vector<uint64_t> to_labels(const py::object& ids, std::size_t n, std::size_t start) {
  std::vector<uint64_t> labels(n);
  if (ids.is_none()) {
    for (std::size_t i = 0; i < n; ++i) labels[i] = start + i;
    return labels;
  }
  py::array arr = py::array::ensure(ids);
  if (!arr) throw py::type_error("ids: expected an integer array");
  if (arr.ndim() == 0) arr = arr.attr("reshape")(1);
  if (arr.ndim() != 1 || static_cast<std::size_t>(arr.shape(0)) != n) {
    throw py::value_error("ids: expected shape (" + std::to_string(n) + ",)");
  }
  const auto kind = arr.dtype().kind();
  if (kind == 'u') {
    auto u = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    std::memcpy(labels.data(), u.data(), n * sizeof(uint64_t));
  } else if (kind == 'i') {
    auto s = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    for (std::size_t i = 0; i < n; ++i) {
      if (s.data()[i] < 0) throw py::value_error("ids: labels must be non-negative");
      labels[i] = static_cast<uint64_t>(s.data()[i]);
    }
  } else {
    throw py::type_error("ids: expected an integer array");
  }
  return labels;
}

/// Build a C++ filter from None / callable / bool mask / integer allow-list.
/// Must be called with the GIL held; the returned object must also be
/// destroyed with the GIL held (it may own a Python reference).
std::unique_ptr<hnsw::Filter> make_filter(const py::object& filter) {
  if (filter.is_none()) return nullptr;
  if (py::isinstance<py::array>(filter) || py::isinstance<py::list>(filter) ||
      py::isinstance<py::tuple>(filter)) {
    py::array arr = py::array::ensure(filter);
    if (!arr || arr.ndim() != 1) throw py::value_error("filter: expected a 1-D array");
    const auto kind = arr.dtype().kind();
    if (kind == 'b') {
      auto mask = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(arr);
      auto f = std::make_unique<hnsw::BitsetFilter>(static_cast<uint64_t>(mask.shape(0)));
      for (py::ssize_t i = 0; i < mask.shape(0); ++i) {
        if (mask.data()[i]) f->set(static_cast<uint64_t>(i));
      }
      return f;
    }
    if (kind == 'i' || kind == 'u') {
      const auto labels = to_labels(arr, static_cast<std::size_t>(arr.shape(0)), 0);
      return std::make_unique<hnsw::BitsetFilter>(
          hnsw::BitsetFilter::from_labels(labels.data(), labels.size()));
    }
    throw py::type_error("filter: array must be bool (mask over labels) or integer (allowed labels)");
  }
  if (PyCallable_Check(filter.ptr())) {
    py::function fn = py::reinterpret_borrow<py::function>(filter);
    return std::make_unique<hnsw::FunctionFilter>([fn](uint64_t label) {
      py::gil_scoped_acquire gil;
      return fn(label).cast<bool>();
    });
  }
  throw py::type_error("filter: expected None, a callable(label) -> bool, a bool mask or an id array");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "HNSW approximate nearest-neighbor index (C++20 core)";
  m.attr("__version__") = "0.1.0";
  m.def(
      "simd_isa", [] { return std::string(hnsw::simd::isa_name(hnsw::simd::active().isa)); },
      "Name of the SIMD instruction set selected at startup.");

  py::class_<Index>(m, "Index")
      .def(py::init([](std::size_t dim, const std::string& metric, std::size_t M, std::size_t ef_construction,
                       std::size_t max_elements, uint64_t seed) {
             hnsw::Params p;
             p.dim = dim;
             p.metric = hnsw::parse_metric(metric);
             p.M = M;
             p.ef_construction = ef_construction;
             p.max_elements = max_elements;
             p.seed = seed;
             return std::make_unique<Index>(p);
           }),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("M") = 16, py::arg("ef_construction") = 200,
           py::arg("max_elements") = 0, py::arg("seed") = 100)
      .def(
          "add",
          [](Index& self, const py::object& vectors, const py::object& ids, std::size_t num_threads) {
            const FloatArray arr = to_float_array(vectors, "vectors");
            const auto [n, single] = check_matrix(arr, self.dim(), "vectors");
            (void)single;
            const auto labels = to_labels(ids, n, self.size());
            py::gil_scoped_release release;
            self.add_batch(arr.data(), labels.data(), n, num_threads);
          },
          py::arg("vectors"), py::arg("ids") = py::none(), py::arg("num_threads") = 0)
      .def(
          "search",
          [](const Index& self, const py::object& queries, std::size_t k, std::optional<std::size_t> ef,
             std::size_t num_threads, const py::object& filter) -> py::tuple {
            if (k == 0) throw py::value_error("k must be > 0");
            const FloatArray arr = to_float_array(queries, "queries");
            const auto [nq, single] = check_matrix(arr, self.dim(), "queries");
            auto f = make_filter(filter);
            std::vector<hnsw::Neighbor> out(nq * k);
            {
              py::gil_scoped_release release;
              self.search_batch(arr.data(), nq, k, ef.value_or(0), out.data(), num_threads, f.get());
            }
            py::array_t<int64_t> ids(
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
            py::array_t<float> dists(
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
            auto* ip = ids.mutable_data();
            auto* dp = dists.mutable_data();
            for (std::size_t i = 0; i < nq * k; ++i) {
              ip[i] = static_cast<int64_t>(out[i].label);  // kInvalidLabel -> -1
              dp[i] = out[i].distance;
            }
            if (single) return py::make_tuple(ids.attr("reshape")(k), dists.attr("reshape")(k));
            return py::make_tuple(ids, dists);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = py::none(), py::arg("num_threads") = 0,
          py::arg("filter") = py::none())
      .def("set_ef", &Index::set_ef, py::arg("ef"))
      .def(
          "mark_deleted",
          [](Index& self, int64_t label) {
            if (label < 0) throw py::value_error("labels are non-negative");
            self.mark_deleted(static_cast<uint64_t>(label));
          },
          py::arg("label"))
      .def(
          "is_deleted",
          [](const Index& self, int64_t label) {
            return label >= 0 && self.is_deleted(static_cast<uint64_t>(label));
          },
          py::arg("label"))
      .def("__contains__",
           [](const Index& self, int64_t label) {
             return label >= 0 && self.contains(static_cast<uint64_t>(label));
           })
      .def(
          "get_vector",
          [](const Index& self, int64_t label) {
            if (label < 0) throw py::value_error("labels are non-negative");
            const auto v = self.get_vector(static_cast<uint64_t>(label));
            py::array_t<float> a(static_cast<py::ssize_t>(v.size()));
            std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(float));
            return a;
          },
          py::arg("label"))
      .def("__len__", &Index::size)
      .def(
          "save",
          [](const Index& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
          },
          py::arg("path"))
      .def_static(
          "load",
          [](const std::string& path, bool mmap) {
            py::gil_scoped_release release;
            return std::make_unique<Index>(Index::load(path, mmap));
          },
          py::arg("path"), py::arg("mmap") = false)
      .def_property_readonly("dim", &Index::dim)
      .def_property_readonly("metric",
                             [](const Index& self) { return std::string(hnsw::metric_name(self.metric())); })
      .def_property_readonly("M", [](const Index& self) { return self.params().M; })
      .def_property_readonly("ef_construction",
                             [](const Index& self) { return self.params().ef_construction; })
      .def_property_readonly("max_elements", [](const Index& self) { return self.params().max_elements; })
      .def_property("ef", &Index::ef, &Index::set_ef)
      .def_property_readonly("read_only", &Index::read_only)
      .def_property_readonly("max_level", &Index::max_level)
      .def_property_readonly("memory_usage", &Index::memory_usage)
      .def("__repr__", [](const Index& self) {
        return "<hnsw_engine.Index dim=" + std::to_string(self.dim()) +
               " metric=" + std::string(hnsw::metric_name(self.metric())) +
               " M=" + std::to_string(self.params().M) + " size=" + std::to_string(self.size()) +
               (self.read_only() ? " read_only" : "") + ">";
      });
}
