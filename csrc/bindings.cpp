#include <Python.h>
#include <frameobject.h>
#include <internal/pycore_frame.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "dpe_api.h"
#include <iostream>

namespace py = pybind11;

static int dpe_mvs_py(const std::string& dense_folder,
                      int gpu_index,
                      bool fusion,
                      bool viz,
                      bool weak) {
    const bool no_fusion = !fusion;
    const bool no_viz    = !viz;
    const bool no_weak   = !weak;

    py::scoped_ostream_redirect out(std::cout);
    py::scoped_ostream_redirect err(std::cerr, py::module_::import("sys").attr("stderr"));

    int ret = run_dpe(dense_folder, gpu_index, no_fusion, no_viz, no_weak);
    if (ret != 0) throw std::runtime_error("DPE-MVS failed with code " + std::to_string(ret));
    return ret;
}

PYBIND11_MODULE(_dpe, m) {
    m.doc() = "Python-only bindings for DPE-MVS";
    m.def("dpe_mvs", &dpe_mvs_py,
          py::arg("dense_folder"),
          py::arg("gpu_index") = 0,
          py::kw_only(),
          py::arg("fusion") = true,
          py::arg("viz")    = false,
          py::arg("weak")   = false);
}
