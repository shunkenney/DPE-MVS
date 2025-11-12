#include <Python.h>
#include <frameobject.h>  // TODO: Check if this is really needed
#include <internal/pycore_frame.h>  // TODO: Check if this is really needed
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "main.h"
#include <iostream>

namespace py = pybind11;

static int dpe_mvs_py(const std::string& dense_folder,
                      int gpu_index,
                      bool verbose,
                      bool fusion,
                      bool viz,
                      bool depth,
                      bool normal,
                      bool weak,
                      bool edge) {

    py::scoped_ostream_redirect out(std::cout);
    py::scoped_ostream_redirect err(std::cerr, py::module_::import("sys").attr("stderr"));

    int ret = RunDPEPipeline(dense_folder, gpu_index, verbose, fusion, viz, depth, normal, weak, edge);
    if (ret != 0) throw std::runtime_error("DPE-MVS failed with code " + std::to_string(ret));
    return ret;
}

PYBIND11_MODULE(_dpe, m) {
    m.doc() = "Python-only bindings for DPE-MVS";
    m.def("dpe_mvs", &dpe_mvs_py,
          py::arg("dense_folder"),
          py::arg("gpu_index") = 0,
          py::arg("verbose") = true,
          py::arg("fusion") = false,
          py::arg("viz") = false,
          py::arg("depth") = true,
          py::arg("normal") = false,
          py::arg("weak") = false,
          py::arg("edge") = false);
}
