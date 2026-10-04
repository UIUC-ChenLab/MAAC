#include "pipeline.hpp"

#include <ATen/dlpack.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
namespace {
struct TensorLease {
    DLManagedTensor tensor{};
    std::shared_ptr<e2e::TrainingStorage> storage;
    std::int64_t length = 0;
};
void delete_tensor(DLManagedTensor* managed) noexcept {
    // No direct hipFree here. The shared owner remains live while either the
    // pipeline or any capsule/tensor still references any of its allocations.
    delete static_cast<TensorLease*>(managed->manager_ctx);
}
py::capsule capsule(const e2e::TrainingView& view) {
    auto lease = std::make_unique<TensorLease>();
    lease->storage = view.owner;
    lease->length = view.length;
    auto& tensor = lease->tensor;
    tensor.manager_ctx = lease.get();
    tensor.deleter = delete_tensor;
    tensor.dl_tensor.data = view.pointer;
    tensor.dl_tensor.device = {kDLROCM, view.device};
    tensor.dl_tensor.ndim = 1;
    tensor.dl_tensor.dtype = {kDLInt, 32, 1};
    tensor.dl_tensor.shape = &lease->length;
    tensor.dl_tensor.strides = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    py::capsule result(&tensor, "dltensor", [](PyObject* object) {
        // PyTorch renames consumed capsules to used_dltensor and assumes the
        // managed-tensor deleter. Unconsumed capsules are cleaned up here.
        if (PyCapsule_IsValid(object, "dltensor")) {
            auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(object, "dltensor"));
            if (tensor && tensor->deleter) tensor->deleter(tensor);
        }
    });
    lease.release();
    return result;
}
// One iteration, as seen by training.py.
py::dict batch_dict(const e2e::TrainingBatch& batch) {
    py::dict out;
    out["slot"] = batch.slot;                          // which set of GPU buffers holds the data
    out["generation"] = batch.generation;
    out["group"] = batch.group;
    out["rotation"] = batch.rotation;                  // SLR s sampled mini-batch (s + rotation) % 3
    out["counts"] = batch.counts;                      // per SLR: valid samples, rows, cols, edges
    out["sampling_slr_ms"] = batch.sampling.slr_ms;    // each SLR: start -> done
    out["sampling_all_ms"] = batch.sampling.all_slrs_ms;
    out["p2p_user_ms"] = batch.p2p.wall_ms;            // the twelve FPGA -> GPU transfers together
    return out;
}
py::dict info_dict(const e2e::PipelineInfo& info) {
    py::dict out;
    out["dataset"] = info.dataset;
    out["slot_count"] = info.slot_count;
    out["num_nodes"] = info.num_nodes;
    out["num_edges"] = info.num_edges;
    out["fanouts"] = info.fanouts;
    out["minibatch_seeds"] = info.minibatch_seeds;
    out["skipped_seeds"] = info.skipped_seeds;
    out["minibatch_counts"] = info.minibatch_counts;
    out["nvme_load_ms"] = info.nvme.load_ms;           // NVMe -> FPGA graph loading
    return out;
}
} // namespace

PYBIND11_MODULE(_fpga_pipeline, module) {
    module.doc() = "NVMe -> FPGA neighbor sampling -> P2P -> GPU: the producer side of the MAAC pipeline";
    py::class_<e2e::Pipeline>(module, "Pipeline")
        .def(py::init([](const std::string& graph_root, const std::string& dataset, bool diagnostic) {
            py::gil_scoped_release unlocked;
            return std::make_unique<e2e::Pipeline>(graph_root, dataset, 0, diagnostic);   // GPU 0
        }), py::arg("graph_root"), py::arg("dataset"), py::arg("diagnostic") = false)
        .def("info", [](const e2e::Pipeline& pipeline) { return info_dict(pipeline.info()); })
        .def("view", [](const e2e::Pipeline& pipeline, unsigned slot, unsigned slr, unsigned kind) {
            return capsule(pipeline.view(slot, slr, kind));
        }, py::arg("slot"), py::arg("slr"), py::arg("kind"))
        .def("reference", [](const e2e::Pipeline& pipeline, unsigned minibatch) {
            e2e::Outputs values;
            { py::gil_scoped_release unlocked; values = pipeline.reference(minibatch); }
            py::tuple result(4);
            for (unsigned kind = 0; kind < 4; ++kind) {
                py::array_t<std::int32_t> array(values[kind].size());
                if (!values[kind].empty()) std::memcpy(array.mutable_data(), values[kind].data(),
                                                      values[kind].size() * sizeof(std::int32_t));
                result[kind] = std::move(array);
            }
            return result;
        }, py::arg("minibatch"))
        .def("start", &e2e::Pipeline::start, py::arg("groups"), py::call_guard<py::gil_scoped_release>())
        .def("acquire", [](e2e::Pipeline& pipeline) {
            e2e::TrainingBatch batch;
            { py::gil_scoped_release unlocked; batch = pipeline.acquire(); }
            return batch_dict(batch);
        })
        .def("begin_consume", &e2e::Pipeline::begin_consume, py::arg("slot"),
             py::arg("generation"), py::arg("stream"), py::call_guard<py::gil_scoped_release>())
        .def("release", &e2e::Pipeline::release, py::arg("slot"), py::arg("generation"),
             py::arg("stream"), py::call_guard<py::gil_scoped_release>())
        .def("finish", &e2e::Pipeline::finish, py::call_guard<py::gil_scoped_release>())
        .def("close", &e2e::Pipeline::close, py::call_guard<py::gil_scoped_release>())
        .def("__enter__", [](e2e::Pipeline& pipeline) -> e2e::Pipeline& { return pipeline; },
             py::return_value_policy::reference_internal)
        .def("__exit__", [](e2e::Pipeline& pipeline, py::object, py::object, py::object) {
            py::gil_scoped_release unlocked;
            pipeline.close();
        });
}
