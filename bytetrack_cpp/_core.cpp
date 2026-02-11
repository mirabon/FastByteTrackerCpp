// bytetrack_cpp/_core.cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <vector>
#include <string>
#include <stdexcept>

#include "BYTETracker.h"

static PyObject* py_str(const char* s) { return PyUnicode_FromString(s); }

typedef struct {
    PyObject_HEAD
    BYTETracker* tracker;
} PyTracker;

static void PyTracker_dealloc(PyTracker* self) {
    delete self->tracker;
    self->tracker = nullptr;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTracker_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    (void)args; (void)kwds;
    PyTracker* self = (PyTracker*)type->tp_alloc(type, 0);
    if (!self) return nullptr;
    self->tracker = nullptr;
    return (PyObject*)self;
}

static int PyTracker_init(PyTracker* self, PyObject* args, PyObject* kwds) {
    int frame_rate = 30;
    int track_buffer = 30;

    static const char* kwlist[] = {"frame_rate", "track_buffer", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", (char**)kwlist, &frame_rate, &track_buffer)) {
        return -1;
    }

    try {
        self->tracker = new BYTETracker(frame_rate, track_buffer);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create BYTETracker");
        return -1;
    }
    return 0;
}

static bool parse_detection(PyObject* item, float& x, float& y, float& w, float& h, int& label, float& score) {
    // Accept (x,y,w,h,score) or (x,y,w,h,label,score)
    if (!PySequence_Check(item)) return false;
    PyObject* seq = PySequence_Fast(item, "detection must be a sequence");
    if (!seq) return false;

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n != 5 && n != 6) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "each detection must have length 5 or 6");
        return false;
    }

    PyObject** elems = PySequence_Fast_ITEMS(seq);

    auto to_float = [](PyObject* o, float& out) -> bool {
        PyObject* f = PyNumber_Float(o);
        if (!f) return false;
        out = (float)PyFloat_AsDouble(f);
        Py_DECREF(f);
        return !PyErr_Occurred();
    };

    auto to_int = [](PyObject* o, int& out) -> bool {
        PyObject* i = PyNumber_Long(o);
        if (!i) return false;
        out = (int)PyLong_AsLong(i);
        Py_DECREF(i);
        return !PyErr_Occurred();
    };

    if (!to_float(elems[0], x) || !to_float(elems[1], y) || !to_float(elems[2], w) || !to_float(elems[3], h)) {
        Py_DECREF(seq);
        return false;
    }

    if (n == 5) {
        label = 0;
        if (!to_float(elems[4], score)) {
            Py_DECREF(seq);
            return false;
        }
    } else {
        if (!to_int(elems[4], label) || !to_float(elems[5], score)) {
            Py_DECREF(seq);
            return false;
        }
    }

    Py_DECREF(seq);
    return true;
}

static PyObject* PyTracker_update(PyTracker* self, PyObject* args) {
    if (!self->tracker) {
        PyErr_SetString(PyExc_RuntimeError, "Tracker not initialized");
        return nullptr;
    }

    PyObject* detections_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &detections_obj)) return nullptr;

    PyObject* seq = PySequence_Fast(detections_obj, "detections must be an iterable");
    if (!seq) return nullptr;

    std::vector<Object> objects;
    objects.reserve((size_t)PySequence_Fast_GET_SIZE(seq));

    PyObject** items = PySequence_Fast_ITEMS(seq);
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);

    for (Py_ssize_t i = 0; i < n; ++i) {
        float x, y, w, h, score;
        int label;
        if (!parse_detection(items[i], x, y, w, h, label, score)) {
            Py_DECREF(seq);
            return nullptr;  // error already set
        }

        Object obj;
        obj.rect = cv::Rect_<float>(x, y, w, h);
        obj.label = label;
        obj.prob = score;
        objects.push_back(obj);
    }

    Py_DECREF(seq);

    std::vector<STrack> tracks;
    try {
        tracks = self->tracker->update(objects);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "BYTETracker::update failed");
        return nullptr;
    }

    PyObject* out_list = PyList_New((Py_ssize_t)tracks.size());
    if (!out_list) return nullptr;

    for (Py_ssize_t i = 0; i < (Py_ssize_t)tracks.size(); ++i) {
        const STrack& tr = tracks[(size_t)i];

        PyObject* d = PyDict_New();
        if (!d) { Py_DECREF(out_list); return nullptr; }

        PyObject* tlwh = PyTuple_Pack(
            4,
            PyFloat_FromDouble(tr.tlwh.t),
            PyFloat_FromDouble(tr.tlwh.l),
            PyFloat_FromDouble(tr.tlwh.w),
            PyFloat_FromDouble(tr.tlwh.h)
        );
        PyObject* tlbr = PyTuple_Pack(
            4,
            PyFloat_FromDouble(tr.tlbr.l), // x1
            PyFloat_FromDouble(tr.tlbr.t), // y1
            PyFloat_FromDouble(tr.tlbr.r), // x2
            PyFloat_FromDouble(tr.tlbr.b)  // y2
        );

        if (!tlwh || !tlbr) {
            Py_XDECREF(tlwh);
            Py_XDECREF(tlbr);
            Py_DECREF(d);
            Py_DECREF(out_list);
            return nullptr;
        }

        PyDict_SetItem(d, py_str("track_id"), PyLong_FromLong(tr.track_id));
        PyDict_SetItem(d, py_str("label"), PyLong_FromLong(tr.obj_id));
        PyDict_SetItem(d, py_str("score"), PyFloat_FromDouble(tr.score));
        PyDict_SetItem(d, py_str("frame_id"), PyLong_FromLong(tr.frame_id));
        PyDict_SetItem(d, py_str("state"), PyLong_FromLong(tr.state));
        PyDict_SetItem(d, py_str("is_activated"), tr.is_activated ? Py_True : Py_False);
        PyDict_SetItem(d, py_str("tlwh"), tlwh);
        PyDict_SetItem(d, py_str("tlbr"), tlbr);

        Py_DECREF(tlwh);
        Py_DECREF(tlbr);

        PyList_SET_ITEM(out_list, i, d);  // steals ref
    }

    return out_list;
}

static PyMethodDef PyTracker_methods[] = {
    {"update", (PyCFunction)PyTracker_update, METH_VARARGS, PyDoc_STR("update(detections) -> list[dict]")},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject PyTrackerType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "bytetrack_cpp.Tracker",
    .tp_basicsize = sizeof(PyTracker),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyTracker_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("BYTETrack tracker (C++ extension)."),
    .tp_methods = PyTracker_methods,
    .tp_init = (initproc)PyTracker_init,
    .tp_new = PyTracker_new,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "bytetrack_cpp._core",
    "C++ extension for BYTETrack.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit__core(void) {
    if (PyType_Ready(&PyTrackerType) < 0) return nullptr;

    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return nullptr;

    Py_INCREF(&PyTrackerType);
    if (PyModule_AddObject(m, "Tracker", (PyObject*)&PyTrackerType) < 0) {
        Py_DECREF(&PyTrackerType);
        Py_DECREF(m);
        return nullptr;
    }

    PyModule_AddStringConstant(m, "__version__", "0.1.1");
    return m;
}
