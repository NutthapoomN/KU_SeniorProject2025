/* Include files */

#include "modelInterface.h"
#include "m_NGiGOBVa6l8QfX7tNnMjx.h"
#include "mwstringutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */
static void cgxe_mdl_start(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance);
static void cgxe_mdl_initialize(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance);
static void cgxe_mdl_outputs(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance);
static void cgxe_mdl_update(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance);
static void cgxe_mdl_derivative(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance);
static void cgxe_mdl_enable(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance);
static void cgxe_mdl_disable(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance);
static void cgxe_mdl_terminate(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance);
static void CheckPythonError(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *pyObjsToRelease[], int32_T numObjToRelease);
static real_T PyObj_marshalIn(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *pyToMarshal, PyObject *pyOwner);
static PyObject *getPyNamespaceDict(void);
static void assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  PyObject *dict, char_T *key, real_T val);
static void execPyScript(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  char_T *script, PyObject *ns);
static PyObject *getPyDictVal(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key);
static void b_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void c_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void d_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void e_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void f_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void g_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void b_execPyScript(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  char_T *script, PyObject *ns);
static PyObject *b_getPyDictVal(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key);
static int32_T deleteDictItem(PyObject *dict, char_T *key);
static int32_T b_deleteDictItem(PyObject *dict, char_T *key);
static int32_T c_deleteDictItem(PyObject *dict, char_T *key);
static int32_T d_deleteDictItem(PyObject *dict, char_T *key);
static int32_T e_deleteDictItem(PyObject *dict, char_T *key);
static void c_execPyScript(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  char_T *script, PyObject *ns);
static void init_simulink_io_address(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance);

/* Function Definitions */
static void cgxe_mdl_start(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance)
{
  init_simulink_io_address(moduleInstance);
  cgxertSetSimStateCompliance(moduleInstance->S, 2);
}

static void cgxe_mdl_initialize(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance)
{
  PyObject *r;
  cgxertInitMLPythonIFace();
  moduleInstance->GIL = PyGILState_Ensure();
  moduleInstance->namespaceDict = getPyNamespaceDict();
  assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "output", 0.0);
  execPyScript(moduleInstance, "", moduleInstance->namespaceDict);
  r = getPyDictVal(moduleInstance, moduleInstance->namespaceDict, "output");
  *moduleInstance->b_y0 = PyObj_marshalIn(moduleInstance, r, NULL);
  Py_DecRef(r);
  PyGILState_Release(moduleInstance->GIL);
}

static void cgxe_mdl_outputs(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance)
{
  PyObject *r;
  moduleInstance->GIL = PyGILState_Ensure();
  b_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "day",
                   *moduleInstance->u0);
  c_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "hour",
                   *moduleInstance->u1);
  d_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "output",
                   *moduleInstance->b_y0);
  e_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "people",
                   *moduleInstance->u2);
  f_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "t_room",
                   *moduleInstance->u3);
  g_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "t_tes",
                   *moduleInstance->u4);
  b_execPyScript(moduleInstance,
                 "import numpy as np\nimport skfuzzy as fuzz\nfrom skfuzzy import control as ctrl\n\n# ==========================================="
                 "===============\n# Persistent Fuzzy System (\xe0\xb8\xaa\xe0\xb8\xa3\xe0\xb9\x89\xe0\xb8\xb2\xe0\xb8\x87\xe0\xb8\x84\xe0\xb8\xa3"
                 "\xe0\xb8\xb1\xe0\xb9\x89\xe0\xb8\x87\xe0\xb9\x80\xe0\xb8\x94\xe0\xb8\xb5\xe0\xb8\xa2\xe0\xb8\xa7)\n# ==========================="
                 "===============================\ndef create_fuzzy_system():\n\n    # -------- INPUTS --------\n    people = ctrl.Antecedent(np.a"
                 "range(1, 32, 1), \'people\')\n    room_temp = ctrl.Antecedent(np.arange(20, 31, 1), \'room_temp\')\n\n    # -------- OUTPUTS ---"
                 "-----\n    cond_use = ctrl.Consequent(np.arange(0, 1.1, 0.1), \'cond_use\')\n    comp_use = ctrl.Consequent(np.arange(0, 1.1, 0."
                 "1), \'comp_use\')\n    pump_speed = ctrl.Consequent(np.arange(0, 101, 1), \'pump_speed\')\n    fan_speed = ctrl.Consequent(np.ar"
                 "ange(0, 101, 1), \'fan_speed\')\n    valve_2 = ctrl.Consequent(np.arange(0, 1.1, 0.1), \'valve_2\')\n    valve_3 = ctrl.Conseque"
                 "nt(np.arange(0, 1.1, 0.1), \'valve_3\')\n\n    # -------- MEMBERSHIP FUNCTIONS --------\n    people[\'few\'] = fuzz.trapmf(peopl"
                 "e.universe, [0, 0, 4, 6])\n    people[\'medium\'] = fuzz.trapmf(people.universe, [5, 10, 15, 15])\n    people[\'many\'] = fuzz.t"
                 "rapmf(people.universe, [14, 25, 31, 31])\n\n    room_temp[\'normal\'] = fuzz.trimf(room_temp.universe, [20, 20, 26])\n    room_t"
                 "emp[\'hot\'] = fuzz.trimf(room_temp.universe, [24, 27, 29])\n    room_temp[\'very_hot\'] = fuzz.trapmf(room_temp.universe, [28, "
                 "29, 30, 30])\n\n    for output in [cond_use, comp_use, valve_2, valve_3]:\n        output[\'off\'] = fuzz.trimf(output.universe,"
                 " [0, 0, 0.5])\n        output[\'on\']  = fuzz.trimf(output.universe, [0.5, 1, 1])\n\n    pump_speed[\'low\'] = fuzz.trimf(pump_s"
                 "peed.universe, [0, 30, 40])\n    pump_speed[\'high\'] = fuzz.trimf(pump_speed.universe, [30, 100, 100])\n\n    fan_speed[\'low\'"
                 "] = fuzz.trimf(fan_speed.universe, [0, 30, 40])\n    fan_speed[\'high\'] = fuzz.trimf(fan_speed.universe, [30, 100, 100])\n\n   "
                 " # -------- RULES --------\n    rules = [\n\n        ctrl.Rule(people[\'few\'] & room_temp[\'normal\'],\n                  [cond"
                 "_use[\'off\'], comp_use[\'off\'], pump_speed[\'low\'],\n                   fan_speed[\'low\'], valve_2[\'off\'], valve_3[\'off\'"
                 "]]),\n\n        ctrl.Rule(people[\'medium\'] & room_temp[\'hot\'],\n                  [cond_use[\'on\'], comp_use[\'on\'], pump_"
                 "speed[\'low\'],\n                   fan_speed[\'high\'], valve_2[\'on\'], valve_3[\'off\']]),\n\n        ctrl.Rule(people[\'many"
                 "\'] & room_temp[\'very_hot\'],\n                  [cond_use[\'on\'], comp_use[\'on\'], pump_speed[\'high\'],\n                  "
                 " fan_speed[\'high\'], valve_2[\'on\'], valve_3[\'off\']])\n    ]\n\n    system = ctrl.ControlSystem(rules)\n    return ctrl.Cont"
                 "rolSystemSimulation(system)\n\n\n# ==========================================================\n# Controller (Time-Step Function)"
                 "\n# ==========================================================\nfuzzy_sim = create_fuzzy_system()\n\ndef tes_controller(day, hou"
                 "r, people, t_room, t_tes):\n    \"\"\"\n    Time-step controller (\xe0\xb9\x80\xe0\xb8\xa3\xe0\xb8\xb5\xe0\xb8\xa2\xe0\xb8\x81\xe0"
                 "\xb8\x84\xe0\xb8\xa3\xe0\xb8\xb1\xe0\xb9\x89\xe0\xb8\x87\xe0\xb8\xa5\xe0\xb8\xb0 1 timestep)\n\n    Inputs:\n        day     : 0"
                 "-6\n        hour    : 0-23\n        people  : \xe0\xb8\x88\xe0\xb8\xb3\xe0\xb8\x99\xe0\xb8\xa7\xe0\xb8\x99\xe0\xb8\x84\xe0\xb8\x99"
                 "\n        t_room  : \xe0\xb8\xad\xe0\xb8\xb8\xe0\xb8\x93\xe0\xb8\xab\xe0\xb8\xa0\xe0\xb8\xb9\xe0\xb8\xa1\xe0\xb8\xb4\xe0\xb8\xab"
                 "\xe0\xb9\x89\xe0\xb8\xad\xe0\xb8\x87 (\xc2\xb0"
                 "C)\n        t_tes   : \xe0\xb8\xad\xe0\xb8\xb8\xe0\xb8\x93\xe0\xb8\xab\xe0\xb8\xa0"
                 "\xe0\xb8\xb9\xe0\xb8\xa1\xe0\xb8\xb4\xe0\xb8\x96\xe0\xb8\xb1\xe0\xb8\x87 (\xc2\xb0"
                 "C)\n\n    Returns:\n        cond, v2, v3, p"
                 "ump, fan, comp\n    \"\"\"\n\n    # Default Output\n    cond = 0\n    v2 = 0\n    v3 = 0\n    pump = 0\n    fan = 0\n    comp = "
                 "0\n\n    # ==============================\n    # MODE 1 : Night Charging\n    # ==============================\n    if hour >= 2"
                 "2 or hour < 9:\n\n        if people < 1:\n            if t_tes > 5:\n                cond = 1\n                v3 = 1\n         "
                 "       comp = 1\n        else:\n            if t_tes > 5 and t_room > 24:\n                cond = 1\n                v2 = 1\n   "
                 "             v3 = 1\n                fan = 1\n                comp = 1\n            elif t_tes > 5:\n                cond = 1\n "
                 "               v3 = 1\n\n    # ==============================\n    # MODE 2 : Day Operation\n    # ============================="
                 "=\n    else:\n\n        if t_room > 24:\n\n            fuzzy_sim.input[\'people\'] = people\n            fuzzy_sim.input[\'room_"
                 "temp\'] = t_room\n            fuzzy_sim.compute()\n\n            cond = int(fuzzy_sim.output[\'cond_use\'] >= 0.5)\n            "
                 "comp = int(fuzzy_sim.output[\'comp_use\'] >= 0.5)\n            v2   = int(fuzzy_sim.output[\'valve_2\'] >= 0.5)\n            v3 "
                 "  = int(fuzzy_sim.output[\'valve_3\'] >= 0.5)\n\n            pump = round(fuzzy_sim.output[\'pump_speed\'] / 100, 2)\n          "
                 "  fan  = round(fuzzy_sim.output[\'fan_speed\'] / 100, 2)\n\n    return np.array([cond, v2, v3, pump, fan, comp])\n\noutput = tes"
                 "_controller(1, 14, 12, 28, 7)", moduleInstance->namespaceDict);
  r = b_getPyDictVal(moduleInstance, moduleInstance->namespaceDict, "output");
  *moduleInstance->b_y0 = PyObj_marshalIn(moduleInstance, r, NULL);
  Py_DecRef(r);
  PyGILState_Release(moduleInstance->GIL);
}

static void cgxe_mdl_update(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_derivative(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_enable(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_disable(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_terminate(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance)
{
  moduleInstance->GIL = PyGILState_Ensure();
  deleteDictItem(moduleInstance->namespaceDict, "day");
  b_deleteDictItem(moduleInstance->namespaceDict, "hour");
  c_deleteDictItem(moduleInstance->namespaceDict, "people");
  d_deleteDictItem(moduleInstance->namespaceDict, "t_room");
  e_deleteDictItem(moduleInstance->namespaceDict, "t_tes");
  c_execPyScript(moduleInstance, "", moduleInstance->namespaceDict);
  Py_DecRef(moduleInstance->namespaceDict);
  PyGILState_Release(moduleInstance->GIL);
}

static void CheckPythonError(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *pyObjsToRelease[], int32_T numObjToRelease)
{
  PyObject *pMsg;
  PyObject *pTraceback = NULL;
  PyObject *pType = NULL;
  PyObject *pValue = NULL;
  PyObject *sep = NULL;
  PyObject *tracebackList = NULL;
  PyObject *tracebackModule = NULL;
  int32_T i;
  int32_T idx;
  char_T *cMsg;
  void *slString;
  i = suStringStackSize();
  PyErr_Fetch(&pType, &pValue, &pTraceback);
  PyErr_NormalizeException(&pType, &pValue, &pTraceback);
  if (pType != NULL) {
    if (pTraceback != NULL) {
      tracebackModule = PyImport_ImportModule("traceback");
      tracebackList = PyObject_CallMethod(tracebackModule, "format_exception",
        "OOO", pType, pValue, pTraceback);
      sep = PyUnicode_FromString("");
      pMsg = PyUnicode_Join(sep, tracebackList);
    } else if (pValue != NULL) {
      pMsg = PyObject_Str(pValue);
    } else {
      pMsg = PyObject_Str(pType);
    }

    cMsg = (char_T *)PyUnicode_AsUTF8(pMsg);
    if (cMsg == NULL) {
      cMsg =
        "Simulink encountered an error when converting a python error message to UTF-8";
      PyErr_Clear();
    } else {
      slString = suAddStackString(cMsg);
      cMsg = suToCStr(slString);
    }

    if (sep != NULL) {
      Py_DecRef(sep);
    }

    if (tracebackList != NULL) {
      Py_DecRef(tracebackList);
    }

    if (tracebackModule != NULL) {
      Py_DecRef(tracebackModule);
    }

    if (pMsg != NULL) {
      Py_DecRef(pMsg);
    }

    pMsg = pType;
    if (pMsg != NULL) {
      Py_DecRef(pMsg);
    }

    pMsg = pValue;
    if (pMsg != NULL) {
      Py_DecRef(pMsg);
    }

    pMsg = pTraceback;
    if (pMsg != NULL) {
      Py_DecRef(pMsg);
    }

    for (idx = 0; idx < numObjToRelease; idx++) {
      pMsg = pyObjsToRelease[idx];
      if (pMsg != NULL) {
        Py_DecRef(pMsg);
      }
    }

    PyGILState_Release(moduleInstance->GIL);
    cgxertReportError(moduleInstance->S, -1, -1,
                      "Simulink:CustomCode:PythonRuntimeError", 3, 1, strlen
                      (cMsg), cMsg);
  }

  suMoveReturnedStringsToTopOfCallerStack(i, 0);
}

static real_T PyObj_marshalIn(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *pyToMarshal, PyObject *pyOwner)
{
  PyObject *pyObjArray[1];
  PyObject *objToRelease;
  real_T outputVal;
  outputVal = PyFloat_AsDouble(pyToMarshal);
  if (pyOwner == NULL) {
    objToRelease = pyToMarshal;
  } else {
    objToRelease = pyOwner;
  }

  pyObjArray[0U] = objToRelease;
  CheckPythonError(moduleInstance, pyObjArray, 1);
  return outputVal;
}

static PyObject *getPyNamespaceDict(void)
{
  return PyDict_Copy(PyModule_GetDict(PyImport_AddModule("__main__")));
}

static void assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void execPyScript(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  char_T *script, PyObject *ns)
{
  PyObject *pyObjArray[2];
  PyObject *codeObject;
  PyObject *originalNamespace;
  PyObject *unusedEvalResult;
  Py_ssize_t i;
  Py_ssize_t numKeysInModifiedNs;
  if (ns != NULL) {
    codeObject = Py_CompileString(script, "Python Code Block", 257);
    CheckPythonError(moduleInstance, NULL, 0);
    originalNamespace = PyDict_Copy(ns);
    unusedEvalResult = PyEval_EvalCode(codeObject, ns, ns);
    pyObjArray[0U] = codeObject;
    pyObjArray[1U] = unusedEvalResult;
    CheckPythonError(moduleInstance, pyObjArray, 2);
    Py_DecRef(codeObject);
    if (unusedEvalResult != NULL) {
      Py_DecRef(unusedEvalResult);
    }

    codeObject = PyDict_Keys(ns);
    numKeysInModifiedNs = PyList_Size(codeObject);
    for (i = 0; i < numKeysInModifiedNs; i++) {
      unusedEvalResult = PySequence_GetItem(codeObject, i);
      CheckPythonError(moduleInstance, NULL, 0);
      if ((PyDict_Contains(originalNamespace, unusedEvalResult) == 0) &&
          (!PyModule_Check(PyDict_GetItem(ns, unusedEvalResult)))) {
        PyDict_DelItem(ns, unusedEvalResult);
      }

      Py_DecRef(unusedEvalResult);
    }

    Py_DecRef(codeObject);
    Py_DecRef(originalNamespace);
  }
}

static PyObject *getPyDictVal(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key)
{
  PyObject *b_value;
  b_value = PyDict_GetItemString(dict, key);
  CheckPythonError(moduleInstance, NULL, 0);
  Py_IncRef(b_value);
  return b_value;
}

static void b_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void c_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void d_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void e_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void f_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void g_assignToPyDict(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key, real_T val)
{
  PyObject *pyObj;
  if (dict != NULL) {
    pyObj = PyFloat_FromDouble(val);
    CheckPythonError(moduleInstance, NULL, 0);
    PyDict_SetItemString(dict, key, pyObj);
    Py_DecRef(pyObj);
  }
}

static void b_execPyScript(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  char_T *script, PyObject *ns)
{
  PyObject *pyObjArray[2];
  PyObject *codeObject;
  PyObject *originalNamespace;
  PyObject *unusedEvalResult;
  Py_ssize_t i;
  Py_ssize_t numKeysInModifiedNs;
  if (ns != NULL) {
    codeObject = Py_CompileString(script, "Python Code Block", 257);
    CheckPythonError(moduleInstance, NULL, 0);
    originalNamespace = PyDict_Copy(ns);
    unusedEvalResult = PyEval_EvalCode(codeObject, ns, ns);
    pyObjArray[0U] = codeObject;
    pyObjArray[1U] = unusedEvalResult;
    CheckPythonError(moduleInstance, pyObjArray, 2);
    Py_DecRef(codeObject);
    if (unusedEvalResult != NULL) {
      Py_DecRef(unusedEvalResult);
    }

    codeObject = PyDict_Keys(ns);
    numKeysInModifiedNs = PyList_Size(codeObject);
    for (i = 0; i < numKeysInModifiedNs; i++) {
      unusedEvalResult = PySequence_GetItem(codeObject, i);
      CheckPythonError(moduleInstance, NULL, 0);
      if ((PyDict_Contains(originalNamespace, unusedEvalResult) == 0) &&
          (!PyModule_Check(PyDict_GetItem(ns, unusedEvalResult)))) {
        PyDict_DelItem(ns, unusedEvalResult);
      }

      Py_DecRef(unusedEvalResult);
    }

    Py_DecRef(codeObject);
    Py_DecRef(originalNamespace);
  }
}

static PyObject *b_getPyDictVal(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance, PyObject *dict, char_T *key)
{
  PyObject *b_value;
  b_value = PyDict_GetItemString(dict, key);
  CheckPythonError(moduleInstance, NULL, 0);
  Py_IncRef(b_value);
  return b_value;
}

static int32_T deleteDictItem(PyObject *dict, char_T *key)
{
  if (dict != NULL) {
    PyDict_DelItemString(dict, key);
    PyErr_Clear();
  }

  return 0;
}

static int32_T b_deleteDictItem(PyObject *dict, char_T *key)
{
  if (dict != NULL) {
    PyDict_DelItemString(dict, key);
    PyErr_Clear();
  }

  return 0;
}

static int32_T c_deleteDictItem(PyObject *dict, char_T *key)
{
  if (dict != NULL) {
    PyDict_DelItemString(dict, key);
    PyErr_Clear();
  }

  return 0;
}

static int32_T d_deleteDictItem(PyObject *dict, char_T *key)
{
  if (dict != NULL) {
    PyDict_DelItemString(dict, key);
    PyErr_Clear();
  }

  return 0;
}

static int32_T e_deleteDictItem(PyObject *dict, char_T *key)
{
  if (dict != NULL) {
    PyDict_DelItemString(dict, key);
    PyErr_Clear();
  }

  return 0;
}

static void c_execPyScript(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance,
  char_T *script, PyObject *ns)
{
  PyObject *pyObjArray[2];
  PyObject *codeObject;
  PyObject *originalNamespace;
  PyObject *unusedEvalResult;
  if (ns != NULL) {
    codeObject = Py_CompileString(script, "Python Code Block", 257);
    CheckPythonError(moduleInstance, NULL, 0);
    originalNamespace = PyDict_Copy(ns);
    unusedEvalResult = PyEval_EvalCode(codeObject, ns, ns);
    pyObjArray[0U] = codeObject;
    pyObjArray[1U] = unusedEvalResult;
    CheckPythonError(moduleInstance, pyObjArray, 2);
    Py_DecRef(codeObject);
    if (unusedEvalResult != NULL) {
      Py_DecRef(unusedEvalResult);
    }

    Py_DecRef(originalNamespace);
  }
}

static void init_simulink_io_address(InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
  *moduleInstance)
{
  moduleInstance->emlrtRootTLSGlobal = (void *)cgxertGetEMLRTCtx
    (moduleInstance->S);
  moduleInstance->u0 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 0);
  moduleInstance->u1 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 1);
  moduleInstance->u2 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 2);
  moduleInstance->u3 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 3);
  moduleInstance->u4 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 4);
  moduleInstance->b_y0 = (real_T *)cgxertGetOutputPortSignal(moduleInstance->S,
    0);
}

/* CGXE Glue Code */
static void mdlOutputs_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S, int_T tid)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_outputs(moduleInstance);
}

static void mdlInitialize_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_initialize(moduleInstance);
}

static void mdlUpdate_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S, int_T tid)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_update(moduleInstance);
}

static void mdlDerivatives_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_derivative(moduleInstance);
}

static void mdlTerminate_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_terminate(moduleInstance);
  free((void *)moduleInstance);
}

static void mdlEnable_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_enable(moduleInstance);
}

static void mdlDisable_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_disable(moduleInstance);
}

static void mdlStart_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
  InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *moduleInstance =
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx *)calloc(1, sizeof
    (InstanceStruct_NGiGOBVa6l8QfX7tNnMjx));
  moduleInstance->S = S;
  cgxertSetRuntimeInstance(S, (void *)moduleInstance);
  ssSetmdlOutputs(S, mdlOutputs_NGiGOBVa6l8QfX7tNnMjx);
  ssSetmdlInitializeConditions(S, mdlInitialize_NGiGOBVa6l8QfX7tNnMjx);
  ssSetmdlUpdate(S, mdlUpdate_NGiGOBVa6l8QfX7tNnMjx);
  ssSetmdlDerivatives(S, mdlDerivatives_NGiGOBVa6l8QfX7tNnMjx);
  ssSetmdlTerminate(S, mdlTerminate_NGiGOBVa6l8QfX7tNnMjx);
  ssSetmdlEnable(S, mdlEnable_NGiGOBVa6l8QfX7tNnMjx);
  ssSetmdlDisable(S, mdlDisable_NGiGOBVa6l8QfX7tNnMjx);
  cgxe_mdl_start(moduleInstance);

  {
    uint_T options = ssGetOptions(S);
    options |= SS_OPTION_RUNTIME_EXCEPTION_FREE_CODE;
    ssSetOptions(S, options);
  }
}

static void mdlProcessParameters_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S)
{
}

void method_dispatcher_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S, int_T method, void
  *data)
{
  switch (method) {
   case SS_CALL_MDL_START:
    mdlStart_NGiGOBVa6l8QfX7tNnMjx(S);
    break;

   case SS_CALL_MDL_PROCESS_PARAMETERS:
    mdlProcessParameters_NGiGOBVa6l8QfX7tNnMjx(S);
    break;

   default:
    /* Unhandled method */
    /*
       sf_mex_error_message("Stateflow Internal Error:\n"
       "Error calling method dispatcher for module: NGiGOBVa6l8QfX7tNnMjx.\n"
       "Can't handle method %d.\n", method);
     */
    break;
  }
}

mxArray *cgxe_NGiGOBVa6l8QfX7tNnMjx_BuildInfoUpdate(void)
{
  mxArray * mxBIArgs;
  mxArray * elem_1;
  mxArray * elem_2;
  mxArray * elem_3;
  double * pointer;
  mxBIArgs = mxCreateCellMatrix(1,3);
  elem_1 = mxCreateDoubleMatrix(0,0, mxREAL);
  pointer = mxGetPr(elem_1);
  mxSetCell(mxBIArgs,0,elem_1);
  elem_2 = mxCreateDoubleMatrix(0,0, mxREAL);
  pointer = mxGetPr(elem_2);
  mxSetCell(mxBIArgs,1,elem_2);
  elem_3 = mxCreateCellMatrix(1,0);
  mxSetCell(mxBIArgs,2,elem_3);
  return mxBIArgs;
}

mxArray *cgxe_NGiGOBVa6l8QfX7tNnMjx_fallback_info(void)
{
  const char* fallbackInfoFields[] = { "fallbackType", "incompatiableSymbol" };

  mxArray* fallbackInfoStruct = mxCreateStructMatrix(1, 1, 2, fallbackInfoFields);
  mxArray* fallbackType = mxCreateString("incompatibleFunction");
  mxArray* incompatibleSymbol = mxCreateString("PyModule_Check");
  mxSetFieldByNumber(fallbackInfoStruct, 0, 0, fallbackType);
  mxSetFieldByNumber(fallbackInfoStruct, 0, 1, incompatibleSymbol);
  return fallbackInfoStruct;
}
