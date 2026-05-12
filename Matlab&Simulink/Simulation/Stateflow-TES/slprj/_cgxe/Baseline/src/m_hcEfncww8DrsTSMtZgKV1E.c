/* Include files */

#include "modelInterface.h"
#include "m_hcEfncww8DrsTSMtZgKV1E.h"
#include "mwstringutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */
static void cgxe_mdl_start(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance);
static void cgxe_mdl_initialize(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void cgxe_mdl_outputs(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void cgxe_mdl_update(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void cgxe_mdl_derivative(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void cgxe_mdl_enable(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void cgxe_mdl_disable(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void cgxe_mdl_terminate(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);
static void CheckPythonError(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *pyObjsToRelease[], int32_T numObjToRelease);
static real_T PyObj_marshalIn(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *pyToMarshal, PyObject *pyOwner);
static PyObject *getPyNamespaceDict(void);
static void assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
  PyObject *dict, char_T *key, real_T val);
static void execPyScript(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
  char_T *script, PyObject *ns);
static PyObject *getPyDictVal(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *dict, char_T *key);
static void b_assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void c_assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void d_assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *dict, char_T *key, real_T val);
static void b_execPyScript(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
  char_T *script, PyObject *ns);
static PyObject *b_getPyDictVal(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *dict, char_T *key);
static int32_T deleteDictItem(PyObject *dict, char_T *key);
static int32_T b_deleteDictItem(PyObject *dict, char_T *key);
static void c_execPyScript(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
  char_T *script, PyObject *ns);
static void init_simulink_io_address(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance);

/* Function Definitions */
static void cgxe_mdl_start(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance)
{
  init_simulink_io_address(moduleInstance);
  cgxertSetSimStateCompliance(moduleInstance->S, 2);
}

static void cgxe_mdl_initialize(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  PyObject *r;
  cgxertInitMLPythonIFace();
  moduleInstance->GIL = PyGILState_Ensure();
  moduleInstance->namespaceDict = getPyNamespaceDict();
  assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "Output", 0.0);
  execPyScript(moduleInstance, "", moduleInstance->namespaceDict);
  r = getPyDictVal(moduleInstance, moduleInstance->namespaceDict, "Output");
  *moduleInstance->b_y0 = PyObj_marshalIn(moduleInstance, r, NULL);
  Py_DecRef(r);
  PyGILState_Release(moduleInstance->GIL);
}

static void cgxe_mdl_outputs(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  PyObject *r;
  moduleInstance->GIL = PyGILState_Ensure();
  b_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "Hr",
                   *moduleInstance->u0);
  c_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "Output",
                   *moduleInstance->b_y0);
  d_assignToPyDict(moduleInstance, moduleInstance->namespaceDict, "Stat",
                   *moduleInstance->u1);
  b_execPyScript(moduleInstance,
                 "input = Hr\nif 11 <= input and input < 17 :\n    Output =Stat\nelse:\n    Output = 0",
                 moduleInstance->namespaceDict);
  r = b_getPyDictVal(moduleInstance, moduleInstance->namespaceDict, "Output");
  *moduleInstance->b_y0 = PyObj_marshalIn(moduleInstance, r, NULL);
  Py_DecRef(r);
  PyGILState_Release(moduleInstance->GIL);
}

static void cgxe_mdl_update(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_derivative(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_enable(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_disable(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  (void)moduleInstance;
}

static void cgxe_mdl_terminate(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  moduleInstance->GIL = PyGILState_Ensure();
  deleteDictItem(moduleInstance->namespaceDict, "Hr");
  b_deleteDictItem(moduleInstance->namespaceDict, "Stat");
  c_execPyScript(moduleInstance, "", moduleInstance->namespaceDict);
  Py_DecRef(moduleInstance->namespaceDict);
  PyGILState_Release(moduleInstance->GIL);
}

static void CheckPythonError(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
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

static real_T PyObj_marshalIn(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
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

static void assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
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

static void execPyScript(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
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

static PyObject *getPyDictVal(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance, PyObject *dict, char_T *key)
{
  PyObject *b_value;
  b_value = PyDict_GetItemString(dict, key);
  CheckPythonError(moduleInstance, NULL, 0);
  Py_IncRef(b_value);
  return b_value;
}

static void b_assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
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

static void c_assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
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

static void d_assignToPyDict(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
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

static void b_execPyScript(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
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

static PyObject *b_getPyDictVal(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
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

static void c_execPyScript(InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance,
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

static void init_simulink_io_address(InstanceStruct_hcEfncww8DrsTSMtZgKV1E
  *moduleInstance)
{
  moduleInstance->emlrtRootTLSGlobal = (void *)cgxertGetEMLRTCtx
    (moduleInstance->S);
  moduleInstance->u0 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 0);
  moduleInstance->u1 = (real_T *)cgxertGetInputPortSignal(moduleInstance->S, 1);
  moduleInstance->b_y0 = (real_T *)cgxertGetOutputPortSignal(moduleInstance->S,
    0);
}

/* CGXE Glue Code */
static void mdlOutputs_hcEfncww8DrsTSMtZgKV1E(SimStruct *S, int_T tid)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_outputs(moduleInstance);
}

static void mdlInitialize_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_initialize(moduleInstance);
}

static void mdlUpdate_hcEfncww8DrsTSMtZgKV1E(SimStruct *S, int_T tid)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_update(moduleInstance);
}

static void mdlDerivatives_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_derivative(moduleInstance);
}

static void mdlTerminate_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_terminate(moduleInstance);
  free((void *)moduleInstance);
}

static void mdlEnable_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_enable(moduleInstance);
}

static void mdlDisable_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)cgxertGetRuntimeInstance(S);
  cgxe_mdl_disable(moduleInstance);
}

static void mdlStart_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
  InstanceStruct_hcEfncww8DrsTSMtZgKV1E *moduleInstance =
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E *)calloc(1, sizeof
    (InstanceStruct_hcEfncww8DrsTSMtZgKV1E));
  moduleInstance->S = S;
  cgxertSetRuntimeInstance(S, (void *)moduleInstance);
  ssSetmdlOutputs(S, mdlOutputs_hcEfncww8DrsTSMtZgKV1E);
  ssSetmdlInitializeConditions(S, mdlInitialize_hcEfncww8DrsTSMtZgKV1E);
  ssSetmdlUpdate(S, mdlUpdate_hcEfncww8DrsTSMtZgKV1E);
  ssSetmdlDerivatives(S, mdlDerivatives_hcEfncww8DrsTSMtZgKV1E);
  ssSetmdlTerminate(S, mdlTerminate_hcEfncww8DrsTSMtZgKV1E);
  ssSetmdlEnable(S, mdlEnable_hcEfncww8DrsTSMtZgKV1E);
  ssSetmdlDisable(S, mdlDisable_hcEfncww8DrsTSMtZgKV1E);
  cgxe_mdl_start(moduleInstance);

  {
    uint_T options = ssGetOptions(S);
    options |= SS_OPTION_RUNTIME_EXCEPTION_FREE_CODE;
    ssSetOptions(S, options);
  }
}

static void mdlProcessParameters_hcEfncww8DrsTSMtZgKV1E(SimStruct *S)
{
}

void method_dispatcher_hcEfncww8DrsTSMtZgKV1E(SimStruct *S, int_T method, void
  *data)
{
  switch (method) {
   case SS_CALL_MDL_START:
    mdlStart_hcEfncww8DrsTSMtZgKV1E(S);
    break;

   case SS_CALL_MDL_PROCESS_PARAMETERS:
    mdlProcessParameters_hcEfncww8DrsTSMtZgKV1E(S);
    break;

   default:
    /* Unhandled method */
    /*
       sf_mex_error_message("Stateflow Internal Error:\n"
       "Error calling method dispatcher for module: hcEfncww8DrsTSMtZgKV1E.\n"
       "Can't handle method %d.\n", method);
     */
    break;
  }
}

mxArray *cgxe_hcEfncww8DrsTSMtZgKV1E_BuildInfoUpdate(void)
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

mxArray *cgxe_hcEfncww8DrsTSMtZgKV1E_fallback_info(void)
{
  const char* fallbackInfoFields[] = { "fallbackType", "incompatiableSymbol" };

  mxArray* fallbackInfoStruct = mxCreateStructMatrix(1, 1, 2, fallbackInfoFields);
  mxArray* fallbackType = mxCreateString("incompatibleFunction");
  mxArray* incompatibleSymbol = mxCreateString("PyModule_Check");
  mxSetFieldByNumber(fallbackInfoStruct, 0, 0, fallbackType);
  mxSetFieldByNumber(fallbackInfoStruct, 0, 1, incompatibleSymbol);
  return fallbackInfoStruct;
}
