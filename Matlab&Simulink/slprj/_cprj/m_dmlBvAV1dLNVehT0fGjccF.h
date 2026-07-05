#ifndef __dmlBvAV1dLNVehT0fGjccF_h__
#define __dmlBvAV1dLNVehT0fGjccF_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_dmlBvAV1dLNVehT0fGjccF
#define typedef_InstanceStruct_dmlBvAV1dLNVehT0fGjccF

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
  real_T *b_y1;
} InstanceStruct_dmlBvAV1dLNVehT0fGjccF;

#endif                                 /* typedef_InstanceStruct_dmlBvAV1dLNVehT0fGjccF */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_dmlBvAV1dLNVehT0fGjccF(SimStruct *S, int_T method,
  void* data);

#endif
