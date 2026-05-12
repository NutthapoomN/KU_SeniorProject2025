#ifndef __hcEfncww8DrsTSMtZgKV1E_h__
#define __hcEfncww8DrsTSMtZgKV1E_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_hcEfncww8DrsTSMtZgKV1E
#define typedef_InstanceStruct_hcEfncww8DrsTSMtZgKV1E

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
} InstanceStruct_hcEfncww8DrsTSMtZgKV1E;

#endif                                 /* typedef_InstanceStruct_hcEfncww8DrsTSMtZgKV1E */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_hcEfncww8DrsTSMtZgKV1E(SimStruct *S, int_T method,
  void* data);

#endif
