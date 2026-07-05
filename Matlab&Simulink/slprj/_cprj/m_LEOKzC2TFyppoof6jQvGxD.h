#ifndef __LEOKzC2TFyppoof6jQvGxD_h__
#define __LEOKzC2TFyppoof6jQvGxD_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_LEOKzC2TFyppoof6jQvGxD
#define typedef_InstanceStruct_LEOKzC2TFyppoof6jQvGxD

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
  real_T *b_y1;
} InstanceStruct_LEOKzC2TFyppoof6jQvGxD;

#endif                                 /* typedef_InstanceStruct_LEOKzC2TFyppoof6jQvGxD */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_LEOKzC2TFyppoof6jQvGxD(SimStruct *S, int_T method,
  void* data);

#endif
