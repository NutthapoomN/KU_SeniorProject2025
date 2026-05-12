#ifndef __8cBzjVIAGIGXoWD8brqk7B_h__
#define __8cBzjVIAGIGXoWD8brqk7B_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_8cBzjVIAGIGXoWD8brqk7B
#define typedef_InstanceStruct_8cBzjVIAGIGXoWD8brqk7B

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *b_y0;
} InstanceStruct_8cBzjVIAGIGXoWD8brqk7B;

#endif                                 /* typedef_InstanceStruct_8cBzjVIAGIGXoWD8brqk7B */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_8cBzjVIAGIGXoWD8brqk7B(SimStruct *S, int_T method,
  void* data);

#endif
