#ifndef __NGiGOBVa6l8QfX7tNnMjx_h__
#define __NGiGOBVa6l8QfX7tNnMjx_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_NGiGOBVa6l8QfX7tNnMjx
#define typedef_InstanceStruct_NGiGOBVa6l8QfX7tNnMjx

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *u2;
  real_T *u3;
  real_T *u4;
  real_T *b_y0;
} InstanceStruct_NGiGOBVa6l8QfX7tNnMjx;

#endif                                 /* typedef_InstanceStruct_NGiGOBVa6l8QfX7tNnMjx */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_NGiGOBVa6l8QfX7tNnMjx(SimStruct *S, int_T method,
  void* data);

#endif
