#ifndef __NiRpZBW33fLJOwLBWZDch_h__
#define __NiRpZBW33fLJOwLBWZDch_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_NiRpZBW33fLJOwLBWZDch
#define typedef_InstanceStruct_NiRpZBW33fLJOwLBWZDch

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
  real_T *b_y1;
  real_T *y2;
  real_T *y3;
  real_T *y4;
  real_T *y5;
} InstanceStruct_NiRpZBW33fLJOwLBWZDch;

#endif                                 /* typedef_InstanceStruct_NiRpZBW33fLJOwLBWZDch */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_NiRpZBW33fLJOwLBWZDch(SimStruct *S, int_T method,
  void* data);

#endif
