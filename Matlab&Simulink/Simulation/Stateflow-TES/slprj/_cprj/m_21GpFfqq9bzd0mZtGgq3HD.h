#ifndef __21GpFfqq9bzd0mZtGgq3HD_h__
#define __21GpFfqq9bzd0mZtGgq3HD_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_21GpFfqq9bzd0mZtGgq3HD
#define typedef_InstanceStruct_21GpFfqq9bzd0mZtGgq3HD

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
} InstanceStruct_21GpFfqq9bzd0mZtGgq3HD;

#endif                                 /* typedef_InstanceStruct_21GpFfqq9bzd0mZtGgq3HD */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_21GpFfqq9bzd0mZtGgq3HD(SimStruct *S, int_T method,
  void* data);

#endif
