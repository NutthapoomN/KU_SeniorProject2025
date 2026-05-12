#ifndef __AuTOk0a5YER3Ek6UMYn24D_h__
#define __AuTOk0a5YER3Ek6UMYn24D_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_AuTOk0a5YER3Ek6UMYn24D
#define typedef_InstanceStruct_AuTOk0a5YER3Ek6UMYn24D

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
  real_T *u5;
  real_T *b_y0;
  real_T *b_y1;
} InstanceStruct_AuTOk0a5YER3Ek6UMYn24D;

#endif                                 /* typedef_InstanceStruct_AuTOk0a5YER3Ek6UMYn24D */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_AuTOk0a5YER3Ek6UMYn24D(SimStruct *S, int_T method,
  void* data);

#endif
