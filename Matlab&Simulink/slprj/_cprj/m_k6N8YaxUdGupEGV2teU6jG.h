#ifndef __k6N8YaxUdGupEGV2teU6jG_h__
#define __k6N8YaxUdGupEGV2teU6jG_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_k6N8YaxUdGupEGV2teU6jG
#define typedef_InstanceStruct_k6N8YaxUdGupEGV2teU6jG

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
} InstanceStruct_k6N8YaxUdGupEGV2teU6jG;

#endif                                 /* typedef_InstanceStruct_k6N8YaxUdGupEGV2teU6jG */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_k6N8YaxUdGupEGV2teU6jG(SimStruct *S, int_T method,
  void* data);

#endif
