#ifndef __Fy1mPMYU2fheSweB1fg0Y_h__
#define __Fy1mPMYU2fheSweB1fg0Y_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_Fy1mPMYU2fheSweB1fg0Y
#define typedef_InstanceStruct_Fy1mPMYU2fheSweB1fg0Y

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *b_y0;
} InstanceStruct_Fy1mPMYU2fheSweB1fg0Y;

#endif                                 /* typedef_InstanceStruct_Fy1mPMYU2fheSweB1fg0Y */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_Fy1mPMYU2fheSweB1fg0Y(SimStruct *S, int_T method,
  void* data);

#endif
