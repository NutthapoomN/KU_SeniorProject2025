#ifndef __fUi2Nrdrc7NmX51idBWkvB_h__
#define __fUi2Nrdrc7NmX51idBWkvB_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_fUi2Nrdrc7NmX51idBWkvB
#define typedef_InstanceStruct_fUi2Nrdrc7NmX51idBWkvB

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
} InstanceStruct_fUi2Nrdrc7NmX51idBWkvB;

#endif                                 /* typedef_InstanceStruct_fUi2Nrdrc7NmX51idBWkvB */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_fUi2Nrdrc7NmX51idBWkvB(SimStruct *S, int_T method,
  void* data);

#endif
