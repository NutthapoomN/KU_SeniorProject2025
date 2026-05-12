#ifndef __P9D5w2CSyhxbNG964ppPhD_h__
#define __P9D5w2CSyhxbNG964ppPhD_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_P9D5w2CSyhxbNG964ppPhD
#define typedef_InstanceStruct_P9D5w2CSyhxbNG964ppPhD

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *b_y0;
} InstanceStruct_P9D5w2CSyhxbNG964ppPhD;

#endif                                 /* typedef_InstanceStruct_P9D5w2CSyhxbNG964ppPhD */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_P9D5w2CSyhxbNG964ppPhD(SimStruct *S, int_T method,
  void* data);

#endif
