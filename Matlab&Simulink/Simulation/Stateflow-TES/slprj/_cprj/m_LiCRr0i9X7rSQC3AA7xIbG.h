#ifndef __LiCRr0i9X7rSQC3AA7xIbG_h__
#define __LiCRr0i9X7rSQC3AA7xIbG_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_LiCRr0i9X7rSQC3AA7xIbG
#define typedef_InstanceStruct_LiCRr0i9X7rSQC3AA7xIbG

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *b_y0;
} InstanceStruct_LiCRr0i9X7rSQC3AA7xIbG;

#endif                                 /* typedef_InstanceStruct_LiCRr0i9X7rSQC3AA7xIbG */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_LiCRr0i9X7rSQC3AA7xIbG(SimStruct *S, int_T method,
  void* data);

#endif
