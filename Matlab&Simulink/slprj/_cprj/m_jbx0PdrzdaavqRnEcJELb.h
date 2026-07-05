#ifndef __jbx0PdrzdaavqRnEcJELb_h__
#define __jbx0PdrzdaavqRnEcJELb_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_jbx0PdrzdaavqRnEcJELb
#define typedef_InstanceStruct_jbx0PdrzdaavqRnEcJELb

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *b_y0;
  real_T *b_y1;
} InstanceStruct_jbx0PdrzdaavqRnEcJELb;

#endif                                 /* typedef_InstanceStruct_jbx0PdrzdaavqRnEcJELb */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_jbx0PdrzdaavqRnEcJELb(SimStruct *S, int_T method,
  void* data);

#endif
