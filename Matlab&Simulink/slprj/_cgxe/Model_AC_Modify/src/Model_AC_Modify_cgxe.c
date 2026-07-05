/* Include files */

#include "Model_AC_Modify_cgxe.h"
#include "m_LEOKzC2TFyppoof6jQvGxD.h"

unsigned int cgxe_Model_AC_Modify_method_dispatcher(SimStruct* S, int_T method,
  void* data)
{
  if (ssGetChecksum0(S) == 19096615 &&
      ssGetChecksum1(S) == 1802518039 &&
      ssGetChecksum2(S) == 2261203961 &&
      ssGetChecksum3(S) == 822967678) {
    method_dispatcher_LEOKzC2TFyppoof6jQvGxD(S, method, data);
    return 1;
  }

  return 0;
}
