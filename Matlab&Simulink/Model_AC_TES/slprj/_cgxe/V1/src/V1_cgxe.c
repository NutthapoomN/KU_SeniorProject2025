/* Include files */

#include "V1_cgxe.h"
#include "m_IbwGJ2kTlnb1jNk3gqvBYC.h"

unsigned int cgxe_V1_method_dispatcher(SimStruct* S, int_T method, void* data)
{
  if (ssGetChecksum0(S) == 3648092906 &&
      ssGetChecksum1(S) == 3349194560 &&
      ssGetChecksum2(S) == 323356825 &&
      ssGetChecksum3(S) == 3614876473) {
    method_dispatcher_IbwGJ2kTlnb1jNk3gqvBYC(S, method, data);
    return 1;
  }

  return 0;
}
