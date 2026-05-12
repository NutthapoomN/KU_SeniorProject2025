/* Include files */

#include "Baseline_cgxe.h"
#include "m_J3s1aD7gUPp5APhMNqKmTH.h"
#include "m_hcEfncww8DrsTSMtZgKV1E.h"
#include "m_IbwGJ2kTlnb1jNk3gqvBYC.h"

unsigned int cgxe_Baseline_method_dispatcher(SimStruct* S, int_T method, void
  * data)
{
  if (ssGetChecksum0(S) == 851471633 &&
      ssGetChecksum1(S) == 2173898321 &&
      ssGetChecksum2(S) == 349637929 &&
      ssGetChecksum3(S) == 1449050620) {
    method_dispatcher_J3s1aD7gUPp5APhMNqKmTH(S, method, data);
    return 1;
  }

  if (ssGetChecksum0(S) == 3642982023 &&
      ssGetChecksum1(S) == 315018781 &&
      ssGetChecksum2(S) == 1707943244 &&
      ssGetChecksum3(S) == 1529549095) {
    method_dispatcher_hcEfncww8DrsTSMtZgKV1E(S, method, data);
    return 1;
  }

  if (ssGetChecksum0(S) == 3648092906 &&
      ssGetChecksum1(S) == 3349194560 &&
      ssGetChecksum2(S) == 323356825 &&
      ssGetChecksum3(S) == 3614876473) {
    method_dispatcher_IbwGJ2kTlnb1jNk3gqvBYC(S, method, data);
    return 1;
  }

  return 0;
}
