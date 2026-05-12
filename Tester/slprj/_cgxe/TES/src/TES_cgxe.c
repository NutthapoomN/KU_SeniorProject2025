/* Include files */

#include "TES_cgxe.h"
#include "m_fUi2Nrdrc7NmX51idBWkvB.h"

unsigned int cgxe_TES_method_dispatcher(SimStruct* S, int_T method, void* data)
{
  if (ssGetChecksum0(S) == 870515955 &&
      ssGetChecksum1(S) == 357462720 &&
      ssGetChecksum2(S) == 2899334208 &&
      ssGetChecksum3(S) == 3630015583) {
    method_dispatcher_fUi2Nrdrc7NmX51idBWkvB(S, method, data);
    return 1;
  }

  return 0;
}
