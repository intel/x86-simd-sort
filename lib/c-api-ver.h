#include "c-api-headers.h"

{
  global:

#define XSS_XI1(n,t) XSS_C_EXP_NAME1(qsort,n);
#define XSS_XF1(n,t) XSS_C_EXP_NAME1(qsort,n);
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_C_EXP_NAME1(qselect,n);
#define XSS_XF1(n,t) XSS_C_EXP_NAME1(qselect,n);
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_C_EXP_NAME1(partial_qsort,n);
#define XSS_XF1(n,t) XSS_C_EXP_NAME1(partial_qsort,n);
#include "x-macro1.h"


#define XSS_XI2(n1,t1, n2,t2) XSS_C_EXP_NAME2(keyvalue_qsort, n1, n2);
#define XSS_XF2(n1,t1, n2,t2) XSS_C_EXP_NAME2(keyvalue_qsort, n1, n2);
#include "x-macro2.h"

#define XSS_XI2(n1,t1, n2,t2) XSS_C_EXP_NAME2(keyvalue_qselect, n1, n2);
#define XSS_XF2(n1,t1, n2,t2) XSS_C_EXP_NAME2(keyvalue_qselect, n1, n2);
#include "x-macro2.h"

#define XSS_XI2(n1,t1, n2,t2) XSS_C_EXP_NAME2(keyvalue_partial_qsort, n1, n2);
#define XSS_XF2(n1,t1, n2,t2) XSS_C_EXP_NAME2(keyvalue_partial_qsort, n1, n2);
#include "x-macro2.h"

  local:
    *;
};
