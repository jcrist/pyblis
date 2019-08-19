/* Simple wrappers around BLIS functions
 *
 * Needed for two reasons:
 *
 * - Rename symbols so that our lib can be used in a process that may also use
 *   a different version of BLIS. This allows us to statically link in BLIS,
 *   and not conflict with any existing version.
 *
 * - Remove pointers to scalars, as numba can't currently handle these easily.
 */
#include <stdbool.h>
#include "blis.h"

void pybli_sgemm(bool transa,
                 bool transb,
                 dim_t   m,
                 dim_t   n,
                 dim_t   k,
                 float  alpha,
                 float*  a, inc_t rsa, inc_t csa,
                 float*  b, inc_t rsb, inc_t csb,
                 float  beta,
                 float*  c, inc_t rsc, inc_t csc) {
    bli_sgemm(transa ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              transb ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              m, n, k,
              &alpha,
              a, rsa, csa,
              b, rsb, csb,
              &beta,
              c, rsc, csc);
}

void pybli_dgemm(bool transa,
                 bool transb,
                 dim_t   m,
                 dim_t   n,
                 dim_t   k,
                 double  alpha,
                 double*  a, inc_t rsa, inc_t csa,
                 double*  b, inc_t rsb, inc_t csb,
                 double  beta,
                 double*  c, inc_t rsc, inc_t csc) {
    bli_dgemm(transa ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              transb ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              m, n, k,
              &alpha,
              a, rsa, csa,
              b, rsb, csb,
              &beta,
              c, rsc, csc);
}
