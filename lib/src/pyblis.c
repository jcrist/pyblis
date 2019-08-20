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
#include "blis/blis.h"

#define INIT_RNTM \
    rntm_t rntm = BLIS_RNTM_INITIALIZER; \
    if (nthreads > 0) { \
        bli_rntm_set_num_threads(nthreads, &rntm); \
    }

#define from_trans_conj(t, c) \
    (c) ? ((t) ? BLIS_CONJ_TRANSPOSE : BLIS_CONJ_NO_TRANSPOSE) : \
          ((t) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE)

void pybli_sgemm(bool a_trans, bool a_conj,
                 bool b_trans, bool b_conj,
                 dim_t   m,
                 dim_t   n,
                 dim_t   k,
                 float  alpha,
                 float*  a, inc_t rsa, inc_t csa,
                 float*  b, inc_t rsb, inc_t csb,
                 float  beta,
                 float*  c, inc_t rsc, inc_t csc,
                 dim_t nthreads) {
    INIT_RNTM;
    bli_sgemm_ex(
        from_trans_conj(a_trans, a_conj),
        from_trans_conj(b_trans, b_conj),
        m, n, k,
        &alpha,
        a, rsa, csa,
        b, rsb, csb,
        &beta,
        c, rsc, csc,
        NULL,
        &rntm
    );
}

void pybli_dgemm(bool a_trans, bool a_conj,
                 bool b_trans, bool b_conj,
                 dim_t   m,
                 dim_t   n,
                 dim_t   k,
                 double  alpha,
                 double*  a, inc_t rsa, inc_t csa,
                 double*  b, inc_t rsb, inc_t csb,
                 double  beta,
                 double*  c, inc_t rsc, inc_t csc,
                 dim_t nthreads) {
    INIT_RNTM;
    bli_dgemm_ex(
        from_trans_conj(a_trans, a_conj),
        from_trans_conj(b_trans, b_conj),
        m, n, k,
        &alpha,
        a, rsa, csa,
        b, rsb, csb,
        &beta,
        c, rsc, csc,
        NULL,
        &rntm
    );
}

void pybli_cgemm(bool a_trans, bool a_conj,
                 bool b_trans, bool b_conj,
                 dim_t   m,
                 dim_t   n,
                 dim_t   k,
                 float  alpha_real, float alpha_imag,
                 scomplex*  a, inc_t rsa, inc_t csa,
                 scomplex*  b, inc_t rsb, inc_t csb,
                 float  beta_real, float beta_imag,
                 scomplex*  c, inc_t rsc, inc_t csc,
                 dim_t nthreads) {
    INIT_RNTM;
    scomplex alpha = { alpha_real, alpha_imag };
    scomplex beta = { beta_real, beta_imag };
    bli_cgemm_ex(
        from_trans_conj(a_trans, a_conj),
        from_trans_conj(b_trans, b_conj),
        m, n, k,
        &alpha,
        a, rsa, csa,
        b, rsb, csb,
        &beta,
        c, rsc, csc,
        NULL,
        &rntm
    );
}

void pybli_zgemm(bool a_trans, bool a_conj,
                 bool b_trans, bool b_conj,
                 dim_t   m,
                 dim_t   n,
                 dim_t   k,
                 double  alpha_real, double alpha_imag,
                 dcomplex*  a, inc_t rsa, inc_t csa,
                 dcomplex*  b, inc_t rsb, inc_t csb,
                 double  beta_real, double beta_imag,
                 dcomplex*  c, inc_t rsc, inc_t csc,
                 dim_t nthreads) {
    INIT_RNTM;
    dcomplex alpha = { alpha_real, alpha_imag };
    dcomplex beta = { beta_real, beta_imag };
    bli_zgemm_ex(
        from_trans_conj(a_trans, a_conj),
        from_trans_conj(b_trans, b_conj),
        m, n, k,
        &alpha,
        a, rsa, csa,
        b, rsb, csb,
        &beta,
        c, rsc, csc,
        NULL,
        &rntm
    );
}
