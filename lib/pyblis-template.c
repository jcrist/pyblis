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

#define from_upper(u) \
    (u) ? BLIS_UPPER : BLIS_LOWER

/* GEMM */
{% for T in all_types %}
void pybli_{{ T.char }}gemm(
    bool a_trans, bool a_conj,
    bool b_trans, bool b_conj,
    dim_t   m,
    dim_t   n,
    dim_t   k,
    {{ T.alpha_sig }},
    {{ T.ctype }}*  a, inc_t rsa, inc_t csa,
    {{ T.ctype }}*  b, inc_t rsb, inc_t csb,
    {{ T.beta_sig }},
    {{T.ctype }}*  c, inc_t rsc, inc_t csc,
    dim_t nthreads
) {
    INIT_RNTM;
    {% if T.is_complex %}
    {{ T.alpha_init }};
    {{ T.beta_init }};
    {% endif %}
    bli_{{ T.char }}gemm_ex(
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
{% endfor %}

/* SYRK */
{% for T in all_types %}
void pybli_{{ T.char }}syrk(
    bool a_trans,
    bool a_conj,
    bool c_upper,
    dim_t   m,
    dim_t   k,
    {{ T.alpha_sig }},
    {{ T.ctype }}*  a, inc_t rsa, inc_t csa,
    {{ T.beta_sig }},
    {{T.ctype }}*  c, inc_t rsc, inc_t csc,
    dim_t nthreads
) {
    INIT_RNTM;
    {% if T.is_complex %}
    {{ T.alpha_init }};
    {{ T.beta_init }};
    {% endif %}
    bli_{{ T.char }}syrk_ex(
        from_upper(c_upper),
        from_trans_conj(a_trans, a_conj),
        m, k,
        &alpha,
        a, rsa, csa,
        &beta,
        c, rsc, csc,
        NULL,
        &rntm
    );
}
{% endfor %}

/* MKSYMM */
{% for T in all_types %}
void pybli_{{ T.char }}mksymm(
    bool upper,
    dim_t   m,
    {{ T.ctype }}*  a, inc_t rsa, inc_t csa,
    dim_t nthreads
) {
    INIT_RNTM;
    bli_{{ T.char }}mksymm_ex(
        from_upper(upper),
        m,
        a, rsa, csa,
        NULL,
        &rntm
    );
}
{% endfor %}
