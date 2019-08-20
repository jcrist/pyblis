import sys
import jinja2


class Type(object):
    def __init__(self, char, ctype, is_complex, rtype=None):
        self.char = char
        self.ctype = ctype
        self.rtype = rtype or ctype
        self.is_complex = is_complex
        if is_complex:
            alpha_sig = "{0} alpha_real, {0} alpha_imag".format(self.rtype)
            beta_sig = "{0} beta_real, {0} beta_imag".format(self.rtype)
            alpha_init = "%s alpha = {alpha_real, alpha_imag}" % (self.ctype)
            beta_init = "%s beta = {beta_real, beta_imag}" % (self.ctype)
            alpha_py_sig = "ct.c_%s, ct.c_%s" % (self.rtype, self.rtype)
            alpha_py_call = "alpha.real, alpha.imag"
            beta_py_sig = "ct.c_%s, ct.c_%s" % (self.rtype, self.rtype)
            beta_py_call = "beta.real, beta.imag"
        else:
            alpha_sig = "{0} alpha".format(self.ctype)
            beta_sig = "{0} beta".format(self.ctype)
            alpha_init = beta_init = ""
            alpha_py_sig = beta_py_sig = "ct.c_%s" % self.ctype
            alpha_py_call = "alpha"
            beta_py_call = "beta"
        self.alpha_sig = alpha_sig
        self.beta_sig = beta_sig
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.alpha_py_sig = alpha_py_sig
        self.alpha_py_call = alpha_py_call
        self.beta_py_sig = beta_py_sig
        self.beta_py_call = beta_py_call


float32 = Type("s", "float", False)
float64 = Type("d", "double", False)
complex64 = Type("c", "scomplex", True, "float")
complex128 = Type("z", "dcomplex", True, "double")

all_types = [float32, float64, complex64, complex128]

parameters = dict(all_types=all_types)


def generate_source(template, target):
    with open(template) as f:
        data = f.read()
    template = jinja2.Template(data)
    output = template.render(**parameters)
    with open(target, 'w') as f:
        f.write(output)


if __name__ == "__main__":
    generate_source(sys.argv[1], sys.argv[2])
