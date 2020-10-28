from decimal import Decimal


class Expression:
    _str_repr = "{x}"
    _static_vars = dict()
    _dynamic_vars = ['x']
    _var_nicknames = dict()
    _eval_vars = []
    _special_vars = dict()
    _magic_constants = False

    def __init__(self, **statics):
        self._static_vars.update(statics)
        self._dynamic_var_names = dict()

        for e in self._dynamic_vars:
            if e in self._var_nicknames:
                self._dynamic_var_names[e] = self._var_nicknames[e]
            else:
                self._dynamic_var_names[e] = e

        for e in self._var_nicknames:
            if e in self._static_vars:
                self._dynamic_var_names[e] = self._var_nicknames[e]

        if self._magic_constants:
            for e in self._static_vars:
                if e not in self._var_nicknames:
                    self._dynamic_var_names[e] = e
        self.computed = dict()
        self._last_execution_result = None
        self._last_exec_duplicated = False
        self.post_init()

    def post_init(self):
        pass

    def get_last_execution_result(self):
        return self._last_execution_result

    def is_last_execution_duplicated(self) -> bool:
        return self._last_exec_duplicated

    def execute(self,
                ignore_duplicates=True,
                **dynamic):
        for var in self._dynamic_vars:
            if var not in dynamic:
                raise ValueError(f"No such variable {var}")
        res = self._f(**dynamic, **self._static_vars)
        self._last_execution_result = res
        self._last_exec_duplicated = False
        if res in self.computed and self.computed[res] == dynamic:
            self._last_exec_duplicated = True
            if not ignore_duplicates:
                print(f"DUPLICATE")
        else:
            self.computed[res] = dynamic
        return res

    def _f(self, x, **kw):
        return x

    def compile(self, no_edit=False, include_static=True, **kwargs):
        if not no_edit:
            for e in kwargs:
                if str(kwargs[e]).startswith('-'):
                    kwargs[e] = f"({str(kwargs[e])})"
        if include_static:
            kwargs.update(self._static_vars)
        result = self._str_repr.format(**kwargs)
        return result

    def get_unified(self):
        t = dict(**self._static_vars)
        t.update(self._dynamic_var_names)
        t.update(self._special_vars)
        return self.compile(include_static=False, **t)

    def get_local(self, **dynamic):
        return self.compile(**dynamic,
                            **self._special_vars) + \
               f" = {self.execute(**dynamic)}"

    def __str__(self):
        return self._str_repr


class StaticExpression(Expression):
    _dynamic_vars = []
    _magic_constants = True


class MainExpr(Expression):
    _str_repr = "({x} + {L})/({x}^2 + {x} + {K})"

    def _f(self, x: Decimal, L: Decimal, K: Decimal, **kwargs):
        return (x + L) / (x ** 2 + x + K)


class SumMethod:
    _method_name = "basic method"

    def __init__(self, expr: Expression,
                 lbound: Decimal,
                 rbound: Decimal,
                 n: int,
                 accuracy=4,
                 suppres_info=False,
                 **kwargs):
        self.suppres_info = suppres_info
        self.expr = expr
        self.lbound = lbound
        self.rbound = rbound
        self.n = n
        self.x = Decimal("0")
        self.result = Decimal("0")
        self.stages_results = []
        self.extra = kwargs
        from decimal import getcontext
        getcontext().prec = accuracy
        self.accuracy = accuracy
        self.epsilon = Decimal("1") / 10 ** accuracy
        self.post_init(**kwargs)

    @property
    def name(self) -> str:
        return self._method_name

    def post_init(self, **kwargs):
        pass

    def get_stages(self) -> list:
        return []

    def run(self):
        if not self.suppres_info:
            print(f"Рассчёт суммы на [{self.lbound}; {self.rbound}], {self._method_name} при n = {self.n}:")

        for i, stage in enumerate(self.get_stages(), 1):
            if not issubclass(stage, SumMethod._Stage):
                raise TypeError(f"Expected SolveStage, got {type(stage).__name__}")

            if not self.suppres_info:
                print(f"Шаг #{i} ({stage.stage_name})")
            self.stages_results.append(stage(self).execute())

        return self.result

    @staticmethod
    def compute_with_logging(
            expr: Expression,
            name: str,
            show_full=False,
            level=2,
            only_compute=False,
            ignore_duplicates=False,
            **params) -> Decimal:
        solv = expr.get_local(**params).split(' = ', 2)[0]
        ans = expr.get_last_execution_result()
        if not only_compute:
            s = level * '\t' + f"{name} = "
            if not ignore_duplicates and expr.is_last_execution_duplicated():
                print(f"{s}{ans} (раcсчитано ранее)")
                return ans
            if show_full:
                s += f"{expr.get_unified()} = "
            print(s, end='' if len(s) <= 20 else '\n')
            s2 = ""
            if len(s) > 20:
                s2 += level * '\t' + len(name) * ' ' + ' = '
            s2 += solv + ' = '
            print(s2, end='' if len(s2) <= 30 else '\n')
            if len(s2) > 30:
                print(level * '\t' + len(name) * ' ' + ' = ', end='')
            print(ans)
        return ans

    class _Stage:
        stage_name = "basic stage"

        def __init__(self, method):
            self.rez = Decimal("0")
            self.method = method

        def start(self):
            pass

        def end(self):
            pass

        def step(self):
            pass

        def steps(self):
            pass

        def execute(self):
            self.start()
            self.steps()
            self.end()
            return self.rez


class TrapeziumMethod(SumMethod):
    _method_name = "Метод трапеций"

    def get_stages(self):
        return [
            self._StartStage,
            self._SumStage,
            self._EndStage
        ]

    class _StartStage(SumMethod._Stage):
        stage_name = "Делим [a, b] на n частей"

        def __init__(self, method: SumMethod):
            super().__init__(method)
            self._hExpr = self._H_Expression(
                a=method.lbound,
                b=method.rbound,
                n=method.n
            )

        def steps(self):
            self.rez = SumMethod.compute_with_logging(
                self._hExpr,
                'h', True
            )

        class _H_Expression(StaticExpression):
            _str_repr = "({b} - {a})/{n}"

            def _f(self, n: int, a: Decimal, b: Decimal, **kwargs):
                return (b - a) / n

    class _SumStage(SumMethod._Stage):
        stage_name = "Рассчитаем ∑[1..n)f(x)"

        def __init__(self, method: SumMethod):
            super().__init__(method)
            self.xExpr = self._Xi_Expression(
                a=method.lbound,
                h=method.stages_results[0]
            )
            self.rez = Decimal("0")

        def steps(self):
            for i in range(1, self.method.n):
                print(f"\t[i = {i}]:")
                self.rez += self.step(i)

        def step(self, i: int):
            return SumMethod.compute_with_logging(
                self.method.expr,
                "f(x)",
                x=SumMethod.compute_with_logging(
                    self.xExpr,
                    'x', True,
                    i=i
                )
            )

        class _Xi_Expression(Expression):
            _str_repr = "{a} + {i}*{h}"
            _dynamic_vars = ['i']
            _var_nicknames = {
                'h': 'h'
            }

            def _f(self, i: int, a: Decimal, h: Decimal, **kwargs):
                return a + i * h

    class _EndStage(SumMethod._Stage):
        stage_name = "Рассчитываем конечную сумму I"

        def __init__(self, method: SumMethod):
            super().__init__(method)
            self._IExpr = None

        def start(self):
            self._IExpr = self._I_Expression(
                f_a=SumMethod.compute_with_logging(
                    self.method.expr,
                    "f(a)",
                    x=self.method.lbound,
                    only_compute=(self.method.n != 4)
                ),
                f_b=SumMethod.compute_with_logging(
                    self.method.expr,
                    "f(b)",
                    x=self.method.rbound,
                    only_compute=(self.method.n != 4)
                ),
                sum_f_x=self.method.stages_results[1],
                h=self.method.stages_results[0]
            )

        def steps(self):
            self.method.result = SumMethod.compute_with_logging(
                self._IExpr,
                'I', True
            )

        class _I_Expression(StaticExpression):
            _str_repr = "({h}/2)*({f_a} + {f_b} + 2*{sum_f_x})"
            _var_nicknames = {
                'f_a': 'f(a)',
                'f_b': 'f(b)',
                'h': 'h',
                'sum_f_x': '∑[1..n)f(x)',
            }

            def _f(self,
                   h: Decimal,
                   f_a: Decimal,
                   f_b: Decimal,
                   sum_f_x: Decimal,
                   **kwargs):
                return (h / Decimal("2")) * (f_a + f_b + Decimal("2") * sum_f_x)


class SimpsonMethod(TrapeziumMethod):
    _method_name = "Метод Симпсона"

    def get_stages(self):
        return [
            self._StartStage,
            self._SumEvenStage,
            self._SumOddStage,
            self._EndStage
        ]

    class _StartStage(TrapeziumMethod._StartStage):
        stage_name = "Делим [a, b] на 2n частей"

        class _H_Expression(StaticExpression):
            _str_repr = "({b} - {a})/(2*{n})"

            def _f(self, n: int, a: Decimal, b: Decimal, **kwargs):
                return (b - a) / (2 * n)

    class _SumEvenStage(TrapeziumMethod._SumStage):
        stage_name = "Рассчитаем ∑[1..n)f(x_even)"

        def steps(self):
            for i in range(1, self.method.n):
                print(f"\t[i = {i}]:")
                self.rez += self.step(i * 2)

    class _SumOddStage(_SumEvenStage):
        stage_name = "Рассчитаем ∑[1..n]f(x_odd)"

        def steps(self):
            for i in range(1, self.method.n + 1):
                print(f"\t[i = {i}]:")
                self.rez += self.step(i * 2 - 1)

    class _EndStage(TrapeziumMethod._EndStage):
        def start(self):
            self._IExpr = self._I_Expression(
                f_a=SumMethod.compute_with_logging(
                    self.method.expr,
                    "f(a)",
                    x=self.method.lbound,
                    only_compute=(self.method.n != 4)
                ),
                f_b=SumMethod.compute_with_logging(
                    self.method.expr,
                    "f(b)",
                    x=self.method.rbound,
                    only_compute=(self.method.n != 4)
                ),
                sum_f_x2=self.method.stages_results[1],
                sum_f_x1=self.method.stages_results[2],
                h=self.method.stages_results[0]
            )

        class _I_Expression(StaticExpression):
            _str_repr = "({h}/3)*({f_a} + {f_b} + 2*{sum_f_x2} + 4*{sum_f_x1})"
            _var_nicknames = {
                'f_a': 'f(a)',
                'f_b': 'f(b)',
                'h': 'h',
                'sum_f_x2': '∑[1..n)f(x_even)',
                'sum_f_x1': '∑[1..n]f(x_odd)',
            }

            def _f(self,
                   h: Decimal,
                   f_a: Decimal,
                   f_b: Decimal,
                   sum_f_x1: Decimal,
                   sum_f_x2: Decimal,
                   **kwargs):
                return (h / Decimal("3")) * (f_a + f_b +
                                             Decimal("2") * sum_f_x2 +
                                             Decimal("4") * sum_f_x1)


class GaussMethod(SumMethod):
    _method_name = "Метод Гаусса"

    @staticmethod
    def HT(i, n):
        a, b = [[(Decimal("0.861136"), Decimal("0.347854")),
                 (Decimal("0.339981"), Decimal("0.652145"))],
                [(Decimal("0.932464"), Decimal("0.171324")),
                 (Decimal("0.661209"), Decimal("0.360761")),
                 (Decimal("0.238619"), Decimal("0.467913"))],
                [(Decimal("0.960289"), Decimal("0.101228")),
                 (Decimal("0.796666"), Decimal("0.222381")),
                 (Decimal("0.525532"), Decimal("0.313706")),
                 (Decimal("0.183434"), Decimal("0.362683"))]
                ][n // 2 - 2][i - 1 if i <= n // 2 else n - i]
        return -a if i <= n // 2 else a, b

    def get_stages(self):
        return [
            self._SumStage,
            self._EndStage
        ]

    class _SumStage(SumMethod._Stage):
        stage_name = "Рассчитаем ∑[1..n](Ai*f(x))"

        def __init__(self, method: SumMethod):
            super().__init__(method)
            self.xExpr = self._Xi_Expression(
                a=method.lbound,
                b=method.rbound
            )
            self.rez = Decimal("0")

        def steps(self):
            for i in range(1, self.method.n + 1):
                print(f"\t[i = {i}]:")
                self.rez += self.step(i)

        def step(self, i: int):
            return SumMethod.compute_with_logging(
                self._AF_Expression(
                    A=GaussMethod.HT(i, self.method.n)[1],
                    f_x=SumMethod.compute_with_logging(
                        self.method.expr,
                        "f(x)",
                        x=SumMethod.compute_with_logging(
                            self.xExpr,
                            'x', True,
                            t=GaussMethod.HT(i, self.method.n)[0]
                        )
                    )
                ),
                "A*f(x)"
            )

        class _Xi_Expression(StaticExpression):
            _str_repr = "({a}+{b})/2 + (({b}-{a})*{t})/2"
            _dynamic_vars = ['t']

            def _f(self, t: Decimal, a: Decimal, b: Decimal, **kwargs):
                return (a + b) / Decimal("2") + ((b - a) * t) / Decimal("2")

        class _AF_Expression(StaticExpression):
            _str_repr = "{A}*{f_x}"
            _var_nicknames = {
                'A': 'A(i)',
                'f_x': 'f(x)',
            }

            def _f(self, A: Decimal, f_x: Decimal, **kwargs):
                return A * f_x

    class _EndStage(SumMethod._Stage):
        stage_name = "Рассчитываем конечную сумму I"

        def __init__(self, method: SumMethod):
            super().__init__(method)
            self._IExpr = None

        def start(self):
            self._IExpr = self._I_Expression(
                sum_af_x=self.method.stages_results[0],
                a=self.method.lbound,
                b=self.method.rbound,
            )

        def steps(self):
            self.method.result = SumMethod.compute_with_logging(
                self._IExpr,
                'I', True
            )

        class _I_Expression(StaticExpression):
            _str_repr = "(({b}-{a})*{sum_af_x})/2"
            _var_nicknames = {
                'sum_af_x': '∑[1..n]A(i)f(x)',
            }

            def _f(self,
                   a: Decimal,
                   b: Decimal,
                   sum_af_x: Decimal,
                   **kwargs):
                return ((b - a) * sum_af_x) / Decimal("2")


def main():
    DEBUG = False
    if DEBUG:
        K = Decimal("3.6")
        L = Decimal("2.0")
    else:
        K = Decimal(input("Введите K: "))
        L = Decimal(input("Введите L: "))
    a = (K - L) / Decimal("2")
    b = K + L
    expr = MainExpr(K=K, L=L)

    print("f(x):", expr.get_unified())

    result = {}
    methods = [
        TrapeziumMethod,
        SimpsonMethod,
        GaussMethod
    ]

    for method in methods:
        for n in [4, 6, 8]:
            print("\n\t\t\t---\n")
            solver = method(expr=expr,
                            lbound=a,
                            rbound=b,
                            n=n,
                            accuracy=7,
                            )
            ans = solver.run()
            if solver.name not in result:
                result[solver.name] = []
            result[solver.name].append(ans)
            print(f"Тогда при n = {n}, сумма на отрезке равна {ans}")
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = ["N", "4", "6", "8"]
    for k, v in result.items():
        x.add_row([k, *v])
    print(x)


if __name__ == '__main__':
    main()
