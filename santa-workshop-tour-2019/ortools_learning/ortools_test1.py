from ortools.linear_solver import pywraplp


def main():

    # 首先，调用CBC求解器
    # 还记得我们在LP问题中调用的求解器吗，
    # 是 pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
    solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # 和LP问题一样，定义变量 x,y 并指定其定义域
    # 这里需要注意，我们定义的是Int类型的变量
    x = solver.IntVar(0.0, solver.infinity(), 'x')
    y = solver.IntVar(0.0, solver.infinity(), 'y')

    # 添加约束，方法和LP一样，通过制定系数来添加约束
    # 如果能像cp-sat那么方便就好了
    # x + 7 * y <= 17.5
    constraint1 = solver.Constraint(-solver.infinity(), 17.5)
    constraint1.SetCoefficient(x, 1)
    constraint1.SetCoefficient(y, 7)

    # 添加约束
    # x <= 3.5
    constraint2 = solver.Constraint(-solver.infinity(), 3.5)
    constraint2.SetCoefficient(x, 1)
    constraint2.SetCoefficient(y, 0)

    # 定义目标函数
    # Maximize x + 10 * y.
    objective = solver.Objective()
    objective.SetCoefficient(x, 1)
    objective.SetCoefficient(y, 10)
    objective.SetMaximization()

    # 求解问题，并打印结果，后面的代码就很简单了
    result_status = solver.Solve()
    # The problem has an optimal solution.
    assert result_status == pywraplp.Solver.OPTIMAL

    # The solution looks legit (when using solvers other than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    assert solver.VerifySolution(1e-7, True)

    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())

    # The objective value of the solution.
    print('Optimal objective value = %d' % solver.Objective().Value())
    print()
    # The value of each variable in the solution.
    variable_list = [x, y]

    for variable in variable_list:
        print('%s = %d' % (variable.name(), variable.solution_value()))


if __name__ == '__main__':
    main()
