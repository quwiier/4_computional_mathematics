import numpy as np

def swap_rows(A, B, row1, row2):
    """Меняет местами две строки в матрице A и векторе B"""
    A[row1], A[row2] = A[row2], A[row1]
    B[row1], B[row2] = B[row2], B[row1]

def gauss_elimination_with_pivoting(A, B):
    """Метод Гаусса с выбором главного элемента по столбцам и подсчётом перестановок"""
    n = len(A)
    swap_count = 0

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        if max_row != i:
            swap_rows(A, B, i, max_row)
            swap_count += 1

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]

    return A, B, swap_count

def compute_determinant(U, swap_count):
    """Вычисляет определитель с учетом количества перестановок"""
    det = (-1) ** swap_count
    for i in range(len(U)):
        det *= U[i][i]
    return det

def compute_residual(A, x, B):
    """Вычисляет вектор невязок"""
    residuals = [sum(A[i][j] * x[j] for j in range(len(A))) - B[i] for i in range(len(A))]
    return residuals

def back_substitution(A, B):
    """Обратный ход метода Гаусса"""
    n = len(A)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (B[i] - sum_ax) / A[i][i]
    return x

def main():
    n = int(input("Введите размерность матрицы: "))
    if n > 20:
        print("n > 20, эффективность и точность решения снижена")

    print("Введите коэффициенты матрицы построчно:")
    A = [list(map(float, input().split())) for _ in range(n)]

    print("Введите свободные члены:")
    B = list(map(float, input().split()))

    determinant_numpy = np.linalg.det(np.array(A))
    # print(determinant_numpy) # вывод до выполнения алгоритма, чтобы убедиться, что равен 0, тогда будет ошибка "Система несовместная (0 на главной диагонали)"
    x_numpy = np.linalg.solve(np.array(A), np.array(B))

    A_copy = [row[:] for row in A]
    B_copy = B[:]

    U, B_transformed, swap_count = gauss_elimination_with_pivoting(A_copy, B_copy)
    x_gauss = back_substitution(U, B_transformed)
    determinant = compute_determinant(U, swap_count)
    residuals = compute_residual(A, x_gauss, B)


    print("\nТреугольная матрица после преобразований:")
    for i in range(n):
        print(" ".join(f"{num:.2f}" for num in U[i]) + f" | {B_transformed[i]:.2f}")

    print("\nОпределитель матрицы:", determinant)
    print("Определитель матрицы (NumPy):", determinant_numpy)
    # print(swap_count) # колво перестановок для определения знака детерминанта -1^k, k-колво перестановок

    print("\nРешение системы:", " ".join(f"x{i+1} = {xi:.15f}" for i, xi in enumerate(x_gauss)))
    print("Решение системы (NumPy):", " ".join(f"x{i+1} = {xi:.15f}" for i, xi in enumerate(x_numpy)))

    print("\nВектор невязок:", " ".join(f"{r:.30f}" for r in residuals)) # 30 знаков после запятой, чтобы убедиться, что решение лишь приблизительное

if __name__ == "__main__":
    try:
        main()
    except ZeroDivisionError:
        print("Система несовместная (0 на главной диагонали)") # самый простой вариант так отловить этот случай, решений не будет, если детерминант равен 0, однако мы можем просто запустить алгоритм, в какой-то момент просто произойдет деление на 0, которое и отлавливает этот exeption
    except Exception as e:
        print(f"Произошла ошибка: {e}")
