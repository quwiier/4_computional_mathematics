def swap_rows(A, B, row1, row2):
    """Меняет местами две строки в матрице A и векторе B"""
    A[row1], A[row2] = A[row2], A[row1]
    B[row1], B[row2] = B[row2], B[row1]

def gauss_elimination_with_pivoting(A, B):
    """Метод Гаусса с выбором главного элемента по столбцам"""
    n = len(A)

    # Прямой ход с выбором главного элемента
    for i in range(n):
        # Находим строку с максимальным элементом в текущем столбце
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        # Меняем строки, если найден максимальный элемент ниже текущего
        if max_row != i:
            swap_rows(A, B, i, max_row)

        # Преобразуем в треугольную форму
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]

    return A, B

def compute_determinant(U):
    """Вычисляет определитель как произведение диагональных элементов треугольной матрицы"""
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det

def compute_residual(A, x, B):
    """Вычисляет вектор невязок"""
    n = len(A)
    residuals = [0] * n
    for i in range(n):
        sum_ax = sum(A[i][j] * x[j] for j in range(n))
        residuals[i] = sum_ax - B[i]
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
    # Ввод размерности
    n = int(input("Введите размерность матрицы: "))
    if (n > 20):
        print("n > 20, эффективность и точность решения снижена")

    # Ввод матрицы
    print("Введите коэффициенты матрицы построчно:")
    A = [list(map(float, input().split())) for _ in range(n)]

    # Ввод вектора B
    print("Введите свободные члены:")
    B = list(map(float, input().split()))

    # Копируем данные, чтобы не изменять исходные
    A_copy = [row[:] for row in A]
    B_copy = B[:]

    # Метод Гаусса
    U, B_transformed = gauss_elimination_with_pivoting(A_copy, B_copy)
    x = back_substitution(U, B_transformed)

    # Определитель
    determinant = compute_determinant(U)

    # Вектор невязок
    residuals = compute_residual(A, x, B)

    # Вывод треугольной матрицы (включая преобразованный столбец B)
    print("\nТреугольная матрица после преобразований:")
    for i in range(n):
        print(" ".join(f"{num:.2f}" for num in U[i]) + f" | {B_transformed[i]:.2f}")

    print("\nОпределитель матрицы:", determinant)
    print("\nРешение системы:", " ".join(f"x{i+1} = {xi:.6f}" for i, xi in enumerate(x)))
    print("\nВектор невязок:", " ".join(f"{r:.6f}" for r in residuals))

if __name__ == "__main__":
    main()
