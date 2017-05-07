/*!
    @file
    @brief Contains TNeuroNet class implementation

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/
#include <cstdlib>
#include <values.h>

#include "tneuronet.h"

TNeuroNet::TNeuroNet(const vector<int> &topology) throw(invalid_argument)
{
    // Check nuro net parameters correctness
    if(topology.size() < 3)
        throw invalid_argument("Wrong neural network topology!");
    for(auto element : topology) {
        if(element <=0) throw invalid_argument("Wrong neural network topology!");
    }
    construct(topology);
}

void TNeuroNet::construct(const vector<int> &topology)
{
    neuronCount = topology;
    innerNeurons = vector<int>(topology.begin() + 1, topology.end() - 2);

    // Инициализация массива выходов узлов нейронов
    Y = new double *[innerNeurons.size()+2];
    // Инициализация массива сумм входов нейронов
    inSum = new double *[innerNeurons.size()+2];
    // Инициализация массива значений ошибки обучения сети
    delta = new double *[innerNeurons.size()+2];
    // выделение памяти под вспомогательную переменную
    sum_e_by_w = new double *[innerNeurons.size()+2];
    // Выходы нейронов и ошибки обучения (0 - вход, innerNeurons.size()+1 - выход)
    for(int k=0; k < innerNeurons.size() + 2; k++)
    {
        Y[k] = new double [neuronCount[k]];
        inSum[k] = new double [neuronCount[k]];
        delta[k] = new double [neuronCount[k]];
        // выделение памяти под вспомогательную переменную
        sum_e_by_w[k] = new double [neuronCount[k]];
    }
    // Инициализация указателей на вход и выход нейросети
    in = Y[0];
    out = Y[innerNeurons.size()+1];
    // Инициализация массива значений синаптических весов нейронов
    w = new double **[innerNeurons.size()+1];
    w0 = new double *[innerNeurons.size()+1];
    e_by_x = new double **[innerNeurons.size()+1];
    for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
    {
        w[k-1] = new double *[neuronCount[k]];
        w0[k-1] = new double [neuronCount[k]];
        e_by_x[k-1] = new double *[neuronCount[k]];
        for(int i = 0; i < neuronCount[k]; i++) // Цикл по нейронам текущего слоя
        {
            w[k-1][i] = new double [neuronCount[k-1]]; // - по нейронам предыдущего слоя
            // выделение памяти под вспомогательную переменную
            e_by_x[k-1][i] = new double [neuronCount[k-1]];
        }
    }
    scgMemAllocated = true; // Выделять память под вспомогательные переменные
    // метода SCG
    netInitialized = false;  // Сеть не инициализирована
    firstGACycle = true;     // Первый цикл работы ГА, на котором инициализируются
    // переменные, выделяется память под объекты
    nu = 0.2;
    maxDelta = 0.0;
    // Инициализация генератора случайных чисел
    time_t t;
    srand((unsigned) time(&t));
}

TNeuroNet::TNeuroNet(const string &fileName) throw(invalid_argument, runtime_error)
{
    FILE *f = fopen(fileName.c_str(), "rb");
    if(f == NULL) throw invalid_argument("Can't open the file, check file name");

    try {
        // Read layers count
        int layersCount;
        fread(&layersCount, sizeof(int), 1, f);
        vector<int> layers(layersCount);
        // Чтение количества нейронов в слое
        for(int i = 0; i< layersCount; i++)
            fread(&layers[i], sizeof(int), 1, f);
        // Чтение Функции активации
        fread(&actFuncType, sizeof(ActivationFunctionType), 1, f);

        // Вызов конструктора
        construct(layers);

        // Чтение значений весовых коэффициентов и смещений
        for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
            for(int i=0; i < neuronCount[k]; i++) // Цикл по нейронам текущего слоя
            {
                for(int j = 0; j< neuronCount[k-1]; j++) // - по нейронам предыдущего слоя
                    // Чтение весов
                    fread(&w[k-1][i][j], sizeof(double), 1, f);
                // Чтение смещений (порогов)
                fread(&w0[k-1][i], sizeof(double), 1, f);
            }
    } catch(...) {
        throw runtime_error("Wrong file format!");
    }

    fclose(f);
}

void TNeuroNet::Save(const string &fileName)
{
    FILE *f = fopen(fileName.c_str(), "wb");
    if(f == NULL) return;

    // Write layers count
    int layersCount = neuronCount.size();
    fwrite(&layersCount, sizeof(int), 1, f);
    // Запись количества нейронов в слое
    for(int i = 0; i< layersCount; i++)
        fwrite(&neuronCount[i+1], sizeof(int), 1, f);
    // Запись Функции активации
    fwrite(&actFuncType, sizeof(ActivationFunctionType), 1, f);

    // Запись значений весовых коэффициентов и смещений
    for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
        for(int i = 0; i < neuronCount[k]; i++) // Цикл по нейронам текущего слоя
        {
            for(int j = 0; j < neuronCount[k-1]; j++) // - по нейронам предыдущего слоя
                // Запись весов
                fwrite(&w[k-1][i][j], sizeof(double), 1, f);
            // Запись смещений (порогов)
            fwrite(&w0[k-1][i], sizeof(double), 1, f);
        }
    fclose(f);
}

TNeuroNet::~TNeuroNet()
{
    // Если была использована функция обучения SCG,
    if(!scgMemAllocated)
    {// то очистить выделенную для этого метода память
        scgMemAllocated = true;
        SCGLearning(nullptr, nullptr, 0);
    }
    // Если была использована функция обучения GA
    if(!firstGACycle)
    {
        firstGACycle = true;
        delete ganl;
    }

    // Удаление массива значений синаптических весов нейронов
    for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
    {
        for(int i = 0; i < neuronCount[k]; i++) // Цикл по нейронам текущего слоя
        {
            delete[] w[k-1][i];
            // освобождение памяти, выделенной под вспомогательную переменную
            delete[] e_by_x[k-1][i];
        }
        delete[] w[k-1];
        delete[] w0[k-1];
        delete[] e_by_x[k-1];
    }
    delete[] w;
    delete[] w0;
    delete[] e_by_x;

    // Удаление массива выходов узлов нейронов
    for(int i = 0; i < innerNeurons.size() + 2; i++)
    {
        delete[] Y[i];
        delete[] inSum[i];
        delete[] delta[i];
        // освобождение памяти, выделенной под вспомогательную переменную
        delete[] sum_e_by_w[i];
    }
    delete[] Y;
    delete[] inSum;
    delete[] delta;
    delete[] sum_e_by_w;
}

void TNeuroNet::SetLearningParameters(double nu_Ext, double delta_max_Ext)
{
    // Задание значения скорости обучения
    nu = nu_Ext;
    // Задание максимальной ошибки обучения
    maxDelta = delta_max_Ext;
}

void TNeuroNet::SetActFunction(ActivationFunctionType type)
{
    // Set activation function
    actFuncType = type;

    switch (actFuncType) {
    default:
    case Logistic:
        fAct = &TNeuroNet::fLogistic;
        df = &TNeuroNet::dfLogistic;
        break;
    case HyperbolicTangent:
        fAct = &TNeuroNet::fTanh;
        df = &TNeuroNet::dfTanh;
        break;
    case Exponential:
        fAct = &TNeuroNet::fExp;
        df = &TNeuroNet::dfExp;
        break;
    }
}

void TNeuroNet::Init(double ExtMin, double ExtMax)
{
    wMin = ExtMin;
    wMax = ExtMax;
    Init();
}

void TNeuroNet::Init()
{
    // Инициализация массива значений синаптических весов нейронов
    for(int k=1; k<=innerNeurons.size()+1; k++) // Цикл по слоям
    {
        // Тестовый код
        //      if(k ==innerNeurons.size()+1 ) { WMin= -0.5; WMax = 0.5; }
        for(int i=0; i<neuronCount[k]; i++) // Цикл по нейронам текущего слоя
        {
            for(int j=0; j<neuronCount[k-1]; j++) // - по нейронам предыдущего слоя
                // Инициализация весов
                w[k-1][i][j]= rand()/(double)(RAND_MAX)*(wMax-wMin)+wMin;
            // Инициализация порогов
            w0[k-1][i] = rand()/(double)(RAND_MAX)*(wMax-wMin)+wMin;
        }
    }
    netInitialized = true;
}

void TNeuroNet::Update(double *X)
{
    // Расчет выходных сигналов первого (входного) слоя
    for(int i = 0; i < neuronCount[0]; i++)
        Y[0][i] = X[i];

    for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
        for(int i = 0; i < neuronCount[k]; i++) // Цикл по нейронам текущего слоя
        {
            double S = 0.0;
            for(int j = 0; j < neuronCount[k-1]; j++) // - по нейронам предыдущего слоя
                S += w[k-1][i][j]*Y[k-1][j];
            inSum[k][i] = S + w0[k-1][i]; // Вычисление взвешенной суммы входов
            Y[k][i] =  (this->*fAct)(inSum[k][i]);    // Вычисление выхода нейрона
            sum_e_by_w[k][i] = 0.0;
        }
}

// Функция обучения обратного распространения ошибки
double TNeuroNet::StdLearning(double **X, double **D, int PattCount)
{
    register double SSE = 0.0, deviation, error;

    for(int m = 0; m < PattCount; m++)
    {
        int index = m;
        // Расчет выхода сети (Forward Propagation)
        Update(X[index]);

        // ------------------- Обучение сети ----------------------------------
        // Настройка весовых коэффициентов нейронов выходного слоя
        for(int i=0; i<neuronCount[innerNeurons.size()+1]; i++)
        {
            int k = innerNeurons.size()+1;
            deviation = D[index][i] - Y[k][i];

            // Предотвращение переобучения
            if (fabs(deviation) <= maxDelta)
                continue;

            SSE += deviation*deviation; // Вычисление суммарной квадратичной
            // ошибки (Sum Squared Error) обучения
            error = deviation * df(k, i);
            delta[k][i] = error;
            for(int j=0; j<neuronCount[innerNeurons.size()]; j++)
            {
                // Вычисление ошибки сети
                sum_e_by_w[k-1][j] += error*w[k-1][i][j];
                // Адаптация весов и порогов
                w[k-1][i][j] += nu*delta[k][i]*Y[k-1][j]; // Адаптация весов
                w0[k-1][i]   += nu*delta[k][i]*Y[k-1][j]; // Адаптация порогов
            }
        }
        // Настройка весовых коэффициентов нейронов скрытых слоев
        for(int k=innerNeurons.size(); k>0; k--)
            for(int i=0; i<neuronCount[k]; i++)
            {
                error = sum_e_by_w[k][i]*df(k, i);
                delta[k][i] = error;

                for(int j=0; j<neuronCount[k-1]; j++)
                {
                    // Вычисление ошибки сети
                    sum_e_by_w[k-1][j] += error*w[k-1][i][j];
                    // Адаптация весов и порогов
                    w[k-1][i][j] += nu*delta[k][i]*Y[k-1][j]; // Адаптация весов
                    w0[k-1][i]   += nu*delta[k][i]*Y[k-1][j]; // Адаптация порогов
                }
            }
    }
    return(SSE);
}

// Тестирование сети
double TNeuroNet::Test(double **X, double **D, int PattCount,
                       double *MaxError)
{
    double SSE = 0.0;
    bool CalcMaxError = MaxError != nullptr;
    if(CalcMaxError)
        *MaxError = 0.0;
    for(int m=0; m<PattCount; m++)
    {
        Update(X[m]);
        // Вычисление суммарной квадратичной ошибки (Sum Squared Error)
        int k = innerNeurons.size()+1;
        double Error = 0.0;
        for(int i=0; i<neuronCount[k]; i++)
        {
            double multiplier = Y[k][i] - D[m][i];
            Error += multiplier * multiplier;
            SSE += Error;
        }
        // Поиск максимального отклонения
        if(CalcMaxError)
            if(fabs(Error) > *MaxError)
                *MaxError = Error;
    }
    return(SSE);
}

void TNeuroNet::Print(char *filename, int Precission)
{
    FILE *f = fopen(filename, "w");
    double MinW = 0.0, MaxW = 0.0,
            MinW0 = 0.0, MaxW0 = 0.0;
    for(int k=1; k<innerNeurons.size()+1; k++) // Цикл по слоям
    {
        // Печать выходов нейронов
        for(int i=0; i<neuronCount[k-1]; i++)
            fprintf(f, "%6.3lf ", Y[k-1][i]);
        fprintf(f, "\n-------------------------\n");
        for(int i=0; i<neuronCount[k]; i++)
        {
            fprintf(f, "w%i ", i);
            for(int j=0; j<neuronCount[k-1]; j++)
            {
                fprintf(f, "%i: %6.3lf ", j, w[k-1][i][j]);
                if(MinW > w[k-1][i][j])
                    MinW = w[k-1][i][j];
                if(MaxW < w[k-1][i][j])
                    MaxW = w[k-1][i][j];
            }
            fprintf(f, "\n");
            fprintf(f, "w0 %i: %6.3lf\n", i, w0[k-1][i]);
            if(MinW0 > w0[k-1][i])
                MinW0 = w0[k-1][i];
            if(MaxW0 < w0[k-1][i])
                MaxW0 = w0[k-1][i];
        }
        fprintf(f, "-------------------------\n");
    }
    // Печать выходов нейронов выходного слоя
    for(int i=0; i<neuronCount[innerNeurons.size()+1]; i++)
        fprintf(f, "%6.3lf ", Y[innerNeurons.size()+1][i]);
    fprintf(f, "\n");
    // Печать мин и макс значения весов и смещений
    fprintf(f, "-------------------------\n");
    fprintf(f, " MinW: %6.3lf     MinW0: %6.3lf\n", MinW, MinW0);
    fprintf(f, " MaxW: %6.3lf     MaxW0: %6.3lf\n", MaxW, MaxW0);

    fclose(f);
}


#define SIGMA_1_SETPOINT 1E-4
#define LAMBDA_1_SETPOINT 1E-6
#define TOLERANCE_SETPOINT 1E-8

// Переменные функции SCGLearning
static int vector_size;
static double* *gradient;

// Функция обучения методом сопряженных градиентов
double TNeuroNet::SCGLearning(TDataSource *data)
{
    static int cc; // (Cycle Count)
    static bool restart_alg = false, stop_alg = false, success;

    static int count_under_tol; /* Алгоритм останавливается после 3-х
                               последовательных шагов ниже допуска
                               (under_tolerance). Это счетчик. */
    static double scg_delta, lambda, lambda_bar,
            norm_of_p_2, norm_of_rk, SSE, old_error;

    static double* *weights;

    static double *old_weights, *old_gradient, *p, *r, *step;

    int i;
    bool start_alg = netInitialized, under_tolerance;
    double sigma, mu, alpha, grand_delta, beta;  // Переменные алгоритма
    static double sigma_1, lambda_1, tolerance;  // Параметры обучения
    netInitialized = false;

    // Выделение памяти под вспомогательные переменные
    if(scgMemAllocated)
    {
        scgMemAllocated = false;
        vector_size = CalculateVectorSize();
        if(weights == nullptr)
        {
            weights = new double *[vector_size];
            gradient = new double *[vector_size];
            old_weights = new double [vector_size];
            old_gradient = new double [vector_size];
            p = new double [vector_size];
            r = new double [vector_size];
            step = new double [vector_size];

            int vector_index = 0;
            // Сохранить адреса весов в векторе весов
            for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
                for(int i = 0; i < neuronCount[k]; i++) // - по нейронам текущего слоя
                {
                    gradient[vector_index] = &delta[k][i];
                    weights[vector_index++] = &w0[k-1][i];
                }
            for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
                for(int i = 0; i < neuronCount[k]; i++) // - по нейронам текущего слоя
                    for(int j = 0; j < neuronCount[k-1]; j++) // - по нейронам предыдущего
                    {
                        gradient[vector_index] = &e_by_x[k-1][i][j];
                        weights[vector_index++] = &w[k-1][i][j];
                    }
            // Инициализация параметров обучения
            sigma_1 = SIGMA_1_SETPOINT;
            lambda_1 = LAMBDA_1_SETPOINT;
            tolerance = TOLERANCE_SETPOINT;
        }
        else
        {
            delete[] weights;
            delete[] gradient;
            delete[] old_weights;
            delete[] old_gradient;
            delete[] p;
            delete[] r;
            delete[] step;

            weights = nullptr; // Для последующего запуска
            return(0.0);
        }
    }
    // Инициализация начальных параметров обучения
    if(start_alg || restart_alg)
    {
        lambda = lambda_1;
        lambda_bar = 0.0;
        success = true;
        cc=1;
        restart_alg = false;
        count_under_tol = 0;
    }
    // Инициализация векторов p и r
    if(start_alg)
    {
        SSE = CalculateGradient(X, D, PattCount); // Расчет вектора градиента
        // Копировать градиент в p и r
        for(i=0; i<vector_size; i++)
        {
            p[i] = - *gradient[i];
            r[i] = p[i];
        }
        norm_of_rk = sqrt(square_of_norm(r, vector_size));
        start_alg = false;
        stop_alg = false;
    }

    if(stop_alg)
        return(SSE);

    // -------------------- Главная часть алгоритма SCG --------------------
    // Вычислить информацию второго порядка
    if(success)
    {
        norm_of_p_2 = square_of_norm(p, vector_size);
        if(norm_of_p_2 <= tolerance * tolerance)
        {
            stop_alg = true;
            cc++;
            return(SSE);
        }
        sigma = sigma_1/sqrt(norm_of_p_2);
        // Чтобы вычислить новую точку, нужен новый градиент
        for(i=0; i<vector_size; i++)
        {
            old_gradient[i] = *gradient[i];
            old_weights[i] = *weights[i];
        }
        old_error = SSE;
        // смещение на новую точку в весовом пространстве
        for(i=0; i<vector_size; i++)
            *weights[i] += sigma*p[i];
        // вычислить новый градиент
        SSE = CalculateGradient(X, D, PattCount);
        // вычислить размер шага
        for(i=0; i<vector_size; i++)
            step[i] = (*gradient[i] - old_gradient[i])/sigma;
        scg_delta = product_of_xt_by_y(p, step, vector_size);
    }
    // масштабирование scg_delta
    scg_delta += (lambda - lambda_bar) * norm_of_p_2;
    if(scg_delta <= 0.0) // Сделать матрицу Гессиана положительно определенной
    {
        lambda_bar = 2.0 * (lambda - scg_delta/norm_of_p_2);
        scg_delta = - scg_delta + lambda * norm_of_p_2;
        lambda = lambda_bar;
    }
    // вычислить размер шага
    mu = product_of_xt_by_y(p, r, vector_size);
    alpha = mu/scg_delta;

    // вычислить параметр сравнения

    /* необходимо вычислить новый градиент но на этот раз не нужно сохранять
      предыдущие значения (ни были нужны только для аппроксимации матрицы
      Гессиана) */

    for(i=0; i<vector_size; i++)
        *weights[i] = old_weights[i] + alpha*p[i];
    SSE = CalculateGradient(X, D, PattCount);
    grand_delta = 2.0*scg_delta*(old_error - SSE)/(mu*mu);
    if(grand_delta >= 0) // удачное уменьшение ошибки может быть выполнено
    {
        double r_sum = 0.0, // произведение r(cc+1) на r(cc)
                swap;
        under_tolerance = 2.0 * fabs(old_error - SSE) <=
                tolerance*(fabs(old_error) + fabs(SSE) + 1e-10);
        // мы уже в w(cc+1) весовом пространстве
        // вычислить |r(cc)| перед измением r(cc) в r(cc+1)
        norm_of_rk = sqrt(square_of_norm(r, vector_size));
        // теперь r < - r(cc+1)
        for(i=0; i<vector_size; i++)
        {
            swap = -1.0* *gradient[i];
            r_sum += swap * r[i];
            r[i] = swap;
        }
        lambda_bar = 0.0;
        success = true;
        if(cc >= vector_size)
        {
            restart_alg = true;
            for(i=0; i<vector_size; i++)
                p[i] = r[i];
        }
        else
        { // вычислить новое сопряженное направление
            beta = (square_of_norm(r, vector_size) - r_sum)/mu;
            for(i=0; i<vector_size; i++)
                p[i] = r[i] + beta*p[i];
            restart_alg = false;
        }
        if(grand_delta >= 0.75)
            lambda /= 4.0;
    }
    else // уменьшить ошибку нельзя
    {
        under_tolerance = false;
        // необходимо вернуться к w(cc), т.к. w(cc)+alpha*p(cc) не лучше
        for(i=0; i<vector_size; i++)
            *weights[i] = old_weights[i];
        SSE = old_error;
        lambda_bar = lambda;
        success = false;
    }
    if(grand_delta < 0.25)
        lambda += scg_delta*(1 - grand_delta)/norm_of_p_2;
    // для предотвращения ошибок при вычислениии чисел с плавающей запятой
    // (lambda может быть очень большой)
    if(lambda > MAXFLOAT)
        lambda = MAXFLOAT;
    // алгоритм останавливается после 3-х последовательных шагов ниже допуска
    // (under tolerance)
    if(under_tolerance)
        count_under_tol++;
    else
        count_under_tol = 0;
    stop_alg = (count_under_tol > 2) || (norm_of_rk <= tolerance);
    if(stop_alg)
        count_under_tol = 0; // в случае, если пользователь хочет попробовать с
    // меньшим значением tolerance
    cc++;

    return(SSE);
}

void TNeuroNet::ClearDeltas()    // Обнуление ошибки обучения
{
    int k, i, j;
    for(k=1; k<=innerNeurons.size()+1; k++) // Цикл по слоям
        for(i=0; i<neuronCount[k]; i++) // Цикл по нейронам текущего слоя
        {
            for(j=0; j<neuronCount[k-1]; j++) // - по нейронам предыдущего слоя
                e_by_x[k-1][i][j] = 0.0;
            delta[k][i] =  0.0;
        }
}

double TNeuroNet::product_of_xt_by_y(double *x, double *y, int array_size)
{
    double s = 0.0;

    for(int i=0; i<array_size; i++)
        s += x[i]*y[i];

    return(s);
}

double TNeuroNet::square_of_norm(double *x, int array_size)
{
    return(product_of_xt_by_y(x, x, array_size));
}

double TNeuroNet::CalculateGradient(double **X, double **D, int PattCount)
{
    double SSE = 0.0;
    ClearDeltas();  // Очистка параметра, описывающего ошибку сети, для
                    // последующего запуска функции обратного распространения

    for(int m=0; m<PattCount; m++)
    {
        int index = m;
        // Расчет выхода сети (Forward Propagation)
        Update(X[index]);
        SSE += PropagateNetBatchBackward(X[index], D[index]);
    }
    for(int i=0; i<vector_size; i++)
        *gradient[i] = - 2.0* *gradient[i];

    return(SSE);
}

double TNeuroNet::PropagateNetBatchBackward(double *X, double *D)
{
    register double SSE = 0.0, deviation, error;

    // ------------ Расчет параметра, описывающего ошибку сети ---------------
    // Расчет параметра для нейронов выходного слоя
    for(int i=0; i<neuronCount[innerNeurons.size()+1]; i++)
    {
        int k = innerNeurons.size() + 1;
        deviation = D[i] - Y[k][i];

        // Предотвращение переобучения
        if (fabs(deviation) <= maxDelta)
            continue;

        SSE += deviation*deviation;   // Вычисление суммарной квадратичной
        // ошибки (Sum Squared Error) обучения
        error = deviation * (this->*df)(k, i);
        delta[k][i] += error;
        for(int j=0; j<neuronCount[innerNeurons.size()]; j++)
        {
            sum_e_by_w[k-1][j] += error*w[k-1][i][j];
            e_by_x[k-1][i][j]  += error*Y[k-1][j];
        }
    }
    // Расчет параметра для нейронов скрытых слоев
    for(int k = innerNeurons.size(); k > 0; k--)
        for(int i = 0; i < neuronCount[k]; i++)
        {
            error = sum_e_by_w[k][i] * (this->*df)(k, i);
            delta[k][i] += error;
            for(int j=0; j<neuronCount[k-1]; j++)
            {
                sum_e_by_w[k-1][j] += error*w[k-1][i][j];
                e_by_x[k-1][i][j]  += error*Y[k-1][j];
            }
        }
    return(SSE);
}

int TNeuroNet::CalculateVectorSize()
{
    int vector_size = 0;

    for(int k = 1; k <= innerNeurons.size() + 1; k++) // Цикл по слоям
        vector_size += neuronCount[k]*(neuronCount[k-1] + 1);

    return(vector_size);
}

double TNeuroNet::GetInput(int i)
{
    return(in[i]);
}

double TNeuroNet::GetOutput(int i)
{
    return(out[i]);
}

double TNeuroNet::GALearning(double **X, double **D, int PattCount)
{
    static int iterNo;
    if(firstGACycle)
    {
        firstGACycle = false;
        ganl = new TGANeuroLearn(this, X, D, PattCount);
        iterNo = 1;
    }
    double Fitness = ganl->OptimizationCycle(iterNo++);
    double SSE = 1.0 / Fitness - 1.0;

    return(SSE);
}

TGANeuroLearn::TGANeuroLearn(TNeuroNet *ExtNet, double **XExt, double **DExt,
                             int PattCountExt) :
    TGeneticAlgorithm(ExtNet->CalculateVectorSize())
{
    Net = ExtNet;
    X = XExt;
    D = DExt;
    PattCount = PattCountExt;

    CalcVectorSize();

    // Создать популяцию особей
    for(int i=0; i<POPULATION_SIZE; i++)
    {
        Net->Init();
        Parents[i]->fillVectorData(vector);
        CalcCromosomeFitness(Parents[i]);

        Net->Init();
        Offsprings[i]->fillVectorData(vector);
        CalcCromosomeFitness(Offsprings[i]);

        AllPopulation[i] = Parents[i];
        AllPopulation[i+POPULATION_SIZE] = Offsprings[i];
    }
    // Отбор лучших особей в родительскую популяцию
    AvgFitness = NewGeneration1();
}

void TGANeuroLearn::CalcVectorSize()
{
    vector_size = Net->CalculateVectorSize();
    vector = new double *[vector_size];

    int vector_index = 0;
    // Сохранить адреса смещений в векторе параметров
    for(int k=1; k<=Net->innerNeurons.size()+1; k++) // Цикл по слоям
        for(int i=0; i<Net->neuronCount[k]; i++) // - по нейронам текущего слоя
            vector[vector_index++] = &(Net->w0[k-1][i]);
    // Сохранить адреса весов в векторе параметров
    for(int k=1; k<=Net->innerNeurons.size()+1; k++) // Цикл по слоям
        for(int i=0; i<Net->neuronCount[k]; i++) // - по нейронам текущего слоя
            for(int j=0; j<Net->neuronCount[k-1]; j++) // - по нейронам предыдущего
                vector[vector_index++] = &(Net->w[k-1][i][j]);
}

void TGANeuroLearn::CalcCromosomeFitness(TChromosome* C)
{
    for(int j=0; j<vector_size; j++)
        *vector[j] = C->geneValue(j);

    double SSE = Net->Test(X, D, PattCount, nullptr);
    C->Fitness = 1.0 / (1.0 + SSE);
}
