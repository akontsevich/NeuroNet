/*!
    @file
    @brief Contains TNeuroNet class definition

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/

#ifndef TNEURONET_H
#define TNEURONET_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include "tdatasource.h"
#include "tgeneticalgorithm.h"

using namespace std;

class TGANeuroLearn;

class TNeuroNet
{
    friend class TGANeuroLearn;

public:
    ///< Learning function pointer type
    typedef double (TNeuroNet::* LearningFunction)(TDataSource *);

    enum ActivationFunctionType {
        Logistic,
        HyperbolicTangent,
        Exponential
    };

    /*!
     * \brief TNeuroNet
     * \param topology first element - input layer neuron count,
     * last element - output layer neuron count,
     * intermediate element(s) - inner layer(s) neurons count (so at least 3
     * elements should be present)
     * \throws invalid_argument throws if topology.size() < 3
     * or any topology element <= 0
     */
    TNeuroNet(const vector<int> &topology) throw(invalid_argument);
    ///< Construct ANN from presaved binary file
    TNeuroNet(const string &fileName) throw(invalid_argument, runtime_error);
    ~TNeuroNet();

    inline double Learning(TDataSource *data) { return (this->*mLearningFunction)(data); }

    void Save(const string &fileName);  ///< Save neuro net to a file
    void Print(char *filename, int Precission);

    /// Установить параметры обучения
    void SetLearningParameters(double nu, double delta_max);
    /// Задать активационную функцию
    void SetActFunction(ActivationFunctionType type);
    /// Инициализация значений весов и порогов сети случайными значениями
    void Init(double WMin, double WMax);
    void Update(double *X); // Расчет выхода сети для входного вектора
                            // сигналов X
    /// Neuro net testing
    double Test(double **X, double **D, int PattCount, double *MaxError = NULL);
    /// Функции возвращающие значения входа и выхода нейросети
    double GetInput(int);
    double GetOutput(int);

protected:
    // ------------------------ Learning functions ----------------------------
    /// Back propagation error learing function (gradient method)
    double StdLearning(TDataSource *data);
    /// SCG - scaled conjugate gradient learing function
    double SCGLearning(TDataSource *data);
    /// ANN learing function with genetic algorithm
    double GALearning(TDataSource *data);

private:
    LearningFunction mLearningFunction = &TNeuroNet::StdLearning;

    void construct(const vector<int> &topology);

    /// Activation function
    double (TNeuroNet::*fAct)(double x) = &TNeuroNet::fLogistic;
    /*!
     * \brief df Derivative of activation function
     * \param k for k-th layer
     * \param i ant its i-th neuron
     * \return
     */
    double (TNeuroNet::*df)(int k, int i) = &TNeuroNet::dfLogistic;

    // Individual activation functions
    /// Logistic activation function
    double fLogistic(double x) { return(1.0 / (1.0 + exp(-x))); }
    /// Hyperbolic tangent activation function
    double fTanh(double x) { return tanh(x); }
    /// Exponential activation function
    double fExp(double x) { return exp(-0.5*x*x); }

    // Individual derivatives of activation functions
    /// Logistic activation function
    double dfLogistic(int k, int i) { double OUT = Y[k][i];
                                      return OUT * (1.0 - OUT); }
    /// Hyperbolic tangent activation function
    double dfTanh(int k, int i) { double OUT = Y[k][i];
                                  return 1.0 - OUT*OUT; }
    /// Exponential activation function
    double dfExp(int k, int i) { double OUT = inSum[k][i];
                                 return -OUT * exp(-0.5 * OUT*OUT); }

    /// Initialization function
    void Init();

    // SCG helper methods
    void ClearDeltas(void);    // Обнуление ошибки обучения
    double CalculateGradient(TDataSource *data);
    double PropagateNetBatchBackward(double *X, double *D);
    double product_of_xt_by_y(double *x, double *y, int array_size);
    double square_of_norm(double *x, int array_size);
    int CalculateVectorSize(void);

    // ------------------------ Class data --------------------------
    vector<int> neuronCount,    // All layers neurons count
                innerNeurons;   // Inner neurons count
    double *in, *out; // Указатели на вход и выход нейросети

    double **Y,       // Выходы узлов нейронов
          ***w,       // Значения весов синаптических связей
          **w0;       // смещения

    double **inSum,     // Сумма входов нейрона
           **delta;   // Ошибка обучения

    // Вспомогательные переменные (дополнительная информация)
    double ***e_by_x; // Вспомогательная переменная - произведение ошибки выхода
                      // нейрона (E)rror на значение соответствующего входа (X)
    double **sum_e_by_w; // Вспомогательная переменная -  сумма произведений
                         // ошибки выхода нейрона (E)rror на соответствующий
                         // вес (w) (применяется в функции обратного
                         // распространения ошибки)
    double wMin,          // Параметры инициализации сети: макс и мин значения
           wMax;          // весов и смещений

    ActivationFunctionType actFuncType;  // Activation function type

    // Learning parameters
    double nu,           // Скорость обучения (learning rate)
           maxDelta;    // Максимальная ошибка обучения: разница между обучающим
                        // значением и выходом сети (распространяется назад как
                        // 0)

    // Вспомогательные переменные (флаги) метода SCG
    bool scgMemAllocated, netInitialized;

    // Вспомогательные переменные, флаги метода GA
    TGANeuroLearn *ganl;
    bool firstGACycle;
 };

/*!
 * \brief The TGANeuroLearn class
 */
class TGANeuroLearn : public TGeneticAlgorithm
{
public:
    TGANeuroLearn(TNeuroNet *Net, // За основу берем сеть, которую
                                   // необходимо обучить, но для обучения
                                   // создаем дополнительно популяцию из
                                   // POPULATION_SIZE хромосом-сетей
                  double **X,
                  double **D,
                  int PattCount);

    virtual ~TGANeuroLearn() {}

protected:
    double **X, **D; // Векторы обучающих шаблонов
    int PattCount;  // Размер обучающих шаблонов

    virtual void CalcVectorSize();
    virtual void CalcCromosomeFitness(TChromosome*);

    TNeuroNet *Net;
};

#endif // TNEURONET_H
