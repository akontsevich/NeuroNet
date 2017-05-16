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

#include "tgeneticalgorithm.h"

using namespace std;

class TLearnPattern;
class TGANeuroLearn;

///< first - Min, second - Max
struct MinMax : public pair<double, double>
{
    using pair<double, double>::pair;

    double& Min() { return first; }
    double Min() const { return first; }
    double& Max() { return second; }
    double Max() const { return second; }
};

class TNeuroNet
{
    friend class TGANeuroLearn;

public:
    ///< Learning function pointer type
    typedef double (TNeuroNet::* LearningFunction)(TLearnPattern &pattern);

    enum ActivationFunctionType {
        Logistic,
        HyperbolicTangent,
        Exponential,
        // -----------
        FunctionCount
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

    ///< Neuro net learning function
    inline double Learning(TLearnPattern &pattern) {
        return (this->*mLearningFunction)(pattern);
    }

    void Save(const string &fileName);  ///< Save neuro net to a file
    void Print(const string &fileName, int Precission);

    /// Установить параметры обучения
    void SetLearningParameters(double nu, double delta_max);
    /// Задать активационную функцию
    void SetActFunction(ActivationFunctionType type);
    /// Инициализация значений весов и порогов сети случайными значениями
    void Init(double WMin, double WMax);
    /*!
     *  \brief Calculates neuro net output for input vector X
     *  \param X vector accessible by operator[] (i.e double*, TPatternRow,
     * vector<double>, etc)
     */
    template<typename T> void Update(T X);
    ///< Returns SSE for whole learning pattern
    double PatternSSE(TLearnPattern &pattern, double *MaxError = nullptr);
    ///< Neuro net testing: returns SSE for test pattern
    double Test(TLearnPattern &pattern, double *MaxError = nullptr);
    /// Функции возвращающие значения входа и выхода нейросети
    double Input(int i)  { return in[i]; }  ///< Neuro net input
    double Output(int i) { return out[i]; } ///< Neuro net output
    double operator[](int i) { return out[i]; } ///< Neuro net output

    const MinMax inMinMax() { return mInMinMax[actFuncType]; }
    const MinMax outMinMax() { return { (this->*fAct)(mInMinMax[actFuncType].Min()),
                                        (this->*fAct)(mInMinMax[actFuncType].Max()) }; }

protected:
    // ------------------------ Learning functions ----------------------------
    /// Back propagation error learing function (gradient method)
    double StdLearning(TLearnPattern &pattern);
    /// SCG - scaled conjugate gradient learing function
    double SCGLearning(TLearnPattern &pattern);
    /// ANN learing function with genetic algorithm
    double GALearning(TLearnPattern &pattern);

private:
    LearningFunction mLearningFunction = &TNeuroNet::StdLearning;

    void Construct(const vector<int> &topology);

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
    double fExp(double x) { return exp(-0.5 * x*x); }

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
    double CalculateGradient(TLearnPattern &pattern);
    /*!
     * \brief PropagateNetBatchBackward
     * \param D desired neuro net output vector
     * \return Sum Square Error (SSE)
     */
    template<typename T> double PropagateNetBatchBackward(T D);
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
    // Min/max values for input/output learning data normalization depends on
    // activation function: should be taken into account for good ANN approximation
    const MinMax mInMinMax[FunctionCount] = {{-1, 1}, {-1, 1}, {0, 1}};

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

    /*!
     * \brief TGANeuroLearn
     * \param mNet За основу берем сеть, которую необходимо обучить, но для обучения
     * создаем дополнительно популяцию из POPULATION_SIZE хромосом-сетей
     * \param pattern Learning pattern (normalized data)
     */
    TGANeuroLearn(TNeuroNet *net, TLearnPattern &pattern);
    virtual ~TGANeuroLearn() {}

protected:
    virtual void CalcCromosomeFitness(TChromosome*);
    virtual void InitParameters()   { mNet->Init(); }
    virtual void LinkParameters();

private:
    TNeuroNet *mNet;
    TLearnPattern &mPattern;
};

#endif // TNEURONET_H
