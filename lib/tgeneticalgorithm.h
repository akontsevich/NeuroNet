/*!
    @file
    @brief Contains TGeneticAlgorithm class definition

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/
#ifndef TGENETICALGORITHM_H
#define TGENETICALGORITHM_H


#define POPULATION_SIZE 100
// Доля потомков полученных посредством мутации
#define P_MUTATION      0.90    // Вероятность выполнения операции мутации

#define P_LOWFITNESS    0.10    // Вероятность переноса потомка в новую
                                // популяцию при значении приспособленности
                                // < AvgFitness
#define NU_INIT         0.001

#include <vector>

using namespace std;

// Хромосома или особь
class TChromosome
{
public:
    TChromosome(size_t size);
    ~TChromosome();
    double& operator [](int no) { return data[no]; }

    size_t size() { return data.size(); }   ///< Genes count in chromosome
    void setFitness(double fitness) { mFitness = fitness; }
    double fitness() { return mFitness; }

    void fillVectorData(double *vector);
    void fillVectorData(double **vector);

    void Print();

private:
    vector<double> data;
    double Min, Max;         // Начальные мин и макс значения хромосомы
    double mFitness;              // Приспособленность особи
};

/*!
 * \brief The TGeneticAlgorithm class - real value coding genetic algorithm
 */
class TGeneticAlgorithm
{
public:
    TGeneticAlgorithm(size_t vector_size);
    virtual ~TGeneticAlgorithm();

    // Вызывается из функции обучения ИНС,
    // возвращает приспособленность наиболее приспособленной особи
    double OptimizationCycle(int iteration = -1);
    ///< Optimized parameters vector size
    size_t vectorSize()     { return mParameters.size(); }

protected:
    // Функция рассчета приспособленности особи
    virtual void CalcCromosomeFitness(TChromosome*) = 0;
    ///< Initialize optimized parameters
    /// \note called for each chromosome initialization, should differ for each call
    virtual void InitParameters() = 0;
    ///< Link optimized subject field parameters to class parameters
    virtual void LinkParameters() = 0;
    double*& operator [](int i) { return mParameters[i]; }

private:
    void GeneratePopulation(); ///< Population initialization method

    inline void Crossover(int cNo1,     // Номера хромосом в списке
                          int cNo2,     // для скрещивания
                          int ChildNo); // Номер потомка
    ///< Copy chromosome without modifications
    void Copy(TChromosome *Dest, TChromosome *Source);

    // Оператор мутации
    inline void Mutation(TChromosome &Offspring, TChromosome &Parent);

    // Оператор селекции - отбор родителей
    inline void Selection(int *p1, int *p2 = 0); // рулетка (Fitness > 0)

    inline void RoulettePreparation(); ///< Calc intervals for "roulette" method

    // New population creation functions
    ///< Population creation function: Select best from all chromosomes
    double NewGenerationBest();
    ///< Population creation function: Full Generation Substitution
    double NewGenerationFullSubstitution();

    inline void QSort(TChromosome **array, int First, int Last);

    // Генератор случайных чисел распределения Коши
    double cauchy_gen();
    // Генератор появления события с вероятностью P
    bool random_event_gen(double P);

    double AvgFitness;           // Средняя приспособленность популяции

    TChromosome **Parents;      // Хромосомы-родители
    TChromosome **Offsprings;   // Хромосомы-потомки
    TChromosome **AllPopulation;// Вся популяция (для механизма селекции)

    double *GP;             ///< Intervals for "roulette" selection method
    vector<double *>mParameters;    ///< Parameters vector to be optimized
    int ChildAvgCountSum;
    double nu4all;

    bool mPopulationInitialized = false;
};

#endif // TGENETICALGORITHM_H
