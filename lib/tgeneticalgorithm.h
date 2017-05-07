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
#define BETA            0.0873

// Хромосома или особь
class TChromosome
{
friend class TGeneticAlgorithm;
friend class TGANeuroLearn;

public:
    TChromosome(int n);
    ~TChromosome();
    double geneValue(int No);
    void Print();

private: // Comment this for debugging ;-)

    void fillVectorData(double *vector);
    void fillVectorData(double **vector);

    double *data;
    double *al, *si, tau, tau1;
    int Size;              // Количество ген в хромосоме
    double Min, Max;         // Начальные мин и макс значения хромосомы

    double Fitness;              // Приспособленность особи
};

/*!
 * \brief The TGeneticAlgorithm class - real value coding genetic algorithm
 */
class TGeneticAlgorithm
{
public:
    TGeneticAlgorithm(int vector_size);
    virtual ~TGeneticAlgorithm();

    // Вызывается из функции обучения ИНС,
    // возвращает приспособленность наиболее приспособленной особи
    double OptimizationCycle(int iteration = -1);

protected: // Comment this for debugging ;-)
    inline void Crossover(int cNo1,     // Номера хромосом в списке
                          int cNo2,     // для скрещивания
                          int ChildNo); // Номер потомка
    void Copy(TChromosome *Dest,        // Копирование хромосомы без изменений
              TChromosome *Source);

    // Оператор мутации
    inline void Mutation(TChromosome *Offspring, TChromosome *Parent);
    inline void Mutation1(TChromosome *Offspring, TChromosome *Parent);

    // Оператор селекции - отбор родителей
    inline void Selection(int *p1, int *p2 = 0); // рулетка (Fitness > 0)

    double *GP;
    inline void RoulettePreparation(); // Рассчет интервалов для рулетки

    // Создание новой популяции
    double NewGeneration1();   // Выбор лучших из всех особей
    double NewGeneration2();   // Полная смена популяции

    double AvgFitness;           // Средняя приспособленность популяции

    TChromosome **Parents;      // Хромосомы-родители
    TChromosome **Offsprings;   // Хромосомы-потомки
    TChromosome **AllPopulation;// Вся популяция (для механизма селекции)

    inline void QSort(TChromosome **array, int First, int Last);

    // Функция получения размера оптимизируемого вектора (индивидуальная для
    // каждого типа решаемых задач)
    virtual void CalcVectorSize() {}
    // Функция рассчета приспособленности особи
    virtual void CalcCromosomeFitness(TChromosome*) {}

    double **vector;             // Вектор параметров, значения которых
                                // необходимо оптимизировать
    int vector_size;            // Размер вектора
    int ChildAvgCountSum;
    double nu4all;
};

#endif // TGENETICALGORITHM_H
