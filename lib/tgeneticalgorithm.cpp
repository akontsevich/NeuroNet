/*!
    @file
    @brief Contains TGeneticAlgorithm class implementation

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/

#include <iostream>
#include <algorithm>

#include "tgeneticalgorithm.h"

using namespace std;

void TChromosome::fillVectorData(double **vector)
{
    for(size_t j = 0; j < size(); j++)
        data[j] = *(vector[j]);
}

void TChromosome::fillVectorData(double *vector)
{
    for(size_t j = 0; j < size(); j++)
        data[j] = vector[j];
}

TChromosome::TChromosome(size_t size)
{
    data = vector<double>(size);
}

TChromosome::~TChromosome()
{
}

void TChromosome::Print()
{
    printf("(%5.3f", data[0]);
    for(size_t j = 1; j < size(); j++)
        printf(", %5.3f", data[j]);
    printf(")\n");
}

TGeneticAlgorithm::TGeneticAlgorithm(size_t vector_size)
{
    mParameters = vector<double *>(vector_size);

    // Создание хромосом-родителей
    Parents = new TChromosome *[POPULATION_SIZE];
    // Создание хромосом-потомков
    Offsprings = new TChromosome *[POPULATION_SIZE];

    AllPopulation = new TChromosome *[2*POPULATION_SIZE];

    for(int i = 0; i < POPULATION_SIZE; i++){
        Parents[i] = new TChromosome(vector_size);
        Offsprings[i] = new TChromosome(vector_size);
    }
    // Создание интервалов для селекции методом "рулетки"
    GP = new double [POPULATION_SIZE+1];

    ChildAvgCountSum = 0;
    nu4all = NU_INIT;

    // Инициализация генератора случайных чисел
    time_t t;
    srand((int) time(&t));
}

void TGeneticAlgorithm::GeneratePopulation()
{
    for(int i = 0; i < POPULATION_SIZE; i++)
    {
        InitParameters();
        Parents[i]->fillVectorData(mParameters.data());
        CalcCromosomeFitness(Parents[i]);

        InitParameters();
        Offsprings[i]->fillVectorData(mParameters.data());
        CalcCromosomeFitness(Offsprings[i]);

        AllPopulation[i] = Parents[i];
        AllPopulation[i + POPULATION_SIZE] = Offsprings[i];
    }
    // Отбор лучших особей в родительскую популяцию
    AvgFitness = NewGenerationBest();

    mPopulationInitialized = true;
}

TGeneticAlgorithm::~TGeneticAlgorithm()
{
    for(int i = 0; i < POPULATION_SIZE; i++)
    {
        delete Parents[i];
        delete Offsprings[i];
    }
    delete[] Parents;
    delete[] Offsprings;
    delete[] AllPopulation;

    delete[] GP;
}

void TGeneticAlgorithm::Copy(TChromosome *Dest, TChromosome *Source)
{
    *Dest = *Source;    // C++ copy
}

void TGeneticAlgorithm::Crossover(int pn1, int pn2, int Child)
{
    // Конкатенация компонентов векторов родителей
    //    int CrossoverPoint = (int)((double)rand()/(double)(RAND_MAX)*vector_size);
    for(size_t j = 0; j < vectorSize(); j++)
        // одноточечное скрещивание
        //        if(j < CrossoverPoint)
        // многоточечное скрещивание
        //        if(j%2 == 0)
        // Случайный обмен хромосомами
        if(random_event_gen(0.5))
            (*Offsprings[Child])[j] = (*Parents[pn1])[j];
        else
            (*Offsprings[Child])[j] = (*Parents[pn2])[j];
    // Арифметическое среднее
    //        Offsprings[Child]->data[j] = 0.5*(Parents[pn1]->data[j] +
    //                                          Parents[pn2]->data[j]);
}

void TGeneticAlgorithm::Mutation(TChromosome &Offspring, TChromosome &Parent)
{
    for(size_t j = 0; j < vectorSize(); j++){
        if(random_event_gen(0.5))
            Offspring[j] = Parent[j] + nu4all * cauchy_gen();
    }
}

// Вызывается из функции обучения ИНС
// Критерий останова пока является макс количество циклов
double TGeneticAlgorithm::OptimizationCycle(int iter)
{
    // Lazy initialization
    if(!mPopulationInitialized) GeneratePopulation();

    RoulettePreparation();
    // Цикл создания новой популяции хромосом
    int ChildAvgCount = 0;
    for(int i=0; i<POPULATION_SIZE; i++){
        // Выбор родителей пропорционально приспособленности
        int p1, p2; // Номер 1-го и 2-го родителя соответственно
        Selection(&p1, &p2);
        Crossover(p1, p2, i);   // Кроссовер (скрещивание)
        //        int p;
        //        Selection(&p);
        //        Mutation(Offsprings[i], Parents[p]);    // Мутация
        //        if(random_event_gen(P_MUTATION))
        Mutation(*Offsprings[i], *Offsprings[i]);    // Мутация
        // Оценить приспособленность потомка
        CalcCromosomeFitness(Offsprings[i]);
        if(Offsprings[i]->fitness() > AvgFitness)
            ChildAvgCount++;
    }
    cerr << "               Max Fitness: " << Parents[0]->fitness() << endl;
    cerr << "           Average Fitness: " << AvgFitness << endl;
    cerr << "  ChildFitness > Avg Count: " << ChildAvgCount << endl;
    cerr << "             Mutation Rate: " << nu4all << endl;
    cerr << "-------------------------------------------------------" << endl;

    ChildAvgCountSum += ChildAvgCount;
    if(iter % 10 == 0){
        if(ChildAvgCountSum / 10 > POPULATION_SIZE/5)
            nu4all /= 0.9;
        else if(ChildAvgCountSum / 10 < POPULATION_SIZE/5)
            nu4all *= 0.9;
        ChildAvgCountSum = 0;
    }
    // Формирование новой популяции
    AvgFitness = NewGenerationBest();

    // Сохранить наиболее приспособленную особь
    for(size_t j = 0; j < vectorSize(); j++)
        *mParameters[j] = (*Parents[0])[j];

    return(Parents[0]->fitness());
}

// Выбор родителей методом рулетки - пропорционально их приспособленности
void TGeneticAlgorithm::RoulettePreparation()
{
    GP[0] = 0.0;
    for(int i = 0; i < POPULATION_SIZE; i++)
    {
        // Вычислить координаты концов интервалов
        double NewFitness = (Parents[i]->fitness() > AvgFitness) ?
                    Parents[i]->fitness() * (1.0 - P_LOWFITNESS) :
                    Parents[i]->fitness() * P_LOWFITNESS;
        GP[i+1] = GP[i] + NewFitness;
    }
}

void TGeneticAlgorithm::Selection(int *p1, int *p2)
{
    double Coord1, Coord2;
    // (выбор 1-го родителя)
    Coord1 = (double)rand()/(double)(RAND_MAX)*GP[POPULATION_SIZE-1];
    *p1 = 0;
    while(GP[(*p1)++] < Coord1);
    (*p1)--;
    if(p2 == NULL)
        return;
    // (выбор 2-го родителя)
    do{ // (не должен совпадать с номером 1-го)
        Coord2 = (double)rand()/(double)(RAND_MAX)*GP[POPULATION_SIZE-1];
        *p2 = 0;
        while(GP[(*p2)++] < Coord2);
        (*p2)--;
    }while(*p2 == *p1);
}

double TGeneticAlgorithm::NewGenerationBest()
{
    // Сортировка особей алгоритмом быстрой сортировки QuickSort
    QSort(AllPopulation, 0, POPULATION_SIZE*2-1);

    // Отбор лучших в новую популяцию
    AvgFitness = 0.0;
    for(int i = 0; i < POPULATION_SIZE; i++)
    {
        Parents[i] = AllPopulation[i];
        AvgFitness += Parents[i]->fitness();
        Offsprings[i] = AllPopulation[i+POPULATION_SIZE];
    }
    AvgFitness /= POPULATION_SIZE;
    return(AvgFitness);
}

double TGeneticAlgorithm::NewGenerationFullSubstitution()
{
    // Сортировка потомков алгоритмом быстрой сортировки QuickSort
    // Полная смена популяции
    AvgFitness = 0.0;
    for(int i = 0; i < POPULATION_SIZE; i++)
    {
        Parents[i] = AllPopulation[i+POPULATION_SIZE];
        AvgFitness += Parents[i]->fitness();
        Offsprings[i] = AllPopulation[i];

        AllPopulation[i] = Parents[i];
        AllPopulation[i+POPULATION_SIZE] = Offsprings[i];
    }
    AvgFitness /= POPULATION_SIZE;
    //    QSort(AllPopulation, 0, POPULATION_SIZE-1);
    return(AvgFitness);
}

// Алгоритм быстрой сортировки по убывающей (реализация для типа TChromosome)
void TGeneticAlgorithm::QSort(TChromosome **array,
                              int First, int Last)
{
    int i,j;
    TChromosome *Median, *Swap;

    i=First;
    j=Last;
    Median= array[(First+Last)/2];
    do {
        while (array[i]->fitness() > Median->fitness()) i++;
        while (Median->fitness() > array[j]->fitness()) j--;
        if (i<=j)
        {
            Swap = array[i];
            array[i++] = array[j];
            array[j--] = Swap;
        }
    } while (i<=j);
    if(First < j) QSort(array, First, j);
    if(i < Last) QSort(array, i, Last);
}

double TGeneticAlgorithm::cauchy_gen()
{
    double u, v;
    do {
        u = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
        v = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
    } while(u*u + v*v > 1.0 || (u == 0.0 && v == 0.0));

    if(u!=0)
        return(v/u);
    else
        return(0.0);
}

bool TGeneticAlgorithm::random_event_gen(double P)
{
    return(rand()/(double)(RAND_MAX) < P);
}
