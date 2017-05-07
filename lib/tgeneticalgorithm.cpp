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
    for(int j=0; j<Size; j++)
        data[j] = *(vector[j]);
}

void TChromosome::fillVectorData(double *vector)
{
    for(int j=0; j<Size; j++)
        data[j] = vector[j];
}

TChromosome::TChromosome(int n)
{
    Size = n;
    data = new double [Size];
    al = new double [Size];
    si = new double [Size];
    for(int j=0; j<Size; j++){
        al[j] = BETA;
        si[j] = NU_INIT;
    }
    tau = 1.0 / sqrt(2.0*sqrt((double)n));
    tau1 = 1.0 / sqrt(2.0 * (double)n);
}

double TChromosome::geneValue(int j)
{
    return(data[j]);
}

TChromosome::~TChromosome()
{
    delete[] data;
    delete[] al;
    delete[] si;
}

void TChromosome::Print()
{
    printf("(%5.3f", data[0]);
    for(int j=1; j<Size; j++)
        printf(", %5.3f", data[j]);
    printf(")\n");
}

TGeneticAlgorithm::TGeneticAlgorithm(int vector_size)
{
    // Создание хромосом-родителей
    Parents = new TChromosome *[POPULATION_SIZE];
    // Создание хромосом-потомков
    Offsprings = new TChromosome *[POPULATION_SIZE];

    AllPopulation = new TChromosome *[2*POPULATION_SIZE];

    for(int i=0; i<POPULATION_SIZE; i++){
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

TGeneticAlgorithm::~TGeneticAlgorithm()
{
    for(int i=0; i<POPULATION_SIZE; i++)
    {
        delete Parents[i];
        delete Offsprings[i];
    }
    delete[] Parents;
    delete[] Offsprings;
    delete[] AllPopulation;

    delete[] vector;
    delete[] GP;
}

void TGeneticAlgorithm::Copy(TChromosome *Dest, TChromosome *Source)
{
    for(int j=0; j<vector_size; j++)
        Dest->data[j] = Source->data[j];
    Dest->Fitness = Source->Fitness;
}

void TGeneticAlgorithm::Crossover(int pn1, int pn2, int Child)
{
    // Конкатенация компонентов векторов родителей
//    int CrossoverPoint = (int)((double)rand()/(double)(RAND_MAX)*vector_size);
    for(int j=0; j<vector_size; j++)
        // одноточечное скрещивание
//        if(j < CrossoverPoint)
        // многоточечное скрещивание
//        if(j%2 == 0)
        // Случайный обмен хромосомами
        if(random_event_gen(0.5))
            Offsprings[Child]->data[j] = Parents[pn1]->data[j];
        else
            Offsprings[Child]->data[j] = Parents[pn2]->data[j];
        // Арифметическое среднее
//        Offsprings[Child]->data[j] = 0.5*(Parents[pn1]->data[j] +
//                                          Parents[pn2]->data[j]);
}

void TGeneticAlgorithm::Mutation(TChromosome *Offspring, TChromosome *Parent)
{
    for(int j=0; j<vector_size; j++){
        if(random_event_gen(0.5))
            Offspring->data[j] = Parent->data[j] + nu4all * cauchy_gen();
    }
}

void TGeneticAlgorithm::Mutation1(TChromosome *Offspring, TChromosome *Parent)
{
    //double gauss = gauss_gen();
    //int jj;
    for(int j=0; j<vector_size; j++){
//        si[j] = si[j] * exp(tau1*gauss + tau*gauss_gen());
    }
}

// Вызывается из функции обучения ИНС
// Критерий останова пока является макс количество циклов
double TGeneticAlgorithm::OptimizationCycle(int iter)
{
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
        Mutation(Offsprings[i], Offsprings[i]);    // Мутация
        // Оценить приспособленность потомка
        CalcCromosomeFitness(Offsprings[i]);
        if(Offsprings[i]->Fitness > AvgFitness)
            ChildAvgCount++;
    }
    cerr << "               Max Fitness: " << Parents[0]->Fitness << endl;
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
    AvgFitness = NewGeneration1();

    // Сохранить наиболее приспособленную особь
    for(int j=0; j<vector_size; j++)
        *vector[j] = Parents[0]->geneValue(j);

    return(Parents[0]->Fitness);
}

// Выбор родителей методом рулетки - пропорционально их приспособленности
void TGeneticAlgorithm::RoulettePreparation()
{
    GP[0] = 0.0;
    for(int i=0; i<POPULATION_SIZE; i++)
    {
        // Вычислить координаты концов интервалов
        double NewFitness = (Parents[i]->Fitness > AvgFitness) ?
                            Parents[i]->Fitness * (1.0 - P_LOWFITNESS) :
                            Parents[i]->Fitness * P_LOWFITNESS;
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

double TGeneticAlgorithm::NewGeneration1()
{
    // Сортировка особей алгоритмом быстрой сортировки QuickSort
    QSort(AllPopulation, 0, POPULATION_SIZE*2-1);
    // Отбор лучших в новую популяцию
    AvgFitness = 0.0;
    for(int i=0; i<POPULATION_SIZE; i++)
    {
        Parents[i] = AllPopulation[i];
        AvgFitness += Parents[i]->Fitness;
        Offsprings[i] = AllPopulation[i+POPULATION_SIZE];
    }
    AvgFitness /= POPULATION_SIZE;
    return(AvgFitness);
}

double TGeneticAlgorithm::NewGeneration2()
{
    // Сортировка потомков алгоритмом быстрой сортировки QuickSort
    // Полная смена популяции
    AvgFitness = 0.0;
    for(int i=0; i<POPULATION_SIZE; i++)
    {
        Parents[i] = AllPopulation[i+POPULATION_SIZE];
        AvgFitness += Parents[i]->Fitness;
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
  do
  {
    while (array[i]->Fitness > Median->Fitness) i++;
    while (Median->Fitness > array[j]->Fitness) j--;
    if (i<=j)
    {
      Swap = array[i];
      array[i++] = array[j];
      array[j--] = Swap;
    }
  }
  while (i<=j);
  if (First < j) QSort(array, First, j);
  if (i < Last) QSort(array, i, Last);
}

