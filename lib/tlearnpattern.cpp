/*!
    @file
    @brief Contains TLearnPattern class implementation

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/

#include <algorithm>

#include "tdatasource.h"
#include "tlearnpattern.h"

TLearnPattern::TLearnPattern(TDataSource *source) : mSource(source)
{
    mMin = vector<double>(mSource->colCount());
    mMax = vector<double>(mSource->colCount());

    // Store min/max locally
    for(size_t i = 0; i < mSource->colCount(); i++)
    {
        mMin[i] = mSource->min(i);
        mMax[i] = mSource->max(i);
    }
}

void TLearnPattern::mapInOut(vector<int> &inCols,
                             vector<int> &outCols) throw(out_of_range)
{
    mInIndexes = vector<int>(inCols.size());
    mOutIndexes = vector<int>(outCols.size());

    // Map input columns
    for(size_t i = 0; i < inCols.size(); i++)
    {
        if(inCols[i] >= 0 && inCols[i] < mSource->colCount())
            mInIndexes[i] = inCols[i];
        else
        {
            clearColIndexes();
            throw out_of_range("Input column No. is out of range");
        }
    }

    // Map output columns
    for(size_t i = 0; i < outCols.size(); i++)
    {
        if(outCols[i] >= 0 && outCols[i] < mSource->colCount())
            mOutIndexes[i] = outCols[i];
        else
        {
            clearColIndexes();
            throw out_of_range("Output column No. is out of range");
        }
    }
}

double TLearnPattern::in(int row, int col)
{
    int rowIndex = (this->*mRowIndexFunc)(row);
    int colIndex = (!mInIndexes.size()) ? col : mInIndexes[col];
    double value = (*mSource)[rowIndex][colIndex];
    // Normalized value
    return (this->*normalizationFunc)(value, colIndex, mNetInMinMax);
}

double TLearnPattern::out(int row, int col)
{
    int rowIndex = (this->*mRowIndexFunc)(row);
    int colIndex = (!mOutIndexes.size()) ? col : mOutIndexes[col];
    double value = (*mSource)[rowIndex][colIndex];
    // Normalized value
    return (this->*normalizationFunc)(value, colIndex, mNetOutMinMax);
}

double TLearnPattern::testIn(int row, int col)
{
    int rowIndex = (this->*mTestIndexFunc)(row);
    int colIndex = (!mInIndexes.size()) ? col : mInIndexes[col];
    double value = (nTest > 0) ? (*mSource)[rowIndex][colIndex] :
                                 (*mTestData)[rowIndex][colIndex];
    // Normalized value
    return (this->*normalizationFunc)(value, colIndex, mNetInMinMax);
}

double TLearnPattern::testOut(int row, int col)
{
    int rowIndex = (this->*mTestIndexFunc)(row);
    int colIndex = (!mOutIndexes.size()) ? col : mOutIndexes[col];
    double value = (nTest > 0) ? (*mSource)[rowIndex][colIndex] :
                                 (*mTestData)[rowIndex][colIndex];
    // Normalized value
    return (this->*normalizationFunc)(value, colIndex, mNetOutMinMax);
}

void TLearnPattern::mapNormalizationRanges(TNeuroNet *net)
{
    // TNeuroNet returns NMin, NMax according to activation function
    // range of definition (input/output)
    mNetInMinMax = net->inMinMax();
    mNetOutMinMax = net->outMinMax();
}

void TLearnPattern::setNormalizationType(NormalizationType type)
{
    normalizationType = type;
    switch (normalizationType) {
    default:
    case Simple:
        normalizationType = Simple;
        normalizationFunc = &TLearnPattern::SimpleNormalization;
        break;
    case Sigmoid:
        normalizationFunc = &TLearnPattern::SigmoidNormalization;
        break;
    case Logarithm:
        normalizationFunc = &TLearnPattern::LogarithmicNormalization;
        break;
    }
}

double TLearnPattern::SimpleNormalization(double X, int col, MinMax norm)
{
    return((X - mMin[col]) / (mMax[col] - mMin[col]) *
           (norm.Max() - norm.Min()) + norm.Min());
}

double TLearnPattern::SigmoidNormalization(double X, int /*col*/, MinMax norm)
{
    return(1.0 / (1.0 + exp(-X)) * (norm.Max() - norm.Min()) + norm.Min());
}

double TLearnPattern::LogarithmicNormalization(double X, int col, MinMax norm)
{
    double klog = 1.0 / floor(log10(mMax[col] - mMin[col]) + 1);
    return(klog * log10(X - mMin[col] + 1.0) * (norm.Max() - norm.Min()) + norm.Min());
}

double TLearnPattern::RestoreInValue(double nValue, int col)
{
    return RestoreValue(nValue, col, mNetInMinMax);
}

double TLearnPattern::RestoreOutValue(double nValue, int col)
{
    return RestoreValue(nValue, col, mNetOutMinMax);
}

double TLearnPattern::RestoreValue(double nValue, int col, MinMax norm)
{
    double value = (nValue - norm.Min()) / (norm.Max() - norm.Min());
    switch(normalizationType)
    {
    case Simple:
        value = value * (mMax[col] - mMin[col]) + mMin[col];
        break;
    case Sigmoid:
        value = log(value / (1.0 - value));
        break;
    case Logarithm:
        double klog = 1.0 / floor(log10(mMax[col] - mMin[col]) + 1);
        value = pow(10.0, nValue / klog);
        value += mMin[col] - 1.0;
        break;
    }
    return (value);
}

void TLearnPattern::setTestData(uint Nth)
{
    nTest = Nth;
    mTestData = nullptr;
    mSequentIndexRowFunc = &TLearnPattern::partIndex;
    mTestIndexFunc = &TLearnPattern::testIndex;

    clearRowIndexes();     // Clear indexes to rebuild them below

    // Generate mPartIndex and mTestIndex
    for(size_t i = 0; i < rowCount(); i++)
    {
        if(i % nTest != 0)
            mPartIndex.push_back(i);
        else
            mTestIndex.push_back(i);
    }

    shuffle(mShuffled); // reshuffle if necessary
}

void TLearnPattern::setTestData(TDataSource *testData) throw(invalid_argument)
{
    if(mSource->colCount() != testData->colCount())
        throw invalid_argument("Test data does not match source data");

    nTest = -1;
    mTestData = testData;
    mSequentIndexRowFunc = &TLearnPattern::fullIndex;
    mTestIndexFunc = &TLearnPattern::fullIndex;

    clearRowIndexes();     // Clear indexes as unnecessary here

    // Adjust min/max according to test data
    for(size_t i = 0; i < mSource->colCount(); i++)
    {
        if(mMin[i] > mTestData->min(i)) mMin[i] = mTestData->min(i);
        if(mMax[i] < mTestData->max(i)) mMax[i] = mTestData->max(i);
    }

    shuffle(mShuffled); // reshuffle if necessary
}

size_t TLearnPattern::rowCount()
{
    return (nTest > 0) ? mPartIndex.size() : mSource->rowCount();
}

size_t TLearnPattern::testCount()
{
    return (nTest > 0) ? mTestIndex.size() : mTestData->rowCount();
}

void TLearnPattern::shuffle(bool mix)
{
    if(mix) {
        mShuffled = true;
        mRowIndexFunc = &TLearnPattern::shuffledIndex;

        mShuffledIndex = vector<int>(rowCount());
        // sequentally populate vector
        iota(mShuffledIndex.begin(), mShuffledIndex.end(), 0);
        // shuffle vector
        random_shuffle(begin(mShuffledIndex), end(mShuffledIndex));
    } else {
        mShuffled = false;
        mRowIndexFunc = (nTest > 0) ? &TLearnPattern::partIndex :
                                      &TLearnPattern::fullIndex;
    }
}

void TLearnPattern::clearRowIndexes()
{
    mPartIndex.clear();
    mTestIndex.clear();
    mShuffledIndex.clear();
}

void TLearnPattern::clearColIndexes()
{
    mInIndexes.clear();
    mOutIndexes.clear();
}
