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

}

double TLearnPattern::in(int row, int col)
{
    int rowIndex = (this->*mRowIndexFunc)(row);
    int colIndex = (!mInIndexes.size()) ? col : mInIndexes[col];
    return (*mSource)[rowIndex][colIndex];
}

double TLearnPattern::out(int row, int col)
{
    int rowIndex = (this->*mRowIndexFunc)(row);
    int colIndex = (!mOutIndexes.size()) ? col : mOutIndexes[col];
    return (*mSource)[rowIndex][colIndex];
}

void TLearnPattern::setTestData(int Nth)
{
    nTest = Nth;
    mTestData = nullptr;
    mSequentIndexRowFunc = &TLearnPattern::partIndex;

    clearIndexes();     // Clear indexes to rebuild them below

    // TODO: calc mPartIndex and mTestIndex
    //...

    // TODO: adjust min/max according to test data
    // ...

    shuffle(mShuffled); // reshuffle if necessary
}

void TLearnPattern::setTestData(TDataSource *testData)
{
    nTest = -1;
    mTestData = testData;
    mSequentIndexRowFunc = &TLearnPattern::fullIndex;

    clearIndexes();     // Clear indexes as unnecessary here

    shuffle(mShuffled); // reshuffle if necessary
}

int TLearnPattern::rowCount()
{
    return (nTest > 0) ? mPartIndex.size() : mSource->rowCount();
}

int TLearnPattern::testCount()
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

void TLearnPattern::clearIndexes()
{
    mPartIndex.clear();
    mTestIndex.clear();
    mShuffledIndex.clear();
}
