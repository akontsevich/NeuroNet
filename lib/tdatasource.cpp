/*!
    @file
    @brief Contains TDataSource class implementation

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/

#include <algorithm>

#include "tdatasource.h"

double TDataSource::data(int row, int col)
{
    return internalData((this->*mIndexFunc)(row), col);
}

double TDataSource::data(int row, string colName)
{
    return internalData((this->*mIndexFunc)(row), colName);
}

double TDataSource::inData(int row, int col)
{
    int colIndex = (!mInIndexes.size()) ? col : mInIndexes[col];
    return data(row, colIndex);
}

double TDataSource::outData(int row, int col)
{
    int colIndex = (!mOutIndexes.size()) ? col : mOutIndexes[col];
    return data(row, colIndex);
}

void TDataSource::shuffle(bool mix)
{
    if(mix) {
        if(!mShuffled) {
            mShuffled = true;
            mIndexFunc = &TDataSource::shuffledIndex;
            mIndexArray = vector<int>(rowCount());
            // sequentally populate vector
            iota(mIndexArray.begin(), mIndexArray.end(), 0);
            // shuffle vector
            random_shuffle(begin(mIndexArray), end(mIndexArray));
        }
        // already shuffled - do nothing
    } else {
        mShuffled = true;
        mIndexFunc = &TDataSource::shuffledIndex;
    }
}
