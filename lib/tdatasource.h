/*!
    @file
    @brief Contains TDataSource class definition

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/

#ifndef TDATASOURCE_H
#define TDATASOURCE_H

#include <string>
#include <vector>
#include <stdexcept>

using namespace std;

class TDataSource
{
public:
    typedef int (TDataSource::* IndexFunction)(int);

    TDataSource() {}

    virtual double data(int row, int col) final;
    virtual double data(int row, string colName) final;

    ///< Neuro net input (normalized values)
    virtual double in(int row, int col) final;
    ///< Neuro net desired output (normalized values)
    virtual double out(int row, int col) final;

    virtual int rowCount() = 0;     ///< Data source row count
    virtual int colCount() = 0;     ///< Data source column count

    virtual int min(int col) = 0;   /// < Min column value by column index
    virtual int max(int col) = 0;   /// < Max column value by column index
    virtual int min(string colName) = 0;  /// < Min column value by column name
    virtual int max(string colName) = 0;  /// < Max column value by column name

    void setInOut(vector<int> &inCols, vector<int> &outCols) throw(invalid_argument);

    void shuffle(bool mix = true);  ///< Randomly shuffle data set

protected:
    virtual double internalData(int row, int col) = 0;
    virtual double internalData(int row, string colName) = 0;

private:
    int sequentIndex(int idx)   { return idx; }
    int shuffledIndex(int idx)  { return mIndexArray[idx]; }

    vector<int> mInIndexes, mOutIndexes;

    IndexFunction mIndexFunc = &TDataSource::sequentIndex;
    bool mShuffled = false;
    vector<int> mIndexArray;
};

#endif // TDATASOURCE_H
