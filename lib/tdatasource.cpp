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

#include "tdatasource.h"

int TDataSource::min(int col) throw(out_of_range)
{
    if(!columnInRange(col)) throw out_of_range("Column No. out of range");
    if(!minByColIdx.count(col))
        calcColumnMinMax(col);  // Calculate on first run only
    return minByColIdx[col];
}

int TDataSource::max(int col) throw(out_of_range)
{
    if(!columnInRange(col)) throw out_of_range("Column No. out of range");
    if(!maxByColIdx.count(col))
        calcColumnMinMax(col);  // Calculate on first run only
    return maxByColIdx[col];
}

int TDataSource::min(string colName) throw(invalid_argument)
{
    int col = columnIndex(colName);
    if(!columnInRange(col)) throw invalid_argument("Such column does not exists!");
    if(!minByColName.count(colName))
    {
        if(!minByColIdx.count(col))
            calcColumnMinMax(col);  // Calculate on first run only
        minByColName[colName] = minByColIdx[col];
    }
    return minByColName[colName];
}

int TDataSource::max(string colName) throw(invalid_argument)
{
    int col = columnIndex(colName);
    if(!columnInRange(col)) throw invalid_argument("Such column does not exists!");
    if(!maxByColName.count(colName))
    {
        if(!maxByColIdx.count(col))
            calcColumnMinMax(col);  // Calculate on first run only
        maxByColName[colName] = maxByColIdx[col];
    }
    return maxByColName[colName];
}

void TDataSource::calcColumnMinMax(uint col)
{
    if(rowCount() <= 0) return;

    double min = data(0, col), max = data(0, col);
    for(size_t i = 1; i < rowCount(); i++)
    {
        if(min > data(i, col)) min = data(i, col);
        if(max < data(i, col)) max = data(i, col);
    }
    minByColIdx[col] = min;
    maxByColIdx[col] = max;
}
