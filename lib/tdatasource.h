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
#include  <vector>
#include <unordered_map>
#include <stdexcept>

using namespace std;

/*!
 * \brief The TDataSource abstract class containing data set for neuro net learning
 */
class TDataSource
{
public:
    TDataSource() {}

    class TDataRow  ///< Proxy class to implement operator[][]
    {
    public:
        TDataRow(TDataSource& source, int row) : ds(source), i(row) {}
        double operator[](int j)       { return ds.data(i, j);    }
        double operator[](string name) { return ds.data(i, name); }

    private:
        TDataSource& ds;
        int i;
    };

    TDataRow operator [](int i)
    {
        return TDataRow(*this, i);
    }

    virtual size_t rowCount() = 0;     ///< Data source row count
    virtual size_t colCount() = 0;     ///< Data source column count

    /// < Min column value by column index
    virtual int min(int col) throw(out_of_range) final;
    /// < Max column value by column index
    virtual int max(int col) throw(out_of_range) final;
    /// < Min column value by column name
    virtual int min(string colName) throw(invalid_argument) final;
    /// < Min column value by column name
    virtual int max(string colName) throw(invalid_argument) final;

    /*!
     * \brief columnName gets column name by its number
     * \param col column number
     * \return column name or empty string if not exist
     */
    virtual string columnName(int col) throw() = 0;
    virtual const vector<string>& columnNames() const throw() = 0;
    /*!
     * \brief columnIndex gets column number by its name
     * \param name column name
     * \return column number or -1 if not exist
     */
    virtual int columnIndex(string name) throw() = 0;

protected:
    virtual double data(int row, int col) = 0;
    virtual double data(int row, string colName) = 0;

private:
    void calcColumnMinMax(uint col);
    bool columnInRange(uint col) { return (/*0 <= col && */col < colCount()); }
    bool rowInRange(uint row)    { return (/*0 <= row && */row < rowCount()); }

    unordered_map<int, double> minByColIdx, maxByColIdx;
    unordered_map<string, double> minByColName, maxByColName;
};

#endif // TDATASOURCE_H
