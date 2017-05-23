/*!
    @file
    @brief Contains  class definition/implementation

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/
#ifndef TCSVSOURCE_H
#define TCSVSOURCE_H

#include <map>
#include <vector>
#include <stdexcept>

#include "tdatasource.h"

using namespace std;

/*!
 * \brief The TCSVSource class TDataSource child for text file implementation
 */
class TCSVSource : public TDataSource
{
public:
    /*!
     * \brief TCSVSource constructs class instance from
     * \param fileName CSV file
     * \param delimiters
     * \param firstRowColumnsNames whether first row contains columns names
     */
    TCSVSource(const string &fileName,
               const string &delimiters = " \t",
               bool firstRowColumnsNames = true) throw(invalid_argument, out_of_range);

    ///< Data source row count
    virtual size_t rowCount() { return mData.size(); }
    ///< Data source column count
    virtual size_t colCount() { return mData[0].size(); }

    virtual string columnName(int col) throw()   { return mColumnsNames[col];}
    virtual const vector<string>& columnNames() const throw() { return mColumnsNames; }
    virtual int columnIndex(string name) throw() { return mColumnsIndexes[name]; }

protected:
    virtual double data(int row, int col) { return mData[row][col]; }
    virtual double data(int row, string colName) {
        return mData[row][mColumnsIndexes[colName]];
    }

private:
    vector<double>
    stringToDouble(vector<string> stringVector) throw(invalid_argument, out_of_range);
    vector<string> split(const string& str, const string &delimiters = " \t");
    void clearData() { mData.clear(); mColumnsNames.clear(); mColumnsIndexes.clear(); }

    vector<vector<double>> mData;
    vector<string> mColumnsNames;
    map<string, int> mColumnsIndexes;
};

#endif // TCSVSOURCE_H
