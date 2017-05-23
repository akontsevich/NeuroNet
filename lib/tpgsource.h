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
#ifndef TPGSOURCE_H
#define TPGSOURCE_H

#include <pqxx/pqxx>
#include "tdatasource.h"

using namespace std;
using namespace pqxx;

class TPgSource : public TDataSource
{
public:
    /*!
     * \brief TPgSource
     * \param connStr PostgreSQL database connection string
     * \param query SQL query to PostgreSQL database
     */
    TPgSource(string connectionString, string query) throw(sql_error, exception);

    ///< Data source row count
    virtual size_t rowCount() { return mRes.size(); }
    ///< Data source column count
    virtual size_t colCount() { return mRes.columns(); }

    virtual string columnName(int col) throw()   { return mRes.column_name(col); }
    virtual const vector<string>& columnNames() const throw() { return mColumns; }
    virtual int columnIndex(string name) throw() { return mRes.column_number(name); }

protected:
    virtual double data(int row, int col) { return mRes[row][col].as(double()); }
    virtual double data(int row, string colName) { return mRes[row][colName].as(double());}

private:
    result mRes;
    vector<string> mColumns;
};

#endif // TPGSOURCE_H
