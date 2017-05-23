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

#include "tpgsource.h"

TPgSource::TPgSource(string connectionString, string query) throw(sql_error,
                                                                  exception)
{
    try {
        connection conn(connectionString);
        work T(conn, "DataQueryTransaction");
        mRes = T.exec(query);

        // Read columns names
        for(tuple_size_type i = 0; i < mRes.columns(); i++)
            mColumns.push_back(mRes.column_name(i));
    } catch(sql_error &e) {
        throw e;
    } catch(exception &e) {
        throw e;
    }
}
