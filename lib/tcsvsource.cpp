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

#include <string>
#include <fstream>      // std::ifstream
#include <algorithm>

#include "tcsvsource.h"

TCSVSource::TCSVSource(const string &fileName,
                       const string &delimiters,
                       bool firstRowColumnsNames) throw(invalid_argument, out_of_range)
    : TDataSource()
{
    ifstream csv(fileName);
    if(!csv.is_open())  throw invalid_argument("Invalid file name, can't open the file!");

    string line;
    vector<string> columns;
    vector<double> values;
    if(!getline(csv, line)) return; // do nothing file is empty
    try {
        columns = split(line, delimiters);
        if(firstRowColumnsNames)
        {
            mColumnsNames = columns;
            for(size_t i = 0; i < mColumnsNames.size(); i++)
                mColumnsIndexes[mColumnsNames[i]] = i;
        }
        else
        {
            // Initialize by empty strings vector
            mColumnsNames = vector<string>(columns.size());
            values = stringToDouble(columns);
            mData.push_back(values);
        }

        while(std::getline(csv, line))
        {
            if (line.empty()) continue;     // Skip empty lines

            columns = split(line, delimiters);
            values = stringToDouble(columns);
            if(values.size() == mColumnsNames.size())
                mData.push_back(values);
            else
            {
                string errorLine = to_string(mData.size() + 1);
                clearData();
                throw invalid_argument("Wrong file format in line: " + errorLine +
                                       "- columns count do not match "
                                       "in \"" + fileName + "\" file");
            }
        }
    } catch (const std::invalid_argument&) {
        string errorLine = to_string(mData.size() + 1);
        clearData();
        throw invalid_argument("Invalid double in " + errorLine + " line of "
                               "\"" + fileName + "\" file");
    } catch (const std::out_of_range&) {
        string errorLine = to_string(mData.size() + 1);
        clearData();
        throw out_of_range("Double out of range in " + errorLine + " line of "
                           "\"" + fileName + "\" file");
    }
}

vector<double>
TCSVSource::stringToDouble(vector<string> stringVector) throw(invalid_argument,
                                                              out_of_range)
{
    vector<double> doubleVector(stringVector.size());
    transform(stringVector.begin(), stringVector.end(), doubleVector.begin(),
              [](string const& val) { return stod(val); });
    return doubleVector;
}

vector<string> TCSVSource::split(const string& str, const string &delimiters)
{
    vector<string> tokens;

    // Start at the beginning
    string::size_type lastPos = 0;
    // Find position of the first delimiter
    string::size_type pos = str.find_first_of(delimiters, lastPos);

    // While we still have string to read
    while (string::npos != pos && string::npos != lastPos)
    {
        // Found a token, add it to the vector
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Look at the next token instead of skipping delimiters
        lastPos = pos + 1;
        // Find the position of the next delimiter
        pos = str.find_first_of(delimiters, lastPos);
    }

    // Push the last token
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    return tokens;
}
