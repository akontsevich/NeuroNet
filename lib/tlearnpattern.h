/*!
    @file
    @brief Contains TLearnPattern class definition

    This file is part of NeuroNet - simple but very fast C++
    back propagation neural network library

    @author Copyright (C) 1999-2017 Aleksey Kontsevich <akontsevich@gmail.com>

    @copyright This Source Code Form is subject to the terms of GNU LESSER GENERAL
    PUBLIC LICENSE Version 3. If a copy of the LGPLv3 was not distributed with
    this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.en.html
*/
#ifndef TLEARNPATTERN_H
#define TLEARNPATTERN_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include "tneuronet.h"

using namespace std;

class TDataSource;

/*!
 * \brief The TLearnPattern class contains normalized data for neuro net learning
 */
class TLearnPattern
{
public:
    typedef int (TLearnPattern::* IndexFunction)(int);
    typedef double (TLearnPattern::* NormalizationFunction)(double X, int col, MinMax norm);
    typedef double (TLearnPattern::* DataFunction)(int, int);

    enum NormalizationType
    {
        Simple,
        Sigmoid,
        Logarithm
    };

    TLearnPattern() {}
    TLearnPattern(TDataSource *source);

    class TPatternRow  ///< Proxy class to implement operator[][]
    {
    public:
        TPatternRow(TLearnPattern& source, int row, DataFunction func)
            : i(row)
            , lp(source)
            , data(func)
        {}
        double operator[](int j) { return (lp.*data)(i, j); }

    private:
        int i;
        TLearnPattern& lp;
        DataFunction data = &TLearnPattern::in;
    };

    TPatternRow operator [](int row) { return TPatternRow(*this, row, &TLearnPattern::in); }
    TPatternRow In(int row)  { return TPatternRow(*this, row, &TLearnPattern::in); }
    TPatternRow Out(int row) { return TPatternRow(*this, row, &TLearnPattern::out); }
    TPatternRow TestIn(int row)  { return TPatternRow(*this, row, &TLearnPattern::testIn); }
    TPatternRow TestOut(int row) { return TPatternRow(*this, row, &TLearnPattern::testOut); }

    ///< Neuro net input (normalized values)
    virtual double in(int row, int col);
    ///< Neuro net desired output (normalized values)
    virtual double out(int row, int col);

    ///< Neuro net input (normalized values)
    virtual double testIn(int row, int col);
    ///< Neuro net desired output (normalized values)
    virtual double testOut(int row, int col);

    virtual size_t rowCount();     ///< Learn pattern row count
    virtual size_t testCount();    ///< Test pattern row count

    size_t inCount()   { return mInIndexes.size(); }
    size_t outCount()  { return mOutIndexes.size(); }

    /*!
     * \brief mapInOut method maps neuro net input data coumns and
     *        desired output data coumns
     * \param inCols input columns numbers
     * \param outCols output columns numbers
     */
    void mapInOut(vector<int> &inCols, vector<int> &outCols) throw(out_of_range);
    void mapNormalizationRanges(TNeuroNet *net);

    /*!
     * \brief setTestData define test data set
     * \param Nth set every N-th element as test data
     */
    void setTestData(uint Nth);
    /*!
     * \brief setTestData define test data set
     * \param testData set external data source as test data
     * \throw invalid_argument if test data does not match source
     */
    void setTestData(TDataSource *testData) throw(invalid_argument);

    /*!
     * \brief setNormalizationType sets data normalization function type
     * \param type mormalization type
     */
    void setNormalizationType(NormalizationType type);
    double RestoreInValue(double nValue, int col);
    double RestoreOutValue(double nValue, int col);

    void shuffle(bool mix = true);  ///< Randomly shuffle data set

private:
    ///< Simple data normalization
    double SimpleNormalization(double X, int ColNumber, MinMax norm);
    ///< Sigmoid data normalization
    double SigmoidNormalization(double X, int, MinMax norm);
    ///< Logarithmic data normalization
    double LogarithmicNormalization(double X, int ColNumber, MinMax norm);

    double RestoreValue(double nValue, int col, MinMax norm);

    int fullIndex(int idx)      { return idx; }
    int partIndex(int idx)      { return mPartIndex[idx]; }
    int testIndex(int idx)      { return mTestIndex[idx]; }
    int shuffledIndex(int idx)  { return mShuffledIndex[(this->*mSequentIndexRowFunc)(idx)]; }

    void clearRowIndexes();    ///< Clears row indexes vectors
    void clearColIndexes();    ///< Clears column indexes vectors

    vector<int> mInIndexes, mOutIndexes;
    // Columns min/max values
    vector<double> mMin, mMax;
    // Normalized min/max values for neuro net input/output
    MinMax mNetInMinMax, mNetOutMinMax;
    NormalizationType normalizationType = Simple;
    NormalizationFunction normalizationFunc = &TLearnPattern::SimpleNormalization;

                  ///< Could be fullIndex() or shuffledIndex() only
    IndexFunction mRowIndexFunc = &TLearnPattern::fullIndex,
                  ///< Could be fullIndex() or partIndex() only
                  mSequentIndexRowFunc = &TLearnPattern::fullIndex,
                  ///< Could be fullIndex() or testIndex()
                  mTestIndexFunc = &TLearnPattern::fullIndex;

    bool mShuffled = false;
    vector<int> mShuffledIndex,
                mPartIndex, // data index if N-th element test set defined
                mTestIndex; // test data index if N-th element test set defined
    TDataSource *mSource,               // Source data
                *mTestData = nullptr;   // External test data
    int nTest = -1;
};

#endif // TLEARNPATTERN_H
