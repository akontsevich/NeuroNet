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
#include <stdexcept>

using namespace std;

class TDataSource;

/*!
 * \brief The TLearnPattern class contains normalized data for neuro net learning
 */
class TLearnPattern
{
public:
    typedef int (TLearnPattern::* IndexFunction)(int);
    typedef double (TLearnPattern::* DataFunction)(int, int);

    TLearnPattern() {}
    TLearnPattern(TDataSource *source);

    class TPatternRow  ///< Proxy class to implement operator[][]
    {
    public:
        TPatternRow(TLearnPattern& source, int row, DataFunction func) :
            i(row),
            lp(source),
            data(func)
        {}
        double operator[](int j) { return (lp.*data)(i, j); }

    private:
        int i;
        TLearnPattern& lp;
        DataFunction data = &TLearnPattern::in;
    };

    TPatternRow operator [](int i) { return TPatternRow(*this, i, &TLearnPattern::in); }
    TPatternRow In(int i)  { return TPatternRow(*this, i, &TLearnPattern::in); }
    TPatternRow Out(int i) { return TPatternRow(*this, i, &TLearnPattern::out); }
    TPatternRow TestIn(int i)  { return TPatternRow(*this, i, &TLearnPattern::testIn); }
    TPatternRow TestOut(int i) { return TPatternRow(*this, i, &TLearnPattern::testOut); }

    ///< Neuro net input (normalized values)
    virtual double in(int row, int col) final;
    ///< Neuro net desired output (normalized values)
    virtual double out(int row, int col) final;

    ///< Neuro net input (normalized values)
    virtual double testIn(int row, int col) final;
    ///< Neuro net desired output (normalized values)
    virtual double testOut(int row, int col) final;

    virtual int rowCount();     ///< Learn pattern row count
    virtual int testCount();    ///< Test pattern row count

    /*!
     * \brief mapInOut method maps neuro net input data coumns and
     *        desired output data coumns
     * \param inCols input columns numbers
     * \param outCols output columns numbers
     */
    void mapInOut(vector<int> &inCols, vector<int> &outCols) throw(invalid_argument);
    /*!
     * \brief setTestData define test data set
     * \param Nth set every N-th element as test data
     */
    void setTestData(int Nth);
    /*!
     * \brief setTestData define test data set
     * \param testData set external data source as test data
     */
    void setTestData(TDataSource *testData);

    void shuffle(bool mix = true);  ///< Randomly shuffle data set

private:
    int fullIndex(int idx)      { return idx; }
    int partIndex(int idx)      { return mPartIndex[idx]; }
    int testIndex(int idx)      { return mTestIndex[idx]; }
    int shuffledIndex(int idx)  { return mShuffledIndex[(this->*mSequentIndexRowFunc)(idx)]; }

    void clearIndexes();    ///< Clears row indexes vectors

    vector<int> mInIndexes, mOutIndexes;

                  ///< Could be fullIndex or shuffledIndex only
    IndexFunction mRowIndexFunc = &TLearnPattern::fullIndex,
                  ///< Could be fullIndex or partIndex only
                  mSequentIndexRowFunc = &TLearnPattern::fullIndex;

    bool mShuffled = false;
    vector<int> mShuffledIndex,
                mPartIndex, // data index if N-th element test set defined
                mTestIndex; // test data index if N-th element test set defined
    TDataSource *mSource,               // Source data
                *mTestData = nullptr;   // External test data
    int nTest = -1;
};

#endif // TLEARNPATTERN_H
