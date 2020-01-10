//  Project AC-SpGEMM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2018 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Martin Winter - martin.winter (at) icg.tugraz.at
//              Daniel Mlakar - daniel.mlakar (at) icg.tugraz.at
//              Rhaleb Zayer - rzayer (at) mpi-inf.mpg.de
//              Hans-Peter Seidel - hpseidel (at) mpi-inf.mpg.de
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#pragma once

#include <cstddef>
#include <algorithm>

template<typename T>
struct CSR;

template<typename T>
struct dCSR
{
	size_t rows, cols, nnz;

	T* data;
	unsigned int* row_offsets;
	unsigned int* col_ids;

	dCSR() : rows(0), cols(0), nnz(0), data(nullptr), row_offsets(nullptr), col_ids(nullptr) { }
	void alloc(size_t rows, size_t cols, size_t nnz, bool allocOffsets = true);
	void reset();
	~dCSR();
};

template<typename T>
void convert(dCSR<T>& dcsr, const CSR<T>& csr, unsigned int padding = 0);

template<typename T>
void convert(dCSR<T>& dcsr, const dCSR<T>& csr, unsigned int padding = 0);

template<typename T>
void convert(CSR<T>& csr, const dCSR<T>& dcsr, unsigned int padding = 0);

template<typename T>
void convert(CSR<T>& csr, const CSR<T>& dcsr, unsigned int padding = 0);