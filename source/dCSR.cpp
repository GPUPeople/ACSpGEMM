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

#include "dCSR.h"
#include "CSR.h"

#include <cuda_runtime.h>

namespace
{
	template<typename T>
	void dealloc(dCSR<T>& mat)
	{
		if (mat.col_ids != nullptr)
			cudaFree(mat.col_ids);
		if (mat.data != nullptr)
			cudaFree(mat.data);
		if (mat.row_offsets != nullptr)
			cudaFree(mat.row_offsets);
		mat.col_ids = nullptr;
		mat.data = nullptr;
		mat.row_offsets = nullptr;
	}
}

template<typename T>
void dCSR<T>::alloc(size_t r, size_t c, size_t n, bool allocOffsets)
{
	dealloc(*this);
	rows = r;
	cols = c;
	nnz = n;
	cudaMalloc(&data, sizeof(T)*n);
	cudaMalloc(&col_ids, sizeof(unsigned int)*n);
	if (allocOffsets)
		cudaMalloc(&row_offsets, sizeof(unsigned int)*(r+1));
}
template<typename T>
dCSR<T>::~dCSR()
{
	dealloc(*this);
}

template<typename T>
void dCSR<T>::reset()
{
	dealloc(*this);
}


template<typename T>
void convert(dCSR<T>& dst, const CSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8*padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data, &src.data[0], src.nnz * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.col_ids, &src.col_ids[0], src.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.row_offsets, &src.row_offsets[0], (src.rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

	if (padding)
	{
		cudaMemset(dst.data + src.nnz, 0, 8 * padding * sizeof(T));
		cudaMemset(dst.col_ids + src.nnz, 0, 8 * padding * sizeof(unsigned int));
		cudaMemset(dst.row_offsets + src.rows + 1, 0, padding * sizeof(unsigned int));
	}
}

template<typename T>
void convert(CSR<T>& dst, const dCSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data.get(), src.data, dst.nnz * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst.col_ids.get(), src.col_ids, dst.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst.row_offsets.get(), src.row_offsets, (dst.rows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

template<typename T>
void convert(dCSR<T>& dst, const dCSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data, src.data, dst.nnz * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dst.col_ids, src.col_ids, dst.nnz * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dst.row_offsets, src.row_offsets, (dst.rows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
}

template<typename T>
void convert(CSR<T>& dst, const CSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	memcpy(dst.data.get(), src.data.get(), dst.nnz * sizeof(T));
	memcpy(dst.col_ids.get(), src.col_ids.get(), dst.nnz * sizeof(unsigned int));
	memcpy(dst.row_offsets.get(), src.row_offsets.get(), (dst.rows + 1) * sizeof(unsigned int));
}

template void dCSR<float>::alloc(size_t r, size_t c, size_t n, bool allocOffsets);
template void dCSR<double>::alloc(size_t r, size_t c, size_t n, bool allocOffsets);

template dCSR<float>::~dCSR();
template dCSR<double>::~dCSR();

template void dCSR<float>::reset();
template void dCSR<double>::reset();

template void convert(dCSR<float>& dcsr, const CSR<float>& csr, unsigned int);
template void convert(dCSR<double>& dcsr, const CSR<double>& csr, unsigned int);

template void convert(CSR<float>& csr, const dCSR<float>& dcsr, unsigned int padding);
template void convert(CSR<double>& csr, const dCSR<double>& dcsr, unsigned int padding);

template void convert(dCSR<float>& dcsr, const dCSR<float>& csr, unsigned int);
template void convert(dCSR<double>& dcsr, const dCSR<double>& csr, unsigned int);

template void convert(CSR<float>& csr, const CSR<float>& dcsr, unsigned int padding);
template void convert(CSR<double>& csr, const CSR<double>& dcsr, unsigned int padding);